import * as ImagePicker from "expo-image-picker";
import React, { useCallback, useState } from "react";
import {
	ActivityIndicator,
	Image,
	SafeAreaView,
	ScrollView,
	StyleSheet,
	Text,
	TouchableOpacity,
	View,
} from "react-native";
import { loadImage } from "react-native-nitro-image";
import MODEL from "../../assets/model/snail_detector_model.tflite";
import { useBundledTensorflowModel } from "../../hooks/use-bundled-tensorflow-model";
import { decodeDetections } from "../../utils/detection-model";

// ─── Constants ────────────────────────────────────────────────────────────────
const CONFIDENCE_THRESHOLD = 0.45;
const DEFAULT_INPUT_SIZE = 320;

// ─── Types ────────────────────────────────────────────────────────────────────

/** Subset of nitro-image's PixelFormat – mirrored locally to avoid a missing re-export. */
type PixelFormat =
	| "ARGB"
	| "BGRA"
	| "ABGR"
	| "RGBA"
	| "XRGB"
	| "BGRX"
	| "XBGR"
	| "RGBX"
	| "RGB"
	| "BGR"
	| "unknown";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Detection {
	xmin: number;
	ymin: number;
	xmax: number;
	ymax: number;
	score: number;
}

interface LetterboxTransform {
	scale: number;
	padX: number;
	padY: number;
	inputWidth: number;
	inputHeight: number;
	sourceWidth: number;
	sourceHeight: number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Converts raw pixel data (any nitro-image PixelFormat) to a letterboxed,
 * normalised Float32Array suitable for an RGB TFLite input tensor [H, W, 3].
 */
function rawToRgbFloat(
	data: Uint8Array,
	srcW: number,
	srcH: number,
	pixelFormat: PixelFormat,
	dstW: number,
	dstH: number,
): { tensor: Float32Array; transform: LetterboxTransform } {
	let rOff: number, gOff: number, bOff: number, bpp: number;
	switch (pixelFormat) {
		case "BGRA":
			bpp = 4;
			rOff = 2;
			gOff = 1;
			bOff = 0;
			break;
		case "RGBA":
			bpp = 4;
			rOff = 0;
			gOff = 1;
			bOff = 2;
			break;
		case "ARGB":
			bpp = 4;
			rOff = 1;
			gOff = 2;
			bOff = 3;
			break;
		case "ABGR":
			bpp = 4;
			rOff = 3;
			gOff = 2;
			bOff = 1;
			break;
		case "BGRX":
			bpp = 4;
			rOff = 2;
			gOff = 1;
			bOff = 0;
			break;
		case "RGBX":
			bpp = 4;
			rOff = 0;
			gOff = 1;
			bOff = 2;
			break;
		case "XRGB":
			bpp = 4;
			rOff = 1;
			gOff = 2;
			bOff = 3;
			break;
		case "XBGR":
			bpp = 4;
			rOff = 3;
			gOff = 2;
			bOff = 1;
			break;
		case "RGB":
			bpp = 3;
			rOff = 0;
			gOff = 1;
			bOff = 2;
			break;
		case "BGR":
			bpp = 3;
			rOff = 2;
			gOff = 1;
			bOff = 0;
			break;
		default:
			bpp = 4;
			rOff = 0;
			gOff = 1;
			bOff = 2;
	}

	const out = new Float32Array(dstH * dstW * 3);
	const scale = Math.min(dstW / srcW, dstH / srcH);
	const scaledWidth = srcW * scale;
	const scaledHeight = srcH * scale;
	const padX = (dstW - scaledWidth) / 2;
	const padY = (dstH - scaledHeight) / 2;

	for (let y = 0; y < dstH; y++) {
		for (let x = 0; x < dstW; x++) {
			const sx = (x + 0.5 - padX) / scale - 0.5;
			const sy = (y + 0.5 - padY) / scale - 0.5;
			if (sx < -0.5 || sy < -0.5 || sx > srcW - 0.5 || sy > srcH - 0.5) {
				continue;
			}
			const x0 = Math.floor(sx);
			const y0 = Math.floor(sy);
			const x1 = Math.min(x0 + 1, srcW - 1);
			const y1 = Math.min(y0 + 1, srcH - 1);
			const dx = sx - x0;
			const dy = sy - y0;

			const i00 = (y0 * srcW + x0) * bpp;
			const i10 = (y0 * srcW + x1) * bpp;
			const i01 = (y1 * srcW + x0) * bpp;
			const i11 = (y1 * srcW + x1) * bpp;
			const oi = (y * dstW + x) * 3;

			for (let [c, co] of [
				[0, rOff],
				[1, gOff],
				[2, bOff],
			] as [number, number][]) {
				const v =
					(1 - dx) * (1 - dy) * data[i00 + co] +
					dx * (1 - dy) * data[i10 + co] +
					(1 - dx) * dy * data[i01 + co] +
					dx * dy * data[i11 + co];
				out[oi + c] = v / 255.0;
			}
		}
	}

	return {
		tensor: out,
		transform: {
			scale,
			padX,
			padY,
			inputWidth: dstW,
			inputHeight: dstH,
			sourceWidth: srcW,
			sourceHeight: srcH,
		},
	};
}

function remapDetectionsFromLetterbox(
	detections: Detection[],
	transform: LetterboxTransform,
): Detection[] {
	return detections.flatMap((detection) => {
		const xmin =
			(detection.xmin * transform.inputWidth - transform.padX) /
			transform.scale /
			transform.sourceWidth;
		const ymin =
			(detection.ymin * transform.inputHeight - transform.padY) /
			transform.scale /
			transform.sourceHeight;
		const xmax =
			(detection.xmax * transform.inputWidth - transform.padX) /
			transform.scale /
			transform.sourceWidth;
		const ymax =
			(detection.ymax * transform.inputHeight - transform.padY) /
			transform.scale /
			transform.sourceHeight;

		const clamped = {
			...detection,
			xmin: Math.max(0, Math.min(1, xmin)),
			ymin: Math.max(0, Math.min(1, ymin)),
			xmax: Math.max(0, Math.min(1, xmax)),
			ymax: Math.max(0, Math.min(1, ymax)),
		};

		if (clamped.xmax <= clamped.xmin || clamped.ymax <= clamped.ymin) {
			return [];
		}

		return [clamped];
	});
}

/**
 * Compute the rendered image rect inside a container using "contain" sizing.
 * Returns { x, y, w, h } in container-relative pixels.
 */
function containRect(
	imgW: number,
	imgH: number,
	ctnW: number,
	ctnH: number,
): { x: number; y: number; w: number; h: number } {
	const imgAR = imgW / imgH;
	const ctnAR = ctnW / ctnH;
	let w: number, h: number;
	if (imgAR > ctnAR) {
		w = ctnW;
		h = ctnW / imgAR;
	} else {
		h = ctnH;
		w = ctnH * imgAR;
	}
	return { x: (ctnW - w) / 2, y: (ctnH - h) / 2, w, h };
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Upload() {
	const plugin = useBundledTensorflowModel(MODEL, []);

	const [imageUri, setImageUri] = useState<string | null>(null);
	const [imageSize, setImageSize] = useState({ w: 1, h: 1 });
	const [detections, setDetections] = useState<Detection[]>([]);
	const [inferring, setInferring] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [containerSize, setContainerSize] = useState({ w: 1, h: 1 });

	const runInference = useCallback(
		async (uri: string, imgW: number, imgH: number) => {
			if (plugin.state !== "loaded" || plugin.model == null) return;
			setInferring(true);
			setError(null);

			try {
				// Load raw pixel data from the selected image.
				const img = await loadImage({ filePath: uri });
				const rawPixelData = await img.toRawPixelDataAsync();

				const inputH =
					plugin.model.inputs[0]?.shape[1] ?? DEFAULT_INPUT_SIZE;
				const inputW =
					plugin.model.inputs[0]?.shape[2] ?? DEFAULT_INPUT_SIZE;

				const src = new Uint8Array(rawPixelData.buffer);
				const { tensor, transform } = rawToRgbFloat(
					src,
					rawPixelData.width,
					rawPixelData.height,
					rawPixelData.pixelFormat,
					inputW,
					inputH,
				);

				const outputs = await plugin.model.run([
					tensor.buffer as ArrayBuffer,
				]);

				setDetections(
					remapDetectionsFromLetterbox(
						decodeDetections(
							plugin.model,
							outputs,
							CONFIDENCE_THRESHOLD,
						),
						transform,
					),
				);
			} catch (err) {
				console.error("[Upload] Inference error:", err);
				setError("Analysis failed. Please try a different image.");
				setDetections([]);
			} finally {
				setInferring(false);
			}
		},
		[plugin],
	);

	const pickImage = useCallback(async () => {
		const result = await ImagePicker.launchImageLibraryAsync({
			mediaTypes: ["images"],
			quality: 1,
		});
		if (result.canceled || result.assets.length === 0) return;

		const asset = result.assets[0];
		setImageUri(asset.uri);
		setImageSize({ w: asset.width, h: asset.height });
		setDetections([]);
		await runInference(asset.uri, asset.width, asset.height);
	}, [runInference]);

	// Compute the rendered rect for the "contain"-mode image.
	const rect = containRect(
		imageSize.w,
		imageSize.h,
		containerSize.w,
		containerSize.h,
	);

	const isReady = plugin.state === "loaded";

	return (
		<SafeAreaView style={styles.screen}>
			<ScrollView
				contentContainerStyle={styles.scroll}
				showsVerticalScrollIndicator={false}
			>
				{/* Header */}
				<View style={styles.header}>
					<Text style={styles.title}>Photo Analysis</Text>
					<Text style={styles.subtitle}>
						Select a photo to scan for snail egg clusters
					</Text>
				</View>

				{/* Image container + bounding-box overlay */}
				<View
					style={styles.imageContainer}
					onLayout={(e) => {
						const { width, height } = e.nativeEvent.layout;
						setContainerSize({ w: width, h: height });
					}}
				>
					{imageUri ? (
						<>
							<Image
								source={{ uri: imageUri }}
								style={styles.image}
								resizeMode="contain"
							/>
							{/* Bounding boxes – positioned relative to the actual rendered image rect */}
							<View
								style={StyleSheet.absoluteFill}
								pointerEvents="none"
							>
								{detections.map((d, i) => (
									<View
										key={i}
										style={[
											styles.bbox,
											{
												left: rect.x + d.xmin * rect.w,
												top: rect.y + d.ymin * rect.h,
												width:
													(d.xmax - d.xmin) * rect.w,
												height:
													(d.ymax - d.ymin) * rect.h,
											},
										]}
									>
										<View style={styles.bboxLabel}>
											<Text style={styles.bboxLabelText}>
												{Math.round(d.score * 100)}%
											</Text>
										</View>
									</View>
								))}
							</View>
							{/* Inference spinner overlay */}
							{inferring && (
								<View style={styles.inferOverlay}>
									<ActivityIndicator
										size="large"
										color="#fff"
									/>
									<Text style={styles.inferText}>
										Analysing…
									</Text>
								</View>
							)}
						</>
					) : (
						<View style={styles.placeholder}>
							<Text style={styles.placeholderIcon}>🔍</Text>
							<Text style={styles.placeholderText}>
								No image selected
							</Text>
						</View>
					)}
				</View>

				{/* Count card */}
				<View style={styles.countRow}>
					<View style={styles.countCard}>
						<Text style={styles.countLabel}>
							EGG CLUSTERS DETECTED
						</Text>
						<Text style={styles.countValue}>
							{detections.length}
						</Text>
					</View>
					{detections.length > 0 && (
						<View style={styles.scoresCard}>
							<Text style={styles.countLabel}>
								TOP CONFIDENCE
							</Text>
							<Text style={styles.countValue}>
								{Math.round(
									Math.max(
										...detections.map((d) => d.score),
									) * 100,
								)}
								%
							</Text>
						</View>
					)}
				</View>

				{/* Error message */}
				{error != null && (
					<View style={styles.errorCard}>
						<Text style={styles.errorText}>{error}</Text>
					</View>
				)}

				{/* CTA */}
				<TouchableOpacity
					style={[
						styles.btn,
						(!isReady || inferring) && styles.btnDisabled,
					]}
					onPress={pickImage}
					disabled={!isReady || inferring}
					activeOpacity={0.8}
				>
					{inferring ? (
						<ActivityIndicator color="#000" />
					) : (
						<Text style={styles.btnText}>
							{!isReady
								? "Loading model…"
								: imageUri
									? "Choose Another Photo"
									: "Choose Photo"}
						</Text>
					)}
				</TouchableOpacity>
			</ScrollView>
		</SafeAreaView>
	);
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
	screen: {
		flex: 1,
		backgroundColor: "#0a0a0a",
	},
	scroll: {
		paddingHorizontal: 20,
		paddingBottom: 40,
		gap: 16,
	},

	// Header
	header: {
		paddingTop: 24,
		paddingBottom: 4,
		gap: 4,
	},
	title: {
		color: "#fff",
		fontSize: 24,
		fontWeight: "700",
		letterSpacing: -0.5,
	},
	subtitle: {
		color: "#666",
		fontSize: 14,
		lineHeight: 20,
	},

	// Image area
	imageContainer: {
		width: "100%",
		aspectRatio: 4 / 3,
		backgroundColor: "#111",
		borderRadius: 16,
		overflow: "hidden",
		borderWidth: 1,
		borderColor: "#1e1e1e",
	},
	image: {
		width: "100%",
		height: "100%",
	},
	placeholder: {
		flex: 1,
		alignItems: "center",
		justifyContent: "center",
		gap: 8,
	},
	placeholderIcon: {
		fontSize: 32,
		opacity: 0.4,
	},
	placeholderText: {
		color: "#444",
		fontSize: 14,
	},
	inferOverlay: {
		...StyleSheet.absoluteFillObject,
		backgroundColor: "rgba(0,0,0,0.55)",
		alignItems: "center",
		justifyContent: "center",
		gap: 10,
	},
	inferText: {
		color: "#fff",
		fontSize: 14,
		fontWeight: "500",
	},

	// Bounding boxes
	bbox: {
		position: "absolute",
		borderWidth: 2,
		borderColor: "#00e5ff",
		borderRadius: 4,
	},
	bboxLabel: {
		position: "absolute",
		top: -22,
		left: -1,
		backgroundColor: "#00e5ff",
		paddingHorizontal: 5,
		paddingVertical: 2,
		borderRadius: 3,
	},
	bboxLabelText: {
		color: "#000",
		fontSize: 11,
		fontWeight: "700",
	},

	// Count cards
	countRow: {
		flexDirection: "row",
		gap: 12,
	},
	countCard: {
		flex: 1,
		backgroundColor: "#111",
		borderWidth: 1,
		borderColor: "#1e1e1e",
		borderRadius: 12,
		paddingHorizontal: 16,
		paddingVertical: 14,
		alignItems: "center",
	},
	scoresCard: {
		flex: 1,
		backgroundColor: "#111",
		borderWidth: 1,
		borderColor: "#1e1e1e",
		borderRadius: 12,
		paddingHorizontal: 16,
		paddingVertical: 14,
		alignItems: "center",
	},
	countLabel: {
		color: "rgba(255,255,255,0.45)",
		fontSize: 10,
		fontWeight: "700",
		letterSpacing: 1.2,
		textTransform: "uppercase",
		textAlign: "center",
	},
	countValue: {
		color: "#fff",
		fontSize: 32,
		fontWeight: "700",
		marginTop: 2,
	},

	// Error
	errorCard: {
		backgroundColor: "rgba(255,59,48,0.12)",
		borderWidth: 1,
		borderColor: "rgba(255,59,48,0.3)",
		borderRadius: 10,
		padding: 14,
	},
	errorText: {
		color: "#ff3b30",
		fontSize: 13,
		lineHeight: 18,
	},

	// CTA button
	btn: {
		paddingVertical: 16,
		backgroundColor: "#fff",
		borderRadius: 12,
		alignItems: "center",
		justifyContent: "center",
		minHeight: 52,
	},
	btnDisabled: {
		opacity: 0.35,
	},
	btnText: {
		color: "#000",
		fontSize: 16,
		fontWeight: "600",
	},
});
