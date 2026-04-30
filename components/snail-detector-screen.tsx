import { useIsFocused } from "@react-navigation/native";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
	ActivityIndicator,
	AppState,
	StyleSheet,
	Text,
	TouchableOpacity,
	View,
	type AppStateStatus,
} from "react-native";
import {
	Camera,
	useCameraDevice,
	useCameraPermission,
	useFrameOutput,
	type CameraRef,
	type Frame,
} from "react-native-vision-camera";
import { runOnJS } from "react-native-worklets";
import MODEL from "../assets/model/snail_detector_model.tflite";
import { useBundledTensorflowModel } from "../hooks/use-bundled-tensorflow-model";

// ─── Constants ────────────────────────────────────────────────────────────────
const CONFIDENCE_THRESHOLD = 0.45;
const DEFAULT_INPUT_SIZE = 320;

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Detection {
	xmin: number;
	ymin: number;
	xmax: number;
	ymax: number;
	score: number;
}

const DEFAULT_NMS_THRESHOLD = 0.45;
const MAX_DETECTIONS = 8;
const YOLO_CONFIDENCE_FLOOR = 0.7;

function clamp01(value: number): number {
	"worklet";
	if (!Number.isFinite(value)) return 0;
	if (value < 0) return 0;
	if (value > 1) return 1;
	return value;
}

function sanitizeScore(value: number): number {
	"worklet";
	if (!Number.isFinite(value) || value <= 0) return 0;
	if (value > 1) return 1;
	return value;
}

function intersectionOverUnion(a: Detection, b: Detection): number {
	"worklet";
	const left = a.xmin > b.xmin ? a.xmin : b.xmin;
	const top = a.ymin > b.ymin ? a.ymin : b.ymin;
	const right = a.xmax < b.xmax ? a.xmax : b.xmax;
	const bottom = a.ymax < b.ymax ? a.ymax : b.ymax;

	const width = right - left;
	const height = bottom - top;
	if (width <= 0 || height <= 0) return 0;

	const intersection = width * height;
	const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
	const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
	const union = areaA + areaB - intersection;
	if (union <= 0) return 0;

	return intersection / union;
}

function nonMaxSuppress(
	detections: Detection[],
	iouThreshold = DEFAULT_NMS_THRESHOLD,
	maxDetections = MAX_DETECTIONS,
): Detection[] {
	"worklet";
	if (detections.length <= 1) return detections;

	const sorted = detections.slice().sort((a, b) => b.score - a.score);
	const kept: Detection[] = [];

	outer: for (let index = 0; index < sorted.length; index++) {
		const candidate = sorted[index];
		for (let keptIndex = 0; keptIndex < kept.length; keptIndex++) {
			if (
				intersectionOverUnion(candidate, kept[keptIndex]) > iouThreshold
			) {
				continue outer;
			}
		}

		kept.push(candidate);
		if (kept.length >= maxDetections) break;
	}

	return kept;
}

function decodeSsdLikeDetections(
	outputs: ArrayBuffer[],
	confidenceThreshold: number,
): Detection[] {
	"worklet";
	const rawBoxes = new Float32Array(outputs[0] ?? new ArrayBuffer(0));
	if (rawBoxes.length < 4 || outputs.length < 3) return [];

	const scoreBuffer = outputs.length >= 4 ? outputs[2] : outputs[1];
	const countBuffer = outputs.length >= 4 ? outputs[3] : outputs[2];
	const rawScores = new Float32Array(scoreBuffer ?? new ArrayBuffer(0));
	const rawCount = countBuffer ? new Float32Array(countBuffer) : null;
	const maxCount = Math.floor(rawBoxes.length / 4);
	const count =
		rawCount != null && rawCount.length > 0
			? Math.min(Math.round(rawCount[0] ?? 0), rawScores.length, maxCount)
			: Math.min(rawScores.length, maxCount);

	const detections: Detection[] = [];
	for (let index = 0; index < count; index++) {
		const score = sanitizeScore(rawScores[index] ?? 0);
		if (score < confidenceThreshold) continue;

		const ymin = clamp01(rawBoxes[index * 4 + 0] ?? 0);
		const xmin = clamp01(rawBoxes[index * 4 + 1] ?? 0);
		const ymax = clamp01(rawBoxes[index * 4 + 2] ?? 1);
		const xmax = clamp01(rawBoxes[index * 4 + 3] ?? 1);
		if (xmax <= xmin || ymax <= ymin) continue;

		detections.push({ xmin, ymin, xmax, ymax, score });
	}

	return nonMaxSuppress(detections);
}

function getYoloLayout(shape: number[]) {
	"worklet";
	if (shape.length < 3) return null;

	const dimA = shape[shape.length - 2] ?? 0;
	const dimB = shape[shape.length - 1] ?? 0;
	if (dimA <= 0 || dimB <= 0) return null;

	if (dimA <= dimB) {
		return {
			channelFirst: true,
			channelCount: dimA,
			candidateCount: dimB,
		};
	}

	return {
		channelFirst: false,
		channelCount: dimB,
		candidateCount: dimA,
	};
}

function readYoloValue(
	data: Float32Array,
	layout: {
		channelFirst: boolean;
		channelCount: number;
		candidateCount: number;
	},
	channel: number,
	index: number,
): number {
	"worklet";
	return layout.channelFirst
		? (data[channel * layout.candidateCount + index] ?? 0)
		: (data[index * layout.channelCount + channel] ?? 0);
}

function yoloBoxesAreNormalized(
	data: Float32Array,
	layout: {
		channelFirst: boolean;
		channelCount: number;
		candidateCount: number;
	},
): boolean {
	"worklet";
	const samples = layout.candidateCount < 32 ? layout.candidateCount : 32;
	let maxCoord = 0;

	for (let index = 0; index < samples; index++) {
		for (let channel = 0; channel < 4; channel++) {
			const value = Math.abs(readYoloValue(data, layout, channel, index));
			if (value > maxCoord) maxCoord = value;
		}
	}

	return maxCoord <= 2;
}

function decodeDetections(
	outputs: ArrayBuffer[],
	confidenceThreshold: number,
	inputW: number,
	inputH: number,
	outputShape: number[],
	outputCount: number,
): Detection[] {
	"worklet";
	if (outputCount !== 1) {
		return decodeSsdLikeDetections(outputs, confidenceThreshold);
	}

	const layout = getYoloLayout(outputShape);
	if (layout == null || layout.channelCount < 5) return [];

	const raw = new Float32Array(outputs[0] ?? new ArrayBuffer(0));
	const candidateCount = Math.min(
		layout.candidateCount,
		Math.floor(raw.length / layout.channelCount),
	);
	if (candidateCount <= 0) return [];

	const effectiveThreshold =
		confidenceThreshold > YOLO_CONFIDENCE_FLOOR
			? confidenceThreshold
			: YOLO_CONFIDENCE_FLOOR;
	const normalizedBoxes = yoloBoxesAreNormalized(raw, layout);
	const normW = normalizedBoxes ? 1 : inputW;
	const normH = normalizedBoxes ? 1 : inputH;
	const detections: Detection[] = [];

	for (let index = 0; index < candidateCount; index++) {
		let bestScore = 0;
		for (let channel = 4; channel < layout.channelCount; channel++) {
			const score = sanitizeScore(
				readYoloValue(raw, layout, channel, index),
			);
			if (score > bestScore) bestScore = score;
		}

		if (bestScore < effectiveThreshold) continue;

		const centerX = readYoloValue(raw, layout, 0, index);
		const centerY = readYoloValue(raw, layout, 1, index);
		const width = Math.abs(readYoloValue(raw, layout, 2, index));
		const height = Math.abs(readYoloValue(raw, layout, 3, index));
		if (
			!Number.isFinite(centerX) ||
			!Number.isFinite(centerY) ||
			!Number.isFinite(width) ||
			!Number.isFinite(height)
		) {
			continue;
		}
		if (width <= 0 || height <= 0) continue;

		const xmin = clamp01((centerX - width / 2) / normW);
		const ymin = clamp01((centerY - height / 2) / normH);
		const xmax = clamp01((centerX + width / 2) / normW);
		const ymax = clamp01((centerY + height / 2) / normH);
		if (xmax <= xmin || ymax <= ymin) continue;

		detections.push({ xmin, ymin, xmax, ymax, score: bestScore });
	}

	return nonMaxSuppress(detections);
}

// ─── Helpers (worklet-compatible) ─────────────────────────────────────────────

/**
 * Bilinear-resize a planar YUV (semi-planar NV12/NV21) frame to a
 * normalised RGB Float32Array [dstH × dstW × 3].
 * Runs inside the worklet runtime.
 */
function buildInputTensorYUV(
	yBuf: Uint8Array,
	uvBuf: Uint8Array | null,
	yStride: number,
	uvStride: number,
	srcW: number,
	srcH: number,
	dstW: number,
	dstH: number,
): Float32Array {
	"worklet";
	const out = new Float32Array(dstH * dstW * 3);
	const xScale = srcW / dstW;
	const yScale = srcH / dstH;

	for (let row = 0; row < dstH; row++) {
		for (let col = 0; col < dstW; col++) {
			const sx = Math.min(Math.floor((col + 0.5) * xScale), srcW - 1);
			const sy = Math.min(Math.floor((row + 0.5) * yScale), srcH - 1);

			const yVal = yBuf[sy * yStride + sx] ?? 0;

			let r: number, g: number, b: number;
			if (uvBuf != null && uvBuf.length > 0) {
				// Semi-planar NV12: interleaved [U, V] at half resolution.
				// NV21 ([V, U]) swaps hue slightly but detection still works.
				const uvRow = Math.floor(sy / 2);
				const uvCol = Math.floor(sx / 2);
				const uvIdx = uvRow * uvStride + uvCol * 2;
				const u = (uvBuf[uvIdx] ?? 128) - 128;
				const v = (uvBuf[uvIdx + 1] ?? 128) - 128;
				r = yVal + 1.402 * v;
				g = yVal - 0.344136 * u - 0.714136 * v;
				b = yVal + 1.772 * u;
			} else {
				r = yVal;
				g = yVal;
				b = yVal;
			}

			const oi = (row * dstW + col) * 3;
			out[oi + 0] = (r < 0 ? 0 : r > 255 ? 255 : r) / 255;
			out[oi + 1] = (g < 0 ? 0 : g > 255 ? 255 : g) / 255;
			out[oi + 2] = (b < 0 ? 0 : b > 255 ? 255 : b) / 255;
		}
	}
	return out;
}

/**
 * Bilinear-resize a non-planar (interleaved BGRA/RGBA/RGB) frame buffer
 * to a normalised RGB Float32Array [dstH × dstW × 3].
 * Runs inside the worklet runtime.
 */
function buildInputTensorPacked(
	src: Uint8Array,
	srcW: number,
	srcH: number,
	dstW: number,
	dstH: number,
	pixelFormat: string,
): Float32Array {
	"worklet";
	let rOff: number, gOff: number, bOff: number, bpp: number;
	if (pixelFormat === "rgb-bgra-8-bit") {
		bpp = 4;
		rOff = 2;
		gOff = 1;
		bOff = 0;
	} else if (pixelFormat === "rgb-rgba-8-bit") {
		bpp = 4;
		rOff = 0;
		gOff = 1;
		bOff = 2;
	} else if (pixelFormat === "rgb-rgb-8-bit") {
		bpp = 3;
		rOff = 0;
		gOff = 1;
		bOff = 2;
	} else {
		bpp = 4;
		rOff = 2;
		gOff = 1;
		bOff = 0;
	}

	const out = new Float32Array(dstH * dstW * 3);
	const xScale = srcW / dstW;
	const yScale = srcH / dstH;

	for (let y = 0; y < dstH; y++) {
		for (let x = 0; x < dstW; x++) {
			const sx = (x + 0.5) * xScale - 0.5;
			const sy = (y + 0.5) * yScale - 0.5;
			const x0 = sx < 0 ? 0 : Math.floor(sx);
			const y0 = sy < 0 ? 0 : Math.floor(sy);
			const x1 = x0 + 1 < srcW ? x0 + 1 : srcW - 1;
			const y1 = y0 + 1 < srcH ? y0 + 1 : srcH - 1;
			const dx = sx - x0 < 0 ? 0 : sx - x0;
			const dy = sy - y0 < 0 ? 0 : sy - y0;

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
					(1 - dx) * (1 - dy) * (src[i00 + co] ?? 0) +
					dx * (1 - dy) * (src[i10 + co] ?? 0) +
					(1 - dx) * dy * (src[i01 + co] ?? 0) +
					dx * dy * (src[i11 + co] ?? 0);
				out[oi + c] = v / 255.0;
			}
		}
	}
	return out;
}

function getCoverRect(
	sourceWidth: number,
	sourceHeight: number,
	viewportWidth: number,
	viewportHeight: number,
): { left: number; top: number; width: number; height: number } {
	if (
		sourceWidth <= 0 ||
		sourceHeight <= 0 ||
		viewportWidth <= 0 ||
		viewportHeight <= 0
	) {
		return {
			left: 0,
			top: 0,
			width: viewportWidth,
			height: viewportHeight,
		};
	}

	const scale = Math.max(
		viewportWidth / sourceWidth,
		viewportHeight / sourceHeight,
	);
	const width = sourceWidth * scale;
	const height = sourceHeight * scale;

	return {
		left: (viewportWidth - width) / 2,
		top: (viewportHeight - height) / 2,
		width,
		height,
	};
}

// ─── Component ────────────────────────────────────────────────────────────────

export function SnailDetectorScreen() {
	const { hasPermission, requestPermission } = useCameraPermission();
	const device = useCameraDevice("back");
	const isFocused = useIsFocused();
	const camera = useRef<CameraRef>(null);
	const lastDetectionSignature = useRef("");
	const [appState, setAppState] = useState<AppStateStatus>(
		AppState.currentState,
	);
	const [torch, setTorch] = useState<"on" | "off">("off");
	const [torchAvailable, setTorchAvailable] = useState(false);
	const [detections, setDetections] = useState<Detection[]>([]);
	const [frameSize, setFrameSize] = useState({ width: 1, height: 1 });
	const [layout, setLayout] = useState({ width: 1, height: 1 });
	const isCameraActive = isFocused && appState === "active";

	// Load the TFLite model (CPU delegate by default).
	const plugin = useBundledTensorflowModel(MODEL, []);

	// Called on the JS thread to update detection state.
	const onDetectionsUpdate = useCallback((dets: Detection[]) => {
		const signature = dets
			.map(
				(d) =>
					`${Math.round(d.score * 100)}:${Math.round(d.xmin * 1000)}:${Math.round(d.ymin * 1000)}:${Math.round(d.xmax * 1000)}:${Math.round(d.ymax * 1000)}`,
			)
			.join("|");

		if (signature === lastDetectionSignature.current) return;
		lastDetectionSignature.current = signature;
		setDetections(dets);
	}, []);

	const syncFrameSize = useCallback((width: number, height: number) => {
		setFrameSize((current) => {
			if (current.width === width && current.height === height) {
				return current;
			}

			return { width, height };
		});
	}, []);

	const toggleTorch = useCallback(async () => {
		if (!torchAvailable) return;

		const controller = camera.current?.controller;
		if (controller == null) return;

		const nextTorch = torch === "on" ? "off" : "on";

		try {
			await controller.setTorchMode(nextTorch);
			setTorch(nextTorch);
		} catch (error) {
			setTorch("off");
			setTorchAvailable(false);
			console.warn("Torch is not available on this camera.", error);
		}
	}, [torch, torchAvailable]);

	// Frame processor worklet – runs on the camera's native thread.
	const onFrame = useCallback(
		(frame: Frame) => {
			"worklet";
			if (plugin.state !== "loaded" || plugin.model == null) {
				frame.dispose();
				return;
			}

			try {
				// Read model input shape: [batch, height, width, channels]
				runOnJS(syncFrameSize)(frame.width, frame.height);
				const inputH =
					plugin.model.inputs[0]?.shape[1] ?? DEFAULT_INPUT_SIZE;
				const inputW =
					plugin.model.inputs[0]?.shape[2] ?? DEFAULT_INPUT_SIZE;
				const outputShape = plugin.model.outputs[0]?.shape ?? [];
				const outputCount = plugin.model.outputs.length;

				let tensor: Float32Array;

				if (frame.isPlanar) {
					// YUV semi-planar (NV12/NV21) – the guaranteed Android Camera2 format.
					const planes = frame.getPlanes();
					const yPlane = planes[0];
					const uvPlane = planes[1] ?? null;
					if (yPlane == null) {
						frame.dispose();
						return;
					}
					const yBuf = new Uint8Array(yPlane.getPixelBuffer());
					const uvBuf =
						uvPlane != null
							? new Uint8Array(uvPlane.getPixelBuffer())
							: null;
					// bytesPerRow may include padding; use it as the stride.
					const yStride = yPlane.bytesPerRow;
					const uvStride =
						uvPlane != null ? uvPlane.bytesPerRow : yStride;
					tensor = buildInputTensorYUV(
						yBuf,
						uvBuf,
						yStride,
						uvStride,
						frame.width,
						frame.height,
						inputW,
						inputH,
					);
				} else {
					// Non-planar (BGRA/RGBA/RGB) – fallback for iOS or future formats.
					const src = new Uint8Array(frame.getPixelBuffer());
					tensor = buildInputTensorPacked(
						src,
						frame.width,
						frame.height,
						inputW,
						inputH,
						frame.pixelFormat,
					);
				}
				// Run synchronous inference.
				const outputs = plugin.model.runSync([
					tensor.buffer as ArrayBuffer,
				]);

				runOnJS(onDetectionsUpdate)(
					decodeDetections(
						outputs,
						CONFIDENCE_THRESHOLD,
						inputW,
						inputH,
						outputShape,
						outputCount,
					),
				);
			} catch {
				// Swallow errors to keep the frame pipeline running.
			} finally {
				frame.dispose();
			}
		},
		[plugin, onDetectionsUpdate, syncFrameSize],
	);

	const previewRect = getCoverRect(
		frameSize.width,
		frameSize.height,
		layout.width,
		layout.height,
	);

	const frameOutput = useFrameOutput({
		onFrame,
		// 'yuv' maps to YUV_420_888 on Android – the only format guaranteed to
		// work alongside a preview SurfaceTexture by the Camera2 spec.
		// 'rgb' (FLEX_RGBA_8888) is not guaranteed and causes
		// "Failed to apply the stream configuration" on many devices.
		pixelFormat: "yuv",
		// Rotate output buffers so the worklet sees the same upright orientation
		// that the preview displays. Otherwise detections are in sensor-space and
		// the overlay must manually account for frame.orientation.
		enablePhysicalBufferRotation: true,
		// Let the camera pick a native preview-sized resolution.
		// Forcing 320×320 (1:1 AR) also triggers stream config rejection.
		enablePreviewSizedOutputBuffers: true,
		dropFramesWhileBusy: true,
	});

	useEffect(() => {
		if (!hasPermission) requestPermission();
	}, [hasPermission, requestPermission]);

	useEffect(() => {
		const subscription = AppState.addEventListener(
			"change",
			(nextAppState) => {
				setAppState(nextAppState);
			},
		);

		return () => {
			subscription.remove();
		};
	}, []);

	useEffect(() => {
		setTorch("off");
		setTorchAvailable(device?.hasTorch ?? false);
	}, [device?.hasTorch, device?.id]);

	useEffect(() => {
		if (!isCameraActive) {
			setTorch("off");
			setDetections([]);
		}
	}, [isCameraActive]);

	// ── Permission gate ──────────────────────────────────────────────────────────
	if (!hasPermission) {
		return (
			<View style={styles.centered}>
				<Text style={styles.gateTitle}>Camera Access Required</Text>
				<Text style={styles.gateBody}>
					Snail Detector needs your camera to scan for egg clusters.
				</Text>
				<TouchableOpacity
					style={styles.btn}
					onPress={requestPermission}
				>
					<Text style={styles.btnText}>Grant Permission</Text>
				</TouchableOpacity>
			</View>
		);
	}

	if (device == null) {
		return (
			<View style={styles.centered}>
				<Text style={styles.gateTitle}>No Camera Found</Text>
			</View>
		);
	}

	// ── Live camera view ─────────────────────────────────────────────────────────
	return (
		<View
			style={styles.container}
			onLayout={(e) => {
				const { width, height } = e.nativeEvent.layout;
				setLayout({ width, height });
			}}
		>
			{/* Camera preview */}
			{isCameraActive && (
				<Camera
					ref={camera}
					style={StyleSheet.absoluteFill}
					device={device}
					outputs={[frameOutput]}
					isActive={isCameraActive}
					resizeMode="cover"
				/>
			)}

			{/* Bounding-box overlay */}
			<View style={StyleSheet.absoluteFill} pointerEvents="none">
				{detections.map((d, i) => (
					<View
						key={i}
						style={[
							styles.bbox,
							{
								left:
									previewRect.left +
									d.xmin * previewRect.width,
								top:
									previewRect.top +
									d.ymin * previewRect.height,
								width: (d.xmax - d.xmin) * previewRect.width,
								height: (d.ymax - d.ymin) * previewRect.height,
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

			{/* Top HUD – egg count + model loading indicator */}
			<View style={styles.hudTop}>
				<View style={styles.countCard}>
					<Text style={styles.countLabel}>EGG CLUSTERS</Text>
					<Text style={styles.countValue}>{detections.length}</Text>
				</View>
				{plugin.state === "loading" && (
					<View style={styles.loadingChip}>
						<ActivityIndicator size="small" color="#fff" />
						<Text style={styles.loadingText}>Loading model…</Text>
					</View>
				)}
			</View>

			{/* Bottom controls – flashlight toggle */}
			<View style={styles.hudBottom}>
				{torchAvailable && (
					<TouchableOpacity
						style={[
							styles.torchBtn,
							torch === "on" && styles.torchBtnActive,
						]}
						onPress={() => {
							void toggleTorch();
						}}
						activeOpacity={0.75}
					>
						<Text style={styles.torchIcon}>⚡</Text>
						<Text style={styles.torchLabel}>
							{torch === "on" ? "Flash On" : "Flash Off"}
						</Text>
					</TouchableOpacity>
				)}
			</View>
		</View>
	);
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
	container: {
		flex: 1,
		backgroundColor: "#000",
	},
	centered: {
		flex: 1,
		alignItems: "center",
		justifyContent: "center",
		backgroundColor: "#0a0a0a",
		paddingHorizontal: 32,
		gap: 12,
	},
	gateTitle: {
		fontSize: 20,
		fontWeight: "600",
		color: "#fff",
		textAlign: "center",
	},
	gateBody: {
		fontSize: 14,
		color: "#888",
		textAlign: "center",
		lineHeight: 20,
	},
	btn: {
		marginTop: 8,
		paddingHorizontal: 24,
		paddingVertical: 12,
		backgroundColor: "#fff",
		borderRadius: 10,
	},
	btnText: {
		color: "#000",
		fontSize: 15,
		fontWeight: "600",
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

	// HUD – top
	hudTop: {
		position: "absolute",
		top: 0,
		left: 0,
		right: 0,
		paddingTop: 56,
		paddingHorizontal: 20,
		flexDirection: "row",
		alignItems: "flex-start",
		justifyContent: "space-between",
	},
	countCard: {
		backgroundColor: "rgba(0,0,0,0.6)",
		borderWidth: 1,
		borderColor: "rgba(255,255,255,0.12)",
		borderRadius: 12,
		paddingHorizontal: 18,
		paddingVertical: 10,
		alignItems: "center",
		minWidth: 110,
	},
	countLabel: {
		color: "rgba(255,255,255,0.55)",
		fontSize: 10,
		fontWeight: "700",
		letterSpacing: 1.2,
		textTransform: "uppercase",
	},
	countValue: {
		color: "#fff",
		fontSize: 36,
		fontWeight: "700",
		lineHeight: 42,
	},
	loadingChip: {
		flexDirection: "row",
		alignItems: "center",
		gap: 6,
		backgroundColor: "rgba(0,0,0,0.6)",
		borderWidth: 1,
		borderColor: "rgba(255,255,255,0.12)",
		borderRadius: 20,
		paddingHorizontal: 14,
		paddingVertical: 8,
	},
	loadingText: {
		color: "rgba(255,255,255,0.7)",
		fontSize: 12,
		fontWeight: "500",
	},

	// HUD – bottom
	hudBottom: {
		position: "absolute",
		bottom: 0,
		left: 0,
		right: 0,
		paddingBottom: 40,
		paddingHorizontal: 24,
		alignItems: "center",
	},
	torchBtn: {
		flexDirection: "row",
		alignItems: "center",
		gap: 6,
		paddingHorizontal: 20,
		paddingVertical: 12,
		backgroundColor: "rgba(255,255,255,0.12)",
		borderWidth: 1,
		borderColor: "rgba(255,255,255,0.2)",
		borderRadius: 30,
	},
	torchBtnActive: {
		backgroundColor: "rgba(255,230,0,0.2)",
		borderColor: "rgba(255,230,0,0.5)",
	},
	torchIcon: {
		fontSize: 16,
	},
	torchLabel: {
		color: "#fff",
		fontSize: 14,
		fontWeight: "500",
	},
});
