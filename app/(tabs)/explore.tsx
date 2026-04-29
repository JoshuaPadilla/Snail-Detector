import { useEffect, useRef, useState } from "react";
import { LayoutChangeEvent, Platform, StyleSheet, View } from "react-native";
import type { TfliteModel } from "react-native-fast-tflite";
import { scheduleOnRN } from "react-native-worklets";

import { ThemedText } from "@/components/themed-text";
import { ThemedView } from "@/components/themed-view";
import snailModelAsset from "../../assets/model/2.tflite";

type VisionCameraModule = typeof import("react-native-vision-camera");

type TensorDescriptor = {
	dataType: string;
	name: string;
	shape: number[];
};

type ModelSummary = {
	inputs: TensorDescriptor[];
	outputs: TensorDescriptor[];
};

type DetectionBox = {
	classId: number;
	height: number;
	id: string;
	label: string;
	left: number;
	score: number;
	top: number;
	width: number;
};

type FrameDebugInfo = {
	bytesPerRow: number;
	height: number;
	isPlanar: boolean;
	orientation: string;
	pixelFormat: string;
	planeCount: number;
	width: number;
};

type Size = {
	height: number;
	width: number;
};

type Rect = {
	height: number;
	left: number;
	top: number;
	width: number;
};

type PreparedPlanarSource =
	| {
			kind: "bi-planar";
			uvBytesPerRow: number;
			uvHeight: number;
			uvPixelStride: number;
			uvPixels: Uint8Array;
			uvWidth: number;
			yBytesPerRow: number;
			yHeight: number;
			yPixelStride: number;
			yPixels: Uint8Array;
			yWidth: number;
	  }
	| {
			kind: "tri-planar";
			uBytesPerRow: number;
			uHeight: number;
			uPixelStride: number;
			uPixels: Uint8Array;
			uWidth: number;
			vBytesPerRow: number;
			vHeight: number;
			vPixelStride: number;
			vPixels: Uint8Array;
			vWidth: number;
			yBytesPerRow: number;
			yHeight: number;
			yPixelStride: number;
			yPixels: Uint8Array;
			yWidth: number;
	  };

const MODEL_NAME = "SSD MobileNet V1";
const DETECTION_THRESHOLD = 0.55;
const DEFAULT_INPUT_SIZE = 300;

const TENSOR_BYTES_PER_ELEMENT: Partial<
	Record<TensorDescriptor["dataType"], number>
> = {
	bool: 1,
	float16: 2,
	float32: 4,
	float64: 8,
	int16: 2,
	int32: 4,
	int64: 8,
	int8: 1,
	uint8: 1,
	uint16: 2,
	uint32: 4,
	uint64: 8,
};

function formatTensor(tensor: TensorDescriptor) {
	const shape = tensor.shape.length > 0 ? tensor.shape.join(" × ") : "scalar";
	return `${tensor.name || "unnamed"}: ${tensor.dataType} [${shape}]`;
}

function clampUnit(value: number) {
	"worklet";

	if (value < 0) {
		return 0;
	}

	if (value > 1) {
		return 1;
	}

	return value;
}

function clampByte(value: number) {
	"worklet";

	if (value < 0) {
		return 0;
	}

	if (value > 255) {
		return 255;
	}

	return Math.round(value);
}

function getDetectionLabel(classId: number) {
	"worklet";

	if (classId >= 0) {
		return `object ${classId}`;
	}

	return "object";
}

function getModelInputSize(summary: ModelSummary | null) {
	const input = summary?.inputs[0];

	if (input == null || input.shape.length < 3) {
		return { height: DEFAULT_INPUT_SIZE, width: DEFAULT_INPUT_SIZE };
	}

	if (input.shape.length >= 4) {
		return {
			height: input.shape[1] || DEFAULT_INPUT_SIZE,
			width: input.shape[2] || DEFAULT_INPUT_SIZE,
		};
	}

	return {
		height: input.shape[0] || DEFAULT_INPUT_SIZE,
		width: input.shape[1] || DEFAULT_INPUT_SIZE,
	};
}

function getTensorElementCount(shape: number[]) {
	return shape.reduce(
		(count, dimension) => count * Math.max(dimension, 1),
		1,
	);
}

function getTensorByteLength(tensor: TensorDescriptor) {
	const bytesPerElement = TENSOR_BYTES_PER_ELEMENT[tensor.dataType];

	if (bytesPerElement == null) {
		return null;
	}

	return getTensorElementCount(tensor.shape) * bytesPerElement;
}

function makeZeroInputBuffer(tensor: TensorDescriptor) {
	const byteLength = getTensorByteLength(tensor);

	if (byteLength == null) {
		return null;
	}

	return new ArrayBuffer(byteLength);
}

function previewBufferValues(
	dataType: TensorDescriptor["dataType"],
	buffer: ArrayBuffer,
) {
	switch (dataType) {
		case "float32":
			return Array.from(new Float32Array(buffer).subarray(0, 6));
		case "float64":
			return Array.from(new Float64Array(buffer).subarray(0, 6));
		case "int32":
			return Array.from(new Int32Array(buffer).subarray(0, 6));
		case "uint8":
			return Array.from(new Uint8Array(buffer).subarray(0, 12));
		case "int8":
			return Array.from(new Int8Array(buffer).subarray(0, 12));
		default:
			return undefined;
	}
}

function getPixelFormatInfo(pixelFormat: string) {
	"worklet";

	switch (pixelFormat) {
		case "rgb-bgra-8-bit":
			return {
				blueOffset: 0,
				bytesPerPixel: 4,
				greenOffset: 1,
				redOffset: 2,
			};
		case "rgb-rgba-8-bit":
			return {
				blueOffset: 2,
				bytesPerPixel: 4,
				greenOffset: 1,
				redOffset: 0,
			};
		case "rgb-rgb-8-bit":
			return {
				blueOffset: 2,
				bytesPerPixel: 3,
				greenOffset: 1,
				redOffset: 0,
			};
		default:
			throw new Error(`Unsupported frame pixel format: ${pixelFormat}`);
	}
}

function getPlanePixelStride(planeWidth: number, planeBytesPerRow: number) {
	"worklet";

	if (planeWidth <= 0) {
		return 1;
	}

	return Math.max(1, Math.round(planeBytesPerRow / planeWidth));
}

function getFrameDebugInfo(
	frame: VisionCameraModule extends never
		? never
		: import("react-native-vision-camera").Frame,
) {
	"worklet";

	let planeCount = 0;

	if (frame.isPlanar) {
		try {
			planeCount = frame.getPlanes().length;
		} catch {
			planeCount = -1;
		}
	}

	return {
		bytesPerRow: frame.bytesPerRow,
		height: frame.height,
		isPlanar: frame.isPlanar,
		orientation: frame.orientation,
		pixelFormat: frame.pixelFormat,
		planeCount,
		width: frame.width,
	};
}

function convertYuvToRgb(
	y: number,
	u: number,
	v: number,
	isVideoRange: boolean,
) {
	"worklet";

	const centeredU = u - 128;
	const centeredV = v - 128;

	if (isVideoRange) {
		const adjustedY = Math.max(0, y - 16);

		return {
			blue: clampByte(1.164 * adjustedY + 2.018 * centeredU),
			green: clampByte(
				1.164 * adjustedY - 0.391 * centeredU - 0.813 * centeredV,
			),
			red: clampByte(1.164 * adjustedY + 1.596 * centeredV),
		};
	}

	return {
		blue: clampByte(y + 1.772 * centeredU),
		green: clampByte(y - 0.344136 * centeredU - 0.714136 * centeredV),
		red: clampByte(y + 1.402 * centeredV),
	};
}

function preparePlanarSource(
	frame: VisionCameraModule extends never
		? never
		: import("react-native-vision-camera").Frame,
) {
	"worklet";

	const planes = frame.getPlanes();

	if (planes.length < 2) {
		throw new Error(
			`Unsupported planar frame: expected at least 2 planes, received ${planes.length}.`,
		);
	}

	const yPlane = planes[0];
	const yPixels = new Uint8Array(yPlane.getPixelBuffer());
	const yPixelStride = getPlanePixelStride(yPlane.width, yPlane.bytesPerRow);

	if (planes.length === 2) {
		const uvPlane = planes[1];
		const uvPixels = new Uint8Array(uvPlane.getPixelBuffer());
		const uvPixelStride = getPlanePixelStride(
			uvPlane.width,
			uvPlane.bytesPerRow,
		);

		return {
			kind: "bi-planar" as const,
			uvBytesPerRow: uvPlane.bytesPerRow,
			uvHeight: uvPlane.height,
			uvPixelStride,
			uvPixels,
			uvWidth: uvPlane.width,
			yBytesPerRow: yPlane.bytesPerRow,
			yHeight: yPlane.height,
			yPixelStride,
			yPixels,
			yWidth: yPlane.width,
		};
	}

	const uPlane = planes[1];
	const vPlane = planes[2];
	const uPixels = new Uint8Array(uPlane.getPixelBuffer());
	const vPixels = new Uint8Array(vPlane.getPixelBuffer());
	const uPixelStride = getPlanePixelStride(uPlane.width, uPlane.bytesPerRow);
	const vPixelStride = getPlanePixelStride(vPlane.width, vPlane.bytesPerRow);

	return {
		kind: "tri-planar" as const,
		uBytesPerRow: uPlane.bytesPerRow,
		uHeight: uPlane.height,
		uPixelStride,
		uPixels,
		uWidth: uPlane.width,
		vBytesPerRow: vPlane.bytesPerRow,
		vHeight: vPlane.height,
		vPixelStride,
		vPixels,
		vWidth: vPlane.width,
		yBytesPerRow: yPlane.bytesPerRow,
		yHeight: yPlane.height,
		yPixelStride,
		yPixels,
		yWidth: yPlane.width,
	};
}

function getYuvPixel(
	planarSource: PreparedPlanarSource,
	sourceX: number,
	sourceY: number,
) {
	"worklet";

	const yOffset =
		Math.min(planarSource.yHeight - 1, sourceY) *
			planarSource.yBytesPerRow +
		Math.min(planarSource.yWidth - 1, sourceX) * planarSource.yPixelStride;
	const y = planarSource.yPixels[yOffset] ?? 0;
	const chromaX = Math.floor(sourceX / 2);
	const chromaY = Math.floor(sourceY / 2);

	if (planarSource.kind === "bi-planar") {
		const uvOffset =
			Math.min(planarSource.uvHeight - 1, chromaY) *
				planarSource.uvBytesPerRow +
			Math.min(planarSource.uvWidth - 1, chromaX) *
				planarSource.uvPixelStride;

		return {
			u: planarSource.uvPixels[uvOffset] ?? 128,
			v: planarSource.uvPixels[uvOffset + 1] ?? 128,
			y,
		};
	}

	const uOffset =
		Math.min(planarSource.uHeight - 1, chromaY) *
			planarSource.uBytesPerRow +
		Math.min(planarSource.uWidth - 1, chromaX) * planarSource.uPixelStride;
	const vOffset =
		Math.min(planarSource.vHeight - 1, chromaY) *
			planarSource.vBytesPerRow +
		Math.min(planarSource.vWidth - 1, chromaX) * planarSource.vPixelStride;

	return {
		u: planarSource.uPixels[uOffset] ?? 128,
		v: planarSource.vPixels[vOffset] ?? 128,
		y,
	};
}

function getLetterboxLayout(
	sourceWidth: number,
	sourceHeight: number,
	targetWidth: number,
	targetHeight: number,
) {
	"worklet";

	const scale = Math.min(
		targetWidth / sourceWidth,
		targetHeight / sourceHeight,
	);
	const contentWidth = sourceWidth * scale;
	const contentHeight = sourceHeight * scale;

	return {
		contentHeight,
		contentWidth,
		offsetX: (targetWidth - contentWidth) / 2,
		offsetY: (targetHeight - contentHeight) / 2,
		targetHeight,
		targetWidth,
	};
}

function getCoverRect(viewportSize: Size, sourceSize: Size): Rect {
	if (
		viewportSize.width <= 0 ||
		viewportSize.height <= 0 ||
		sourceSize.width <= 0 ||
		sourceSize.height <= 0
	) {
		return {
			height: viewportSize.height,
			left: 0,
			top: 0,
			width: viewportSize.width,
		};
	}

	const scale = Math.max(
		viewportSize.width / sourceSize.width,
		viewportSize.height / sourceSize.height,
	);
	const width = sourceSize.width * scale;
	const height = sourceSize.height * scale;

	return {
		height,
		left: (viewportSize.width - width) / 2,
		top: (viewportSize.height - height) / 2,
		width,
	};
}

function createModelInputBuffer(
	frame: VisionCameraModule extends never
		? never
		: import("react-native-vision-camera").Frame,
	targetWidth: number,
	targetHeight: number,
) {
	"worklet";

	const output = new Uint8Array(targetWidth * targetHeight * 3);
	const projection = getLetterboxLayout(
		frame.width,
		frame.height,
		targetWidth,
		targetHeight,
	);
	const isPlanar = frame.isPlanar;
	const planarSource = isPlanar ? preparePlanarSource(frame) : null;
	const formatInfo =
		!isPlanar && frame.pixelFormat !== "unknown"
			? getPixelFormatInfo(frame.pixelFormat)
			: null;
	const source = formatInfo ? new Uint8Array(frame.getPixelBuffer()) : null;
	const isVideoRange = frame.pixelFormat.includes("video");

	for (let y = 0; y < targetHeight; y += 1) {
		const normalizedY =
			(y + 0.5 - projection.offsetY) / projection.contentHeight;

		if (normalizedY < 0 || normalizedY > 1) {
			continue;
		}

		const sourceY = Math.min(
			frame.height - 1,
			Math.max(0, Math.floor(normalizedY * frame.height)),
		);

		for (let x = 0; x < targetWidth; x += 1) {
			const normalizedX =
				(x + 0.5 - projection.offsetX) / projection.contentWidth;

			if (normalizedX < 0 || normalizedX > 1) {
				continue;
			}

			const sourceX = Math.min(
				frame.width - 1,
				Math.max(0, Math.floor(normalizedX * frame.width)),
			);
			const targetOffset = (y * targetWidth + x) * 3;

			if (isPlanar) {
				if (planarSource == null) {
					return null;
				}

				const {
					u,
					v,
					y: yValue,
				} = getYuvPixel(planarSource, sourceX, sourceY);
				const rgb = convertYuvToRgb(yValue, u, v, isVideoRange);

				output[targetOffset] = rgb.red;
				output[targetOffset + 1] = rgb.green;
				output[targetOffset + 2] = rgb.blue;
				continue;
			}

			if (formatInfo == null || source == null) {
				return null;
			}

			const sourceRowOffset = sourceY * frame.bytesPerRow;
			const sourceOffset =
				sourceRowOffset + sourceX * formatInfo.bytesPerPixel;

			output[targetOffset] =
				source[sourceOffset + formatInfo.redOffset] ?? 0;
			output[targetOffset + 1] =
				source[sourceOffset + formatInfo.greenOffset] ?? 0;
			output[targetOffset + 2] =
				source[sourceOffset + formatInfo.blueOffset] ?? 0;
		}
	}

	return {
		buffer: output.buffer,
		projection,
	};
}

function decodeDetections(
	outputs: ArrayBuffer[],
	threshold: number,
	projection: ReturnType<typeof getLetterboxLayout>,
) {
	"worklet";

	const boxes = new Float32Array(outputs[0] ?? new ArrayBuffer(0));
	const classes = new Float32Array(outputs[1] ?? new ArrayBuffer(0));
	const scores = new Float32Array(outputs[2] ?? new ArrayBuffer(0));
	const counts = new Float32Array(outputs[3] ?? new ArrayBuffer(0));
	const detectionCount = Math.min(
		10,
		Math.max(0, Math.floor(counts[0] ?? 0)),
	);
	const detections: DetectionBox[] = [];

	for (let index = 0; index < detectionCount; index += 1) {
		const score = scores[index] ?? 0;

		if (score < threshold) {
			continue;
		}

		const offset = index * 4;
		const top = clampUnit(boxes[offset] ?? 0);
		const left = clampUnit(boxes[offset + 1] ?? 0);
		const bottom = clampUnit(boxes[offset + 2] ?? 0);
		const right = clampUnit(boxes[offset + 3] ?? 0);
		const mappedTop = clampUnit(
			(top * projection.targetHeight - projection.offsetY) /
				projection.contentHeight,
		);
		const mappedLeft = clampUnit(
			(left * projection.targetWidth - projection.offsetX) /
				projection.contentWidth,
		);
		const mappedBottom = clampUnit(
			(bottom * projection.targetHeight - projection.offsetY) /
				projection.contentHeight,
		);
		const mappedRight = clampUnit(
			(right * projection.targetWidth - projection.offsetX) /
				projection.contentWidth,
		);

		if (mappedBottom <= mappedTop || mappedRight <= mappedLeft) {
			continue;
		}

		detections.push({
			classId: Math.round(classes[index] ?? -1),
			height: clampUnit(mappedBottom - mappedTop),
			id: `${index}-${Math.round(score * 1000)}`,
			label: getDetectionLabel(Math.round(classes[index] ?? -1)),
			left: mappedLeft,
			score,
			top: mappedTop,
			width: clampUnit(mappedRight - mappedLeft),
		});
	}

	return detections;
}

function getWorkletErrorMessage(error: unknown) {
	"worklet";

	if (typeof error === "string") {
		return error;
	}

	if (error != null && typeof error === "object" && "message" in error) {
		const message = (error as { message?: unknown }).message;

		if (typeof message === "string") {
			return message;
		}
	}

	return "Live detection failed.";
}

function StatusView({
	title,
	message,
	detail,
}: {
	title: string;
	message: string;
	detail?: string;
}) {
	return (
		<ThemedView style={styles.centered}>
			<ThemedText type="title" style={styles.title}>
				{title}
			</ThemedText>
			<ThemedText style={styles.message}>{message}</ThemedText>
			{detail ? (
				<ThemedText style={styles.detail}>{detail}</ThemedText>
			) : null}
		</ThemedView>
	);
}

function CameraPreview({ visionCamera }: { visionCamera: VisionCameraModule }) {
	const [detections, setDetections] = useState<DetectionBox[]>([]);
	const [detectorStatus, setDetectorStatus] = useState(
		"Loading detector and camera pipeline.",
	);
	const [model, setModel] = useState<TfliteModel | null>(null);
	const [modelSummary, setModelSummary] = useState<ModelSummary | null>(null);
	const [modelError, setModelError] = useState<string | null>(null);
	const [previewSize, setPreviewSize] = useState<Size>({
		height: 0,
		width: 0,
	});
	const [sourceFrameSize, setSourceFrameSize] = useState<Size>({
		height: 0,
		width: 0,
	});
	const [smokeTestMessage, setSmokeTestMessage] = useState(
		"Model info will appear here after the load finishes.",
	);
	const didStartModelLoad = useRef(false);
	const lastDetectionError = useRef<string | null>(null);
	const lastLoggedDetections = useRef("");
	const lastLoggedFrameInfo = useRef("");

	const { Camera, useCameraDevice, useCameraPermission, useFrameOutput } =
		visionCamera;
	const { hasPermission, requestPermission } = useCameraPermission();
	const device = useCameraDevice("back");
	const inputSize = getModelInputSize(modelSummary);
	const previewRect = getCoverRect(previewSize, sourceFrameSize);

	const applyDetections = (nextDetections: DetectionBox[]) => {
		setDetections(nextDetections);

		if (nextDetections.length === 0) {
			lastLoggedDetections.current = "";
			setDetectorStatus(
				"Scanning the full camera view for possible snail eggs...",
			);
			return;
		}

		const signature = nextDetections
			.map(
				(detection) =>
					`${detection.classId}-${Math.round(detection.score * 100)}-${Math.round(detection.left * 100)}-${Math.round(detection.top * 100)}`,
			)
			.join("|");

		if (signature !== lastLoggedDetections.current) {
			console.log(
				`[SnailModel] ${MODEL_NAME} detections`,
				nextDetections.map((detection) => ({
					box: {
						height: Number(detection.height.toFixed(3)),
						left: Number(detection.left.toFixed(3)),
						top: Number(detection.top.toFixed(3)),
						width: Number(detection.width.toFixed(3)),
					},
					classId: detection.classId,
					label: detection.label,
					score: Number(detection.score.toFixed(3)),
				})),
			);
			lastLoggedDetections.current = signature;
		}

		setDetectorStatus(
			`Detected ${nextDetections.length} object${nextDetections.length === 1 ? "" : "s"} with ${MODEL_NAME}.`,
		);
	};

	const reportFrameInfo = (frameInfo: FrameDebugInfo) => {
		const signature = `${frameInfo.pixelFormat}-${frameInfo.width}x${frameInfo.height}-${frameInfo.planeCount}-${frameInfo.isPlanar}`;

		if (signature === lastLoggedFrameInfo.current) {
			return;
		}

		lastLoggedFrameInfo.current = signature;
		console.log(`[SnailModel] ${MODEL_NAME} frame info`, frameInfo);
	};

	const reportUnsupportedFrame = (frameInfo: FrameDebugInfo) => {
		const signature = `unsupported-${frameInfo.pixelFormat}-${frameInfo.width}x${frameInfo.height}-${frameInfo.planeCount}-${frameInfo.isPlanar}`;

		if (signature === lastLoggedFrameInfo.current) {
			return;
		}

		lastLoggedFrameInfo.current = signature;
		console.warn(
			`[SnailModel] ${MODEL_NAME} skipped a frame with unsupported readable pixel data`,
			frameInfo,
		);
		setDetectorStatus(
			"Camera frames are arriving in an unsupported readable format on this device.",
		);
	};

	const syncSourceFrameSize = (width: number, height: number) => {
		setSourceFrameSize((currentSize) => {
			if (currentSize.width === width && currentSize.height === height) {
				return currentSize;
			}

			return { height, width };
		});
	};

	const reportDetectionIssue = (message: string) => {
		if (lastDetectionError.current === message) {
			return;
		}

		lastDetectionError.current = message;
		console.error("[SnailModel] Live detection failed", message);
		setDetectorStatus(message);
	};

	const handlePreviewLayout = (event: LayoutChangeEvent) => {
		const { height, width } = event.nativeEvent.layout;
		setPreviewSize({ height, width });
	};

	const frameOutput = useFrameOutput({
		allowDeferredStart: true,
		dropFramesWhileBusy: true,
		enablePreviewSizedOutputBuffers: true,
		enablePhysicalBufferRotation: true,
		pixelFormat: "yuv",
		targetResolution: inputSize,
		onFrame(frame) {
			"worklet";

			try {
				scheduleOnRN(syncSourceFrameSize, frame.width, frame.height);
				scheduleOnRN(reportFrameInfo, getFrameDebugInfo(frame));

				if (model == null) {
					return;
				}

				const preparedInput = createModelInputBuffer(
					frame,
					inputSize.width,
					inputSize.height,
				);

				if (preparedInput == null) {
					scheduleOnRN(
						reportUnsupportedFrame,
						getFrameDebugInfo(frame),
					);
					return;
				}

				const { buffer: inputBuffer, projection } = preparedInput;
				const outputs = model.runSync([inputBuffer]);
				const nextDetections = decodeDetections(
					outputs,
					DETECTION_THRESHOLD,
					projection,
				);

				scheduleOnRN(applyDetections, nextDetections);
			} catch (error) {
				scheduleOnRN(
					reportDetectionIssue,
					getWorkletErrorMessage(error),
				);
			} finally {
				frame.dispose();
			}
		},
	});

	useEffect(() => {
		if (!hasPermission) {
			void requestPermission();
		}
	}, [hasPermission, requestPermission]);

	useEffect(() => {
		if (didStartModelLoad.current) {
			return;
		}

		didStartModelLoad.current = true;
		let isActive = true;

		void import("react-native-fast-tflite")
			.then(async ({ loadTensorflowModel }) => {
				const loadedModel = await loadTensorflowModel(
					snailModelAsset,
					[],
				);
				const summary = {
					inputs: loadedModel.inputs.map((tensor) => ({
						dataType: tensor.dataType,
						name: tensor.name,
						shape: tensor.shape,
					})),
					outputs: loadedModel.outputs.map((tensor) => ({
						dataType: tensor.dataType,
						name: tensor.name,
						shape: tensor.shape,
					})),
				};

				if (!isActive) {
					return;
				}

				setModel(loadedModel);
				setModelSummary(summary);
				setModelError(null);
				setSmokeTestMessage(
					"Model loaded. Running one zero-input smoke test. Check the JS console.",
				);
				setDetectorStatus("Running detector smoke test...");

				console.log(`[SnailModel] ${MODEL_NAME} model loaded`, {
					inputs: summary.inputs,
					outputs: summary.outputs,
				});
				console.log(
					"[SnailModel] Learning note: the first input tensor tells you how every camera frame must be resized and encoded before inference.",
				);

				const smokeTestInputs = summary.inputs.map(makeZeroInputBuffer);

				if (smokeTestInputs.some((buffer) => buffer == null)) {
					const unsupportedInputs = summary.inputs.filter(
						(_, index) => smokeTestInputs[index] == null,
					);

					console.warn("[SnailModel] Smoke test skipped", {
						unsupportedInputs,
					});
					setSmokeTestMessage(
						"Model loaded, but the dummy smoke test was skipped because one input tensor uses an unsupported data type.",
					);
					return;
				}

				console.log(
					`[SnailModel] ${MODEL_NAME} zero-input smoke test`,
					{
						inputByteLengths: smokeTestInputs.map(
							(buffer) => buffer!.byteLength,
						),
					},
				);

				try {
					const outputs = await loadedModel.run(
						smokeTestInputs as ArrayBuffer[],
					);

					if (!isActive) {
						return;
					}

					const outputPreview = outputs.map((buffer, index) => {
						const tensor = summary.outputs[index];

						return {
							bufferBytes: buffer.byteLength,
							previewValues: tensor
								? previewBufferValues(tensor.dataType, buffer)
								: undefined,
							tensor: tensor
								? formatTensor(tensor)
								: `output-${index}`,
						};
					});

					console.log(
						`[SnailModel] ${MODEL_NAME} smoke test succeeded`,
						{
							outputs: outputPreview,
						},
					);
					console.log(
						"[SnailModel] Learning note: this smoke test uses all-zero input buffers only to prove the model can execute. Real camera inference needs resized pixel data that matches the input tensor.",
					);
					setSmokeTestMessage(
						"Smoke test passed. Open the JS console to inspect tensor shapes and sample output values.",
					);
					setDetectorStatus(
						"Scanning the full camera view for detected objects...",
					);
				} catch (error: unknown) {
					if (!isActive) {
						return;
					}

					const message =
						error instanceof Error
							? error.message
							: "The dummy smoke test failed.";

					console.error("[SnailModel] Smoke test failed", error);
					setSmokeTestMessage(
						`Smoke test failed. Open the JS console for details. ${message}`,
					);
					setDetectorStatus(
						"Smoke test failed. Live detection may not work. Check the console.",
					);
				}
			})
			.catch((error: unknown) => {
				if (!isActive) {
					return;
				}

				setModelError(
					error instanceof Error
						? error.message
						: "The TFLite model could not be loaded.",
				);
				setSmokeTestMessage(
					"Model loading failed before the smoke test could run.",
				);
				setDetectorStatus(
					"Detector failed to load. Check the error panel.",
				);
			});

		return () => {
			isActive = false;
		};
	}, []);

	if (!hasPermission) {
		return (
			<StatusView
				title="Camera permission needed"
				message="Approve the camera permission prompt to continue."
			/>
		);
	}

	if (device == null) {
		return (
			<StatusView
				title="Camera unavailable"
				message="No back camera is available yet in this runtime."
			/>
		);
	}

	return (
		<View style={styles.previewContainer}>
			<Camera
				style={StyleSheet.absoluteFill}
				isActive
				device={device}
				outputs={[frameOutput]}
				resizeMode="cover"
			/>
			<View
				onLayout={handlePreviewLayout}
				pointerEvents="none"
				style={styles.overlay}
			>
				<View style={styles.detectionLayer}>
					{detections.map((detection) => (
						<View
							key={detection.id}
							style={[
								styles.detectionBox,
								{
									height:
										detection.height * previewRect.height,
									left:
										previewRect.left +
										detection.left * previewRect.width,
									top:
										previewRect.top +
										detection.top * previewRect.height,
									width: detection.width * previewRect.width,
								},
							]}
						>
							<View style={styles.detectionBadge}>
								<ThemedText
									lightColor="#0B1419"
									darkColor="#0B1419"
									style={styles.detectionBadgeText}
								>
									{`${detection.label} ${Math.round(detection.score * 100)}%`}
								</ThemedText>
							</View>
						</View>
					))}
				</View>
				<View style={styles.topHud}>
					<ThemedText
						lightColor="#FFFFFF"
						darkColor="#FFFFFF"
						type="defaultSemiBold"
						style={styles.overlayTitle}
					>
						Open view scanning for detected objects
					</ThemedText>
				</View>
				<View style={styles.bottomHud}>
					<ThemedText
						lightColor="#FFFFFF"
						darkColor="#FFFFFF"
						style={styles.overlayHint}
					>
						Hold steady and watch the console for SSD MobileNet V1
						detection logs.
					</ThemedText>
					{modelSummary ? (
						<View style={styles.modelPanel}>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								type="defaultSemiBold"
								style={styles.modelTitle}
							>
								{MODEL_NAME} ready
							</ThemedText>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								{`Inputs: ${modelSummary.inputs.length} | Outputs: ${modelSummary.outputs.length}`}
							</ThemedText>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								{detectorStatus}
							</ThemedText>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								{smokeTestMessage}
							</ThemedText>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								Live detections are logged with class id, score,
								and box coordinates.
							</ThemedText>
							{modelSummary.inputs[0] ? (
								<ThemedText
									lightColor="#FFFFFF"
									darkColor="#FFFFFF"
									style={styles.modelLine}
								>
									{formatTensor(modelSummary.inputs[0])}
								</ThemedText>
							) : null}
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								Each box shows the detected class id from SSD
								MobileNet V1.
							</ThemedText>
							{modelSummary.outputs[0] ? (
								<ThemedText
									lightColor="#FFFFFF"
									darkColor="#FFFFFF"
									style={styles.modelLine}
								>
									{formatTensor(modelSummary.outputs[0])}
								</ThemedText>
							) : null}
						</View>
					) : modelError ? (
						<View style={styles.modelPanel}>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								type="defaultSemiBold"
								style={styles.modelTitle}
							>
								Model failed to load
							</ThemedText>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								{modelError}
							</ThemedText>
						</View>
					) : (
						<View style={styles.modelPanel}>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								type="defaultSemiBold"
								style={styles.modelTitle}
							>
								Loading model
							</ThemedText>
							<ThemedText
								lightColor="#FFFFFF"
								darkColor="#FFFFFF"
								style={styles.modelLine}
							>
								Trying assets/model/2.tflite in Fast TFLite.
							</ThemedText>
						</View>
					)}
				</View>
			</View>
		</View>
	);
}

export default function TabTwoScreen() {
	const [visionCamera, setVisionCamera] = useState<VisionCameraModule | null>(
		null,
	);
	const [loadError, setLoadError] = useState<string | null>(
		Platform.OS === "web"
			? "Vision Camera only runs in a native Android or iOS development build."
			: null,
	);

	useEffect(() => {
		if (Platform.OS === "web") {
			return;
		}

		let isActive = true;

		void import("react-native-vision-camera")
			.then((module) => {
				if (!isActive) {
					return;
				}

				setVisionCamera(module);
				setLoadError(null);
			})
			.catch((error: unknown) => {
				if (!isActive) {
					return;
				}

				setLoadError(
					error instanceof Error
						? error.message
						: "Vision Camera failed to load in the current runtime.",
				);
			});

		return () => {
			isActive = false;
		};
	}, []);

	if (visionCamera == null) {
		const title =
			loadError == null ? "Loading camera" : "Camera runtime not ready";
		const message =
			loadError == null
				? "Preparing Vision Camera for this screen."
				: "Open this screen from a rebuilt Android or iOS development build.";

		return (
			<StatusView
				title={title}
				message={message}
				detail={loadError ?? undefined}
			/>
		);
	}

	return <CameraPreview visionCamera={visionCamera} />;
}

const styles = StyleSheet.create({
	centered: {
		alignItems: "center",
		flex: 1,
		gap: 12,
		justifyContent: "center",
		paddingHorizontal: 24,
	},
	title: {
		textAlign: "center",
	},
	message: {
		textAlign: "center",
	},
	detail: {
		opacity: 0.7,
		textAlign: "center",
	},
	previewContainer: {
		backgroundColor: "#000000",
		flex: 1,
	},
	overlay: {
		...StyleSheet.absoluteFillObject,
	},
	topHud: {
		alignItems: "center",
		backgroundColor: "rgba(0, 0, 0, 0.48)",
		flex: 1,
		justifyContent: "flex-end",
		paddingBottom: 24,
		paddingHorizontal: 24,
	},
	overlayTitle: {
		textAlign: "center",
	},
	middleRow: {
		flexDirection: "row",
	},
	sideShade: {
		backgroundColor: "rgba(0, 0, 0, 0.48)",
		flex: 1,
	},
	scanFrame: {
		aspectRatio: 1,
		backgroundColor: "rgba(255, 255, 255, 0.04)",
		borderRadius: 28,
		borderWidth: 2,
		maxWidth: 320,
		position: "relative",
		width: "74%",
	},
	frameCorner: {
		borderRadius: 6,
		height: 28,
		position: "absolute",
		width: 28,
	},
	topLeftCorner: {
		borderLeftWidth: 4,
		borderTopWidth: 4,
		left: 14,
		top: 14,
	},
	topRightCorner: {
		borderRightWidth: 4,
		borderTopWidth: 4,
		right: 14,
		top: 14,
	},
	bottomLeftCorner: {
		borderBottomWidth: 4,
		borderLeftWidth: 4,
		bottom: 14,
		left: 14,
	},
	bottomRightCorner: {
		borderBottomWidth: 4,
		borderRightWidth: 4,
		bottom: 14,
		right: 14,
	},
	bottomHud: {
		alignItems: "center",
		backgroundColor: "rgba(0, 0, 0, 0.48)",
		flex: 1,
		paddingHorizontal: 32,
		paddingTop: 24,
	},
	overlayHint: {
		opacity: 0.92,
		textAlign: "center",
	},
	modelPanel: {
		backgroundColor: "rgba(9, 15, 20, 0.7)",
		borderRadius: 16,
		gap: 6,
		marginTop: 16,
		maxWidth: 360,
		paddingHorizontal: 14,
		paddingVertical: 12,
		width: "100%",
	},
	modelTitle: {
		textAlign: "center",
	},
	modelLine: {
		fontSize: 13,
		lineHeight: 18,
		textAlign: "center",
	},
	detectionLayer: {
		...StyleSheet.absoluteFillObject,
	},
	detectionBox: {
		borderColor: "#72F2C0",
		borderRadius: 16,
		borderWidth: 3,
		position: "absolute",
	},
	detectionBadge: {
		alignSelf: "flex-start",
		backgroundColor: "#72F2C0",
		borderBottomRightRadius: 12,
		paddingHorizontal: 8,
		paddingVertical: 4,
	},
	detectionBadgeText: {
		fontSize: 12,
		fontWeight: "700",
		lineHeight: 14,
	},
});
