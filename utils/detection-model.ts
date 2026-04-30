import type { TfliteModel } from "react-native-fast-tflite";

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

function decodeYoloLikeDetections(
	model: Pick<TfliteModel, "inputs" | "outputs">,
	outputs: ArrayBuffer[],
	confidenceThreshold: number,
): Detection[] {
	"worklet";
	const outputTensor = model.outputs[0];
	if (outputTensor == null) return [];

	const layout = getYoloLayout(outputTensor.shape);
	if (layout == null || layout.channelCount < 5) return [];

	const raw = new Float32Array(outputs[0] ?? new ArrayBuffer(0));
	const candidateCount = Math.min(
		layout.candidateCount,
		Math.floor(raw.length / layout.channelCount),
	);
	if (candidateCount <= 0) return [];

	const inputH = model.inputs[0]?.shape[1] ?? 320;
	const inputW = model.inputs[0]?.shape[2] ?? 320;
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

export function decodeDetections(
	model: Pick<TfliteModel, "inputs" | "outputs">,
	outputs: ArrayBuffer[],
	confidenceThreshold: number,
): Detection[] {
	"worklet";
	if (model.outputs.length === 1) {
		return decodeYoloLikeDetections(model, outputs, confidenceThreshold);
	}

	return decodeSsdLikeDetections(outputs, confidenceThreshold);
}
