import { Platform } from "react-native";
import { Images, loadImage } from "react-native-nitro-image";

const HARD_CODED_UPLOAD_URL = "https://syncfit.site/api/detection-logs";
// const HARD_CODED_UPLOAD_URL =
// 	"https://6bff-103-224-95-72.ngrok-free.app/api/detection-logs";
const BOX_COLOR = { r: 0, g: 229, b: 255, a: 255 };

export interface DetectionUploadBox {
	xmin: number;
	ymin: number;
	xmax: number;
	ymax: number;
	score: number;
}

export interface CapturedFrameImageData {
	buffer: ArrayBuffer;
	width: number;
	height: number;
	imageFormat: "jpg" | "png" | "heic";
}

export interface SnailDetectionUploadEvent {
	eventId: string;
	photoUri: string;
	capturedFrameImageData?: CapturedFrameImageData;
	capturedAt: string;
	eggClusterCount: number;
	detections: DetectionUploadBox[];
	metadata: {
		cameraDeviceId?: string;
		frameWidth: number;
		frameHeight: number;
		torchEnabled: boolean;
		platform: string;
		platformVersion: string;
		appVersion?: string;
		sessionId: string;
	};
}

export interface UploadResult {
	ok: boolean;
	disabled?: boolean;
	status?: number;
	bodyText?: string;
}

function toArrayBuffer(value: unknown): ArrayBuffer | null {
	if (value instanceof ArrayBuffer) {
		return value.slice(0);
	}

	if (ArrayBuffer.isView(value)) {
		const bytes = new Uint8Array(
			value.buffer,
			value.byteOffset,
			value.byteLength,
		);
		return Uint8Array.from(bytes).buffer;
	}

	if (Array.isArray(value)) {
		return Uint8Array.from(value).buffer;
	}

	if (typeof value === "object" && value != null) {
		const record = value as Record<string, unknown>;

		if ("buffer" in record) {
			const nested = toArrayBuffer(record.buffer);
			if (nested != null) return nested;
		}

		const lengthValue = record.length;
		if (typeof lengthValue === "number" && Number.isFinite(lengthValue)) {
			const bytes = new Uint8Array(lengthValue);
			for (let index = 0; index < lengthValue; index++) {
				bytes[index] = Number(record[String(index)] ?? 0) & 0xff;
			}
			return bytes.buffer;
		}

		const byteEntries = Object.entries(record)
			.filter(([key]) => /^\d+$/.test(key))
			.sort((left, right) => Number(left[0]) - Number(right[0]));

		if (byteEntries.length > 0) {
			const bytes = new Uint8Array(byteEntries.length);
			for (let index = 0; index < byteEntries.length; index++) {
				bytes[index] = Number(byteEntries[index][1] ?? 0) & 0xff;
			}
			return bytes.buffer;
		}
	}

	return null;
}

function normalizeCapturedFrameImageData(
	capturedFrameImageData: CapturedFrameImageData,
): CapturedFrameImageData {
	const normalizedBuffer = toArrayBuffer(capturedFrameImageData.buffer);
	if (normalizedBuffer == null) {
		throw new Error(
			"Captured frame image data buffer could not be normalized to an ArrayBuffer.",
		);
	}

	return {
		...capturedFrameImageData,
		buffer: normalizedBuffer,
	};
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function getStrokeWidth(width: number, height: number): number {
	return Math.max(
		4,
		Math.min(14, Math.round(Math.min(width, height) * 0.008)),
	);
}

function getPixelLayout(pixelFormat: string): {
	bytesPerPixel: number;
	r: number;
	g: number;
	b: number;
	a?: number;
} | null {
	switch (pixelFormat) {
		case "BGRA":
			return { bytesPerPixel: 4, r: 2, g: 1, b: 0, a: 3 };
		case "RGBA":
			return { bytesPerPixel: 4, r: 0, g: 1, b: 2, a: 3 };
		case "ARGB":
			return { bytesPerPixel: 4, r: 1, g: 2, b: 3, a: 0 };
		case "ABGR":
			return { bytesPerPixel: 4, r: 3, g: 2, b: 1, a: 0 };
		case "XRGB":
			return { bytesPerPixel: 4, r: 1, g: 2, b: 3 };
		case "BGRX":
			return { bytesPerPixel: 4, r: 2, g: 1, b: 0 };
		case "XBGR":
			return { bytesPerPixel: 4, r: 3, g: 2, b: 1 };
		case "RGBX":
			return { bytesPerPixel: 4, r: 0, g: 1, b: 2 };
		case "RGB":
			return { bytesPerPixel: 3, r: 0, g: 1, b: 2 };
		case "BGR":
			return { bytesPerPixel: 3, r: 2, g: 1, b: 0 };
		default:
			return null;
	}
}

function paintFilledRect(
	pixels: Uint8Array,
	imageWidth: number,
	imageHeight: number,
	layout: {
		bytesPerPixel: number;
		r: number;
		g: number;
		b: number;
		a?: number;
	},
	left: number,
	top: number,
	width: number,
	height: number,
): void {
	const right = clamp(left + width, 0, imageWidth);
	const bottom = clamp(top + height, 0, imageHeight);
	const safeLeft = clamp(left, 0, imageWidth);
	const safeTop = clamp(top, 0, imageHeight);

	for (let y = safeTop; y < bottom; y++) {
		for (let x = safeLeft; x < right; x++) {
			const offset = (y * imageWidth + x) * layout.bytesPerPixel;
			pixels[offset + layout.r] = BOX_COLOR.r;
			pixels[offset + layout.g] = BOX_COLOR.g;
			pixels[offset + layout.b] = BOX_COLOR.b;
			if (layout.a != null) {
				pixels[offset + layout.a] = BOX_COLOR.a;
			}
		}
	}
}

async function renderBoundingBoxesIntoPhoto(
	photoUri: string,
	capturedFrameImageData: CapturedFrameImageData | undefined,
	detections: DetectionUploadBox[],
): Promise<string> {
	if (detections.length === 0) return photoUri;

	const baseImage =
		capturedFrameImageData != null
			? await loadImage({
					encodedImageData: normalizeCapturedFrameImageData(
						capturedFrameImageData,
					),
				})
			: await loadImage({ filePath: photoUri });
	const rawPixelData = await baseImage.toRawPixelDataAsync();
	const layout = getPixelLayout(rawPixelData.pixelFormat);
	if (layout == null) {
		throw new Error(
			`Unsupported pixel format for annotation: ${rawPixelData.pixelFormat}`,
		);
	}
	const pixels = new Uint8Array(rawPixelData.buffer);
	const strokeWidth = getStrokeWidth(baseImage.width, baseImage.height);

	for (const detection of detections) {
		const left = clamp(
			Math.round(detection.xmin * baseImage.width),
			0,
			Math.max(0, baseImage.width - 1),
		);
		const top = clamp(
			Math.round(detection.ymin * baseImage.height),
			0,
			Math.max(0, baseImage.height - 1),
		);
		const right = clamp(
			Math.round(detection.xmax * baseImage.width),
			left + 1,
			baseImage.width,
		);
		const bottom = clamp(
			Math.round(detection.ymax * baseImage.height),
			top + 1,
			baseImage.height,
		);
		const boxWidth = Math.max(1, right - left);
		const boxHeight = Math.max(1, bottom - top);
		const verticalStroke = Math.min(strokeWidth, boxWidth);
		const horizontalStroke = Math.min(strokeWidth, boxHeight);

		paintFilledRect(
			pixels,
			baseImage.width,
			baseImage.height,
			layout,
			left,
			top,
			boxWidth,
			horizontalStroke,
		);
		paintFilledRect(
			pixels,
			baseImage.width,
			baseImage.height,
			layout,
			left,
			bottom - horizontalStroke,
			boxWidth,
			horizontalStroke,
		);
		paintFilledRect(
			pixels,
			baseImage.width,
			baseImage.height,
			layout,
			left,
			top,
			verticalStroke,
			boxHeight,
		);
		paintFilledRect(
			pixels,
			baseImage.width,
			baseImage.height,
			layout,
			right - verticalStroke,
			top,
			verticalStroke,
			boxHeight,
		);
	}

	const annotatedImage = Images.loadFromRawPixelData(rawPixelData);
	return annotatedImage.saveToTemporaryFileAsync("jpg", 92);
}

export function getSnailDetectionUploadUrl(): string {
	return HARD_CODED_UPLOAD_URL;
}

export function normalizeFileUri(filePathOrUri: string): string {
	if (filePathOrUri.startsWith("file://")) return filePathOrUri;
	if (filePathOrUri.includes("://")) return filePathOrUri;
	return `file://${filePathOrUri}`;
}

function buildPhotoName(event: SnailDetectionUploadEvent): string {
	const safeTimestamp = event.capturedAt.replace(/[:.]/g, "-");
	return `snail-detection-${safeTimestamp}.jpg`;
}

export async function uploadSnailDetectionEvent(
	event: SnailDetectionUploadEvent,
): Promise<UploadResult> {
	const uploadUrl = getSnailDetectionUploadUrl();
	console.log("Uploading snail detection event to:", uploadUrl);
	if (uploadUrl == null) {
		return { ok: false, disabled: true };
	}

	const annotatedPhotoUri = await renderBoundingBoxesIntoPhoto(
		event.photoUri,
		event.capturedFrameImageData,
		event.detections,
	).catch((error) => {
		console.warn(
			"[DetectionUpload] Failed to annotate photo with bounding boxes. Uploading original photo instead.",
			error,
		);
		return event.photoUri;
	});

	if (annotatedPhotoUri.trim().length === 0) {
		throw new Error(
			"No uploadable photo URI was available for this event.",
		);
	}

	const formData = new FormData();
	formData.append("photo", {
		uri: normalizeFileUri(annotatedPhotoUri),
		name: buildPhotoName(event),
		type: "image/jpeg",
	} as never);
	formData.append("eventId", event.eventId);
	formData.append("capturedAt", event.capturedAt);
	formData.append("eggClusterCount", String(event.eggClusterCount));
	formData.append("platform", event.metadata.platform);
	formData.append("metadata", JSON.stringify(event.metadata));

	const response = await fetch(uploadUrl, {
		method: "POST",
		body: formData,
		headers: {
			Accept: "application/json",
		},
	});

	if (response.ok) {
		return { ok: true, status: response.status };
	}

	const bodyText = await response.text().catch(() => "");
	return {
		ok: false,
		status: response.status,
		bodyText,
	};
}

export function buildDetectionEventId(capturedAt: string): string {
	const randomSuffix = Math.random().toString(36).slice(2, 8);
	return `det-${capturedAt}-${Platform.OS}-${randomSuffix}`;
}
