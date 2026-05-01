import { ScrollView, StyleSheet, View } from "react-native";

import { ThemedText } from "@/components/themed-text";
import { Colors, Fonts } from "@/constants/theme";
import { useColorScheme } from "@/hooks/use-color-scheme";

const HERO_METRICS = [
	{ value: "Live", label: "camera scanning" },
	{ value: "Photo", label: "gallery analysis" },
	{ value: "Local", label: "TFLite inference" },
	{ value: "Events", label: "annotated uploads" },
];

const FEATURE_CARDS = [
	{
		eyebrow: "Detection tab",
		title: "Real-time field scanning",
		body: "The app opens the back camera, runs the bundled model on live frames, draws bounding boxes over accepted hits, shows the current egg-cluster count, and exposes torch plus upload status in the HUD.",
	},
	{
		eyebrow: "Photo tab",
		title: "Review saved images",
		body: "You can choose a photo from the library, run the same model against it, and inspect the rendered boxes, total detections, and top confidence without starting a camera session.",
	},
	{
		eyebrow: "Backend flow",
		title: "Capture only meaningful events",
		body: "When the live detector finds a new burst, the app can capture a still, burn the detection boxes into that image, and send the event id, timestamp, count, platform, and session metadata to the server.",
	},
];

const PIPELINE_STEPS = [
	{
		title: "Load the bundled model",
		body: "Snail Detector ships the TensorFlow Lite model inside the app, so inference starts on-device instead of depending on a remote model service.",
	},
	{
		title: "Prepare frames and photos",
		body: "Live camera frames and picked images are resized into the model's square input, then mapped back into screen space so the overlays line up with what you see.",
	},
	{
		title: "Keep the strongest detections",
		body: "Low-confidence boxes are filtered out, overlapping candidates are reduced, and the UI only surfaces the accepted egg-cluster detections.",
	},
	{
		title: "Rate-limit repeated uploads",
		body: "Ongoing detections are cooled down before another upload fires, which prevents backend spam when the boxes jitter but the scene has not meaningfully changed.",
	},
];

const OPERATOR_NOTES = [
	{
		title: "On-device first",
		body: "The detector runs locally for both tabs. The network is not part of the inference path, so the core scan still makes sense even before upload infrastructure is involved.",
	},
	{
		title: "Two ways to inspect the same model",
		body: "Use Detection for live surveying in the field. Use Photo for slower review of saved evidence, screenshots, or follow-up images with the same core model logic.",
	},
	{
		title: "What gets sent upstream",
		body: "Uploads are concise: an annotated JPEG plus structured metadata such as frame size, torch state, app version, platform, and a generated session id.",
	},
];

const aboutPalettes = {
	light: {
		page: "#f4f8fb",
		hero: "#103848",
		heroBadge: "#d9f6ff",
		heroBadgeText: "#0f4257",
		heroText: "#f7fbfd",
		heroMuted: "#b8d4de",
		surface: "#ffffff",
		surfaceAlt: "#eaf4f8",
		text: Colors.light.text,
		muted: "#4f6470",
		accent: "#0a7ea4",
		border: "#cfe0e8",
		numberFill: "#dff4ff",
		numberText: "#0b5470",
	},
	dark: {
		page: "#11181d",
		hero: "#173846",
		heroBadge: "#d8fbff",
		heroBadgeText: "#113746",
		heroText: "#f5fbfd",
		heroMuted: "#bfd6de",
		surface: "#1a242b",
		surfaceAlt: "#152128",
		text: Colors.dark.text,
		muted: "#a3bac4",
		accent: "#83dcff",
		border: "#29414d",
		numberFill: "#1d4555",
		numberText: "#e3f8ff",
	},
} as const;

export default function AboutScreen() {
	const colorScheme = useColorScheme() === "dark" ? "dark" : "light";
	const palette = aboutPalettes[colorScheme];

	return (
		<ScrollView
			style={[styles.screen, { backgroundColor: palette.page }]}
			contentContainerStyle={styles.container}
			showsVerticalScrollIndicator={false}
		>
			<View style={[styles.hero, { backgroundColor: palette.hero }]}>
				<View
					style={[
						styles.heroBadge,
						{ backgroundColor: palette.heroBadge },
					]}
				>
					<ThemedText
						style={[
							styles.heroBadgeText,
							{ color: palette.heroBadgeText },
						]}
					>
						FIELD OVERVIEW
					</ThemedText>
				</View>

				<ThemedText
					style={[styles.heroTitle, { color: palette.heroText }]}
				>
					Snail Detector
				</ThemedText>
				<ThemedText
					style={[styles.heroBody, { color: palette.heroMuted }]}
				>
					A camera-first workflow for spotting snail egg clusters,
					reviewing saved photos, and logging annotated detection
					events when the live scan finds something worth keeping.
				</ThemedText>

				<View style={styles.metricGrid}>
					{HERO_METRICS.map((metric) => (
						<View
							key={metric.label}
							style={[
								styles.metricCard,
								{
									backgroundColor:
										colorScheme === "dark"
											? "rgba(255, 255, 255, 0.08)"
											: "rgba(255, 255, 255, 0.14)",
									borderColor:
										colorScheme === "dark"
											? "rgba(255, 255, 255, 0.12)"
											: "rgba(255, 255, 255, 0.18)",
								},
							]}
						>
							<ThemedText
								style={[
									styles.metricValue,
									{ color: palette.heroText },
								]}
							>
								{metric.value}
							</ThemedText>
							<ThemedText
								style={[
									styles.metricLabel,
									{ color: palette.heroMuted },
								]}
							>
								{metric.label}
							</ThemedText>
						</View>
					))}
				</View>
			</View>

			<View style={styles.sectionHeader}>
				<ThemedText
					style={[styles.sectionEyebrow, { color: palette.accent }]}
				>
					APP MAP
				</ThemedText>
				<ThemedText
					style={[styles.sectionTitle, { color: palette.text }]}
				>
					What this app actually does
				</ThemedText>
			</View>

			{FEATURE_CARDS.map((card) => (
				<View
					key={card.title}
					style={[
						styles.surfaceCard,
						{
							backgroundColor: palette.surface,
							borderColor: palette.border,
						},
					]}
				>
					<ThemedText
						style={[styles.cardEyebrow, { color: palette.accent }]}
					>
						{card.eyebrow}
					</ThemedText>
					<ThemedText
						style={[styles.cardTitle, { color: palette.text }]}
					>
						{card.title}
					</ThemedText>
					<ThemedText
						style={[styles.cardBody, { color: palette.muted }]}
					>
						{card.body}
					</ThemedText>
				</View>
			))}

			<View
				style={[
					styles.pipelineCard,
					{
						backgroundColor: palette.surfaceAlt,
						borderColor: palette.border,
					},
				]}
			>
				<ThemedText
					style={[styles.sectionEyebrow, { color: palette.accent }]}
				>
					PIPELINE
				</ThemedText>
				<ThemedText
					style={[styles.sectionTitle, { color: palette.text }]}
				>
					How detection moves through the app
				</ThemedText>

				<View style={styles.timeline}>
					{PIPELINE_STEPS.map((step, index) => (
						<View key={step.title} style={styles.timelineRow}>
							<View
								style={[
									styles.timelineNumber,
									{ backgroundColor: palette.numberFill },
								]}
							>
								<ThemedText
									style={[
										styles.timelineNumberText,
										{ color: palette.numberText },
									]}
								>
									{index + 1}
								</ThemedText>
							</View>
							<View style={styles.timelineContent}>
								<ThemedText
									style={[
										styles.timelineTitle,
										{ color: palette.text },
									]}
								>
									{step.title}
								</ThemedText>
								<ThemedText
									style={[
										styles.timelineBody,
										{ color: palette.muted },
									]}
								>
									{step.body}
								</ThemedText>
							</View>
						</View>
					))}
				</View>
			</View>

			<View style={styles.sectionHeader}>
				<ThemedText
					style={[styles.sectionEyebrow, { color: palette.accent }]}
				>
					OPERATING NOTES
				</ThemedText>
				<ThemedText
					style={[styles.sectionTitle, { color: palette.text }]}
				>
					Why the workflow is split this way
				</ThemedText>
			</View>

			{OPERATOR_NOTES.map((note) => (
				<View
					key={note.title}
					style={[
						styles.surfaceCard,
						{
							backgroundColor: palette.surface,
							borderColor: palette.border,
						},
					]}
				>
					<ThemedText
						style={[styles.cardTitle, { color: palette.text }]}
					>
						{note.title}
					</ThemedText>
					<ThemedText
						style={[styles.cardBody, { color: palette.muted }]}
					>
						{note.body}
					</ThemedText>
				</View>
			))}
		</ScrollView>
	);
}

const styles = StyleSheet.create({
	screen: {
		flex: 1,
	},
	container: {
		gap: 18,
		paddingBottom: 36,
		paddingHorizontal: 20,
		paddingTop: 72,
	},
	hero: {
		borderRadius: 28,
		gap: 16,
		paddingHorizontal: 20,
		paddingVertical: 22,
	},
	heroBadge: {
		alignSelf: "flex-start",
		borderRadius: 999,
		paddingHorizontal: 12,
		paddingVertical: 7,
	},
	heroBadgeText: {
		fontFamily: Fonts.mono,
		fontSize: 11,
		fontWeight: "700",
		letterSpacing: 1.2,
	},
	heroTitle: {
		fontFamily: Fonts.rounded,
		fontSize: 38,
		fontWeight: "700",
		letterSpacing: -1.1,
		lineHeight: 40,
	},
	heroBody: {
		fontSize: 16,
		lineHeight: 24,
		maxWidth: 620,
	},
	metricGrid: {
		flexDirection: "row",
		flexWrap: "wrap",
		gap: 12,
	},
	metricCard: {
		borderRadius: 18,
		borderWidth: 1,
		flexGrow: 1,
		minWidth: 130,
		paddingHorizontal: 14,
		paddingVertical: 14,
	},
	metricValue: {
		fontFamily: Fonts.rounded,
		fontSize: 22,
		fontWeight: "700",
		lineHeight: 24,
		marginBottom: 4,
	},
	metricLabel: {
		fontSize: 13,
		lineHeight: 18,
	},
	sectionHeader: {
		gap: 6,
		paddingHorizontal: 2,
		paddingTop: 4,
	},
	sectionEyebrow: {
		fontFamily: Fonts.mono,
		fontSize: 11,
		fontWeight: "700",
		letterSpacing: 1.1,
	},
	sectionTitle: {
		fontFamily: Fonts.rounded,
		fontSize: 24,
		fontWeight: "700",
		letterSpacing: -0.6,
		lineHeight: 28,
	},
	surfaceCard: {
		borderRadius: 24,
		borderWidth: 1,
		gap: 10,
		paddingHorizontal: 18,
		paddingVertical: 18,
	},
	cardEyebrow: {
		fontFamily: Fonts.mono,
		fontSize: 11,
		fontWeight: "700",
		letterSpacing: 1,
	},
	cardTitle: {
		fontFamily: Fonts.rounded,
		fontSize: 20,
		fontWeight: "700",
		lineHeight: 24,
	},
	cardBody: {
		fontSize: 15,
		lineHeight: 23,
	},
	pipelineCard: {
		borderRadius: 28,
		borderWidth: 1,
		gap: 14,
		paddingHorizontal: 18,
		paddingVertical: 20,
	},
	timeline: {
		gap: 16,
		marginTop: 4,
	},
	timelineRow: {
		flexDirection: "row",
		gap: 14,
		alignItems: "flex-start",
	},
	timelineNumber: {
		alignItems: "center",
		borderRadius: 999,
		height: 34,
		justifyContent: "center",
		marginTop: 2,
		width: 34,
	},
	timelineNumberText: {
		fontFamily: Fonts.mono,
		fontSize: 14,
		fontWeight: "700",
	},
	timelineContent: {
		flex: 1,
		gap: 4,
	},
	timelineTitle: {
		fontFamily: Fonts.rounded,
		fontSize: 18,
		fontWeight: "700",
		lineHeight: 22,
	},
	timelineBody: {
		fontSize: 15,
		lineHeight: 23,
	},
});
