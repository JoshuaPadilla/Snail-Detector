import { ScrollView, StyleSheet } from "react-native";

import { ThemedText } from "@/components/themed-text";
import { ThemedView } from "@/components/themed-view";

export default function AboutScreen() {
	return (
		<ScrollView contentContainerStyle={styles.container}>
			<ThemedText type="title">About Snail Detector</ThemedText>

			<ThemedView style={styles.section}>
				<ThemedText type="subtitle">
					What The Detection Tab Does
				</ThemedText>
				<ThemedText>
					The Detection tab opens the camera, crops the center guide
					box, resizes it to the model input, and runs a TensorFlow
					Lite object detector on-device.
				</ThemedText>
			</ThemedView>

			<ThemedView style={styles.section}>
				<ThemedText type="subtitle">Model Input</ThemedText>
				<ThemedText>
					The loaded model expects a 300 × 300 RGB image with uint8
					pixel values. That means every frame must be converted into
					a square image before inference runs.
				</ThemedText>
			</ThemedView>

			<ThemedView style={styles.section}>
				<ThemedText type="subtitle">Model Output</ThemedText>
				<ThemedText>
					The detector returns four outputs: bounding boxes, class
					IDs, confidence scores, and the number of valid detections.
					The app uses those boxes to draw labels on top of the
					camera.
				</ThemedText>
			</ThemedView>

			<ThemedView style={styles.section}>
				<ThemedText type="subtitle">
					Current Label Assumption
				</ThemedText>
				<ThemedText>
					This demo currently labels every accepted detection as egg
					snail because the model label file is not available yet. If
					you later add the real labels, the app can map class IDs to
					the correct names.
				</ThemedText>
			</ThemedView>

			<ThemedView style={styles.section}>
				<ThemedText type="subtitle">How To Learn From It</ThemedText>
				<ThemedText>
					Open the Detection tab and watch the Metro logs prefixed
					with SnailModel. They show the model inputs, outputs,
					smoke-test run, and any live detections that reach the
					confidence threshold.
				</ThemedText>
			</ThemedView>
		</ScrollView>
	);
}

const styles = StyleSheet.create({
	container: {
		gap: 20,
		paddingBottom: 32,
		paddingHorizontal: 20,
		paddingTop: 72,
	},
	section: {
		borderRadius: 18,
		gap: 8,
		padding: 16,
	},
});
