import { Asset } from "expo-asset";
import { useEffect, useState } from "react";
import {
	loadTensorflowModel,
	type TensorflowModelDelegate,
	type TensorflowPlugin,
} from "react-native-fast-tflite";

export function useBundledTensorflowModel(
	source: number,
	delegates: TensorflowModelDelegate[],
): TensorflowPlugin {
	const [state, setState] = useState<TensorflowPlugin>({
		model: undefined,
		state: "loading",
	});

	useEffect(() => {
		let cancelled = false;

		const load = async (): Promise<void> => {
			try {
				setState({ model: undefined, state: "loading" });

				const asset = Asset.fromModule(source);
				await asset.downloadAsync();
				const uri = asset.localUri ?? asset.uri;

				if (!uri.includes(":")) {
					throw new Error(
						`TFLite asset did not resolve to a valid URL: ${uri}`,
					);
				}

				const model = await loadTensorflowModel(
					{ url: uri },
					delegates,
				);

				if (!cancelled) {
					setState({ model, state: "loaded" });
				}
			} catch (error) {
				if (!cancelled) {
					setState({
						model: undefined,
						state: "error",
						error: error as Error,
					});
				}
			}
		};

		void load();

		return () => {
			cancelled = true;
		};
		// JSON.stringify compares delegates by value.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [source, JSON.stringify(delegates)]);

	return state;
}
