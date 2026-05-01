import { Asset } from "expo-asset";
import { useEffect, useState } from "react";
import {
	loadTensorflowModel,
	type TensorflowModelDelegate,
	type TensorflowPlugin,
} from "react-native-fast-tflite";

type DelegateAttempts = TensorflowModelDelegate[] | TensorflowModelDelegate[][];

function normalizeDelegateAttempts(
	delegateAttempts: DelegateAttempts,
): TensorflowModelDelegate[][] {
	const firstAttempt = (
		delegateAttempts as (
			| TensorflowModelDelegate
			| TensorflowModelDelegate[]
		)[]
	)[0];

	if (Array.isArray(firstAttempt)) {
		const attempts = delegateAttempts as TensorflowModelDelegate[][];
		return attempts.length > 0 ? attempts : [[]];
	}

	return [delegateAttempts as TensorflowModelDelegate[]];
}

export function useBundledTensorflowModel(
	source: number,
	delegateAttempts: DelegateAttempts,
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

				let model = undefined;
				let lastError: unknown = undefined;

				for (const delegates of normalizeDelegateAttempts(
					delegateAttempts,
				)) {
					try {
						model = await loadTensorflowModel(
							{ url: uri },
							delegates,
						);
						break;
					} catch (error) {
						lastError = error;
					}
				}

				if (model == null) {
					throw lastError instanceof Error
						? lastError
						: new Error("Failed to load TensorFlow model.");
				}

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
		// JSON.stringify compares delegate attempts by value.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [source, JSON.stringify(delegateAttempts)]);

	return state;
}
