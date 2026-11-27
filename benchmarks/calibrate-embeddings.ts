/**
 * Prints the embedding quality self-test and measured cosine calibration of an embedding endpoint
 * (see `src/embeddings/self-test.ts`; the corpus shares no content with the memory benchmark).
 *
 * Usage: tsx benchmarks/calibrate-embeddings.ts [--endpoint http://127.0.0.1:18765/v1] [--api-key <key>] [--in-process <profile>]
 */

import type { EmbeddingProvider } from "../src/contracts/app-capabilities.js";
import { loadEmbedServerConfig } from "../src/embeddings/config.js";
import { buildEmbeddingService } from "../src/embeddings/factory.js";
import { HttpEmbeddingProvider, RouterEmbeddingProvider } from "../src/embeddings/provider.js";
import { runEmbeddingSelfTest } from "../src/embeddings/self-test.js";

function argValue(args: readonly string[], flag: string): string | undefined {
	const i = args.indexOf(flag);
	return i >= 0 ? args[i + 1] : undefined;
}

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	const endpoint = argValue(args, "--endpoint") ?? "http://127.0.0.1:18765/v1";
	const apiKey = argValue(args, "--api-key");
	const profile = argValue(args, "--in-process");
	let provider: EmbeddingProvider;
	let close: (() => Promise<void>) | undefined;
	if (profile) {
		const service = buildEmbeddingService(loadEmbedServerConfig({ ...process.env, EMBED_PROFILE: profile }));
		provider = new RouterEmbeddingProvider(service.router);
		close = () => service.close();
	} else {
		provider = await HttpEmbeddingProvider.connect({ baseUrl: endpoint, ...(apiKey ? { apiKey } : {}), timeoutMs: 300_000 });
	}
	const result = await runEmbeddingSelfTest((texts, inputType) => provider.embed(texts, { inputType }));
	console.log(JSON.stringify({ model: provider.model, ...result }, null, 2));
	await close?.();
}

main().catch((error: unknown) => {
	console.error(error);
	process.exit(1);
});
