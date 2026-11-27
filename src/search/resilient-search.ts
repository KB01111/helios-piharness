import type {
	FileContentSearchOptions,
	FileFindOptions,
	FileIndexMapResult,
	FileSearchHealth,
	FileSearchResult,
	FileSearchScope,
	FileSearchService,
	FileSymbolDefOptions,
} from "../contracts/app-capabilities.js";

export interface FallbackFileSearchOptions {
	readonly primary: FileSearchService;
	readonly fallback: FileSearchService;
	/** Called when the primary engine fails and the request is served by the fallback engine. */
	readonly onFallback?: (operation: string, error: unknown) => void;
}

/**
 * Wraps a primary `FileSearchService` (e.g. `XerjFileSearchService`) with an automatic fallback
 * (e.g. `LocalWorkspaceFileSearchService`) so transient search daemon outages never break agent runs.
 */
export class FallbackFileSearchService implements FileSearchService {
	readonly engine: string;
	private readonly primary: FileSearchService;
	private readonly fallback: FileSearchService;
	private readonly onFallback?: (operation: string, error: unknown) => void;

	constructor(options: FallbackFileSearchOptions) {
		this.primary = options.primary;
		this.fallback = options.fallback;
		this.engine = `${options.primary.engine}+${options.fallback.engine}`;
		this.onFallback = options.onFallback;
	}

	async findFiles(scope: FileSearchScope, options: FileFindOptions): Promise<FileSearchResult> {
		return this.withFallback(
			"findFiles",
			options.signal,
			() => this.primary.findFiles(scope, options),
			() => this.fallback.findFiles(scope, options),
		);
	}

	async searchContent(scope: FileSearchScope, options: FileContentSearchOptions): Promise<FileSearchResult> {
		return this.withFallback(
			"searchContent",
			options.signal,
			() => this.primary.searchContent(scope, options),
			() => this.fallback.searchContent(scope, options),
		);
	}

	async findSymbolDefinitions(scope: FileSearchScope, options: FileSymbolDefOptions): Promise<FileSearchResult> {
		return this.withFallback(
			"findSymbolDefinitions",
			options.signal,
			() => this.primary.findSymbolDefinitions(scope, options),
			() => this.fallback.findSymbolDefinitions(scope, options),
		);
	}

	async getIndexMap(
		scope: FileSearchScope,
		options?: { readonly limit?: number; readonly signal?: AbortSignal },
	): Promise<FileIndexMapResult> {
		try {
			if (this.primary.getIndexMap) {
				return await this.primary.getIndexMap(scope, options);
			}
		} catch (error) {
			if (options?.signal?.aborted) throw error;
			this.onFallback?.("getIndexMap", error);
		}
		if (this.fallback.getIndexMap) {
			const res = await this.fallback.getIndexMap(scope, options);
			return {
				...res,
				notes: [
					...(res.notes ?? []),
					`Primary engine "${this.primary.engine}" unavailable; index map served by "${this.fallback.engine}".`,
				],
			};
		}
		return { engine: this.engine, indexScope: "(unavailable)", datasets: [], totalDocs: 0 };
	}

	async healthCheck(signal?: AbortSignal): Promise<FileSearchHealth> {
		const primaryHealth = this.primary.healthCheck
			? await this.primary.healthCheck(signal)
			: { ok: true, engine: this.primary.engine };
		if (primaryHealth.ok) return primaryHealth;
		const fallbackHealth = this.fallback.healthCheck
			? await this.fallback.healthCheck(signal)
			: { ok: true, engine: this.fallback.engine };
		return {
			ok: fallbackHealth.ok,
			engine: fallbackHealth.ok ? this.fallback.engine : this.primary.engine,
			...(primaryHealth.error ? { error: `primary(${this.primary.engine}): ${primaryHealth.error}` } : {}),
		};
	}

	private async withFallback(
		op: string,
		signal: AbortSignal | undefined,
		runPrimary: () => Promise<FileSearchResult>,
		runFallback: () => Promise<FileSearchResult>,
	): Promise<FileSearchResult> {
		try {
			return await runPrimary();
		} catch (error) {
			if (signal?.aborted) throw error;
			this.onFallback?.(op, error);
			const res = await runFallback();
			return {
				...res,
				notes: [
					...(res.notes ?? []),
					`Primary search engine "${this.primary.engine}" unavailable; served by fallback "${this.fallback.engine}".`,
				],
			};
		}
	}
}
