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
import { XerjRequestError } from "./xerj-client.js";

export interface FallbackFileSearchOptions {
	readonly primary: FileSearchService;
	readonly fallback: FileSearchService;
	/** Called when the primary engine fails transiently and the request is served by the fallback engine. */
	readonly onFallback?: (operation: string, error: unknown) => void;
}

function shouldFallback(error: unknown, signal?: AbortSignal): boolean {
	if (signal?.aborted) return false;
	// Non-retryable client/auth errors (e.g. 401/403 rejected API key or 400 contract error)
	// must surface rather than silently serving fallback results.
	if (error instanceof XerjRequestError && !error.retryable) {
		return false;
	}
	return true;
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
			if (!shouldFallback(error, options?.signal)) throw error;
			this.notifyFallback("getIndexMap", error);
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

	private notifyFallback(operation: string, error: unknown): void {
		try {
			this.onFallback?.(operation, error);
		} catch {
			// Diagnostic callback failures must never block fallback recovery.
		}
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
			if (!shouldFallback(error, signal)) throw error;
			this.notifyFallback(op, error);
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
