import { CodemodeSandbox } from "@earendil-works/pi-codemode";
import type { Extension } from "@earendil-works/pi-durable";
import { createCodemodeExtension } from "@earendil-works/pi-durable/codemode";

export interface CodemodeExtensionOptions {
	/** Heap cap of each QuickJS VM in bytes. */
	readonly memoryLimitBytes: number;
	/** Estimated tokens the `codemode` prompt section may spend on tool declarations. */
	readonly inlineBudget: number;
	/** Hard process-wide cap on concurrently running sandbox workers (all tenants combined). */
	readonly maxWorkers: number;
	/** Per-script wall-clock deadline in ms, tool time included. */
	readonly timeoutMs: number;
}

export interface CodemodeHandle {
	readonly extension: Extension;
	/** Workers currently admitted (for health/metrics). */
	activeWorkers(): number;
	close(): Promise<void>;
}

/**
 * Durable Pi "codemode": the model writes JavaScript that calls the conversation's tools as `tools.<name>(args)`
 * inside a QuickJS/WASM sandbox, and only the script's output reaches the model.
 *
 * Safety posture:
 * - Every `tools.*` call is a *nested tool call*, so `guard-shaping` (tenant-spoof block + permission map) and the
 *   other ToolTask hooks still run for each one. The sandbox has no fs/network/timers of its own.
 * - `models.*` globals are disabled: they would let scripts spend the main provider outside per-turn budget accounting.
 * - Worker admission is capped process-wide so one tenant cannot exhaust CPU/RAM with parallel scripts.
 * - Scripts survive restarts by deterministic replay; finished nested calls are not re-run.
 */
export function createHeliosCodemode(options: CodemodeExtensionOptions): CodemodeHandle {
	let active = 0;
	const sandbox = new CodemodeSandbox({
		memoryLimitBytes: options.memoryLimitBytes,
		timeoutMs: options.timeoutMs,
		acquireWorker: () => {
			if (active >= options.maxWorkers) {
				throw new Error(`codemode is busy: ${options.maxWorkers} script workers already running; retry shortly.`);
			}
			active += 1;
			let released = false;
			return () => {
				if (released) return;
				released = true;
				active -= 1;
			};
		},
	});

	return {
		extension: createCodemodeExtension({
			sandbox,
			inlineBudget: options.inlineBudget,
			models: false,
		}),
		activeWorkers: () => active,
		close: async () => {
			await sandbox.close();
		},
	};
}
