/**
 * Optional supervisor for FastFlowLM (`flm`), AMD's NPU runtime for Ryzen AI (XDNA / XDNA2).
 *
 * FastFlowLM loads its embedding model (`embed-gemma:300m`) next to a chat model: `flm serve <llm> --embed 1`.
 * When `EMBED_NPU_AUTOSTART=true` and nothing answers at the configured port, the supervisor spawns that command
 * (with a small chat model; the NPU lock is only held while a request runs) and stops it on shutdown.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { probe } from "./backends/http-util.js";

export interface FlmSupervisorOptions {
	/** Path to the `flm` executable (default `flm` on PATH). */
	readonly command?: string;
	/** Chat model FastFlowLM must host next to the embedder (default `qwen3:0.6b`, the smallest). */
	readonly chatModel?: string;
	readonly host?: string;
	readonly port: number;
	/** FLM power mode: `powersaver` | `balanced` | `performance` | `turbo` (default `balanced`). */
	readonly powerMode?: string;
	readonly startupTimeoutMs?: number;
	readonly log?: (event: string, fields?: Record<string, unknown>) => void;
	/** Test seam. */
	readonly spawnImpl?: typeof spawn;
	readonly fetchImpl?: typeof fetch;
}

export class FlmSupervisor {
	private child: ChildProcess | undefined;
	private readonly options: FlmSupervisorOptions;

	constructor(options: FlmSupervisorOptions) {
		this.options = options;
	}

	get baseUrl(): string {
		return `http://${this.options.host ?? "127.0.0.1"}:${this.options.port}/v1`;
	}

	private async reachable(): Promise<boolean> {
		const res = await probe(`${this.baseUrl}/models`, {
			timeoutMs: 1_500,
			...(this.options.fetchImpl ? { fetchImpl: this.options.fetchImpl } : {}),
		});
		return res.ok;
	}

	/** Ensures a FastFlowLM server is answering; spawns one if needed. Resolves false when it never came up. */
	async ensureRunning(): Promise<boolean> {
		if (await this.reachable()) return true;
		const log = this.options.log ?? (() => undefined);
		const args = [
			"serve",
			this.options.chatModel ?? "qwen3:0.6b",
			"--embed",
			"1",
			"--host",
			this.options.host ?? "127.0.0.1",
			"--port",
			String(this.options.port),
			"--pmode",
			this.options.powerMode ?? "balanced",
			"--quiet",
		];
		log("flm_spawn", { command: this.options.command ?? "flm", args });
		const child = (this.options.spawnImpl ?? spawn)(this.options.command ?? "flm", args, {
			stdio: "ignore",
			windowsHide: true,
		});
		this.child = child;
		let exited = false;
		child.once("exit", (code) => {
			exited = true;
			if (this.child === child) this.child = undefined;
			log("flm_exit", { code });
		});
		child.once("error", (error) => {
			exited = true;
			if (this.child === child) this.child = undefined;
			log("flm_spawn_failed", { error: error.message });
		});

		const deadline = Date.now() + (this.options.startupTimeoutMs ?? 60_000);
		while (Date.now() < deadline && !exited) {
			if (await this.reachable()) {
				log("flm_ready", { baseUrl: this.baseUrl });
				return true;
			}
			await new Promise((resolve) => setTimeout(resolve, 500));
		}
		log("flm_not_ready", { baseUrl: this.baseUrl, exited });
		await this.stop();
		return false;
	}

	async stop(): Promise<void> {
		const child = this.child;
		this.child = undefined;
		if (!child || child.exitCode !== null) return;
		await new Promise<void>((resolve) => {
			const timer = setTimeout(() => {
				child.kill("SIGKILL");
				resolve();
			}, 5_000);
			child.once("exit", () => {
				clearTimeout(timer);
				resolve();
			});
			child.kill();
		});
	}
}
