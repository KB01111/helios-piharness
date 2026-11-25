import fs from "node:fs";
import path from "node:path";
import type {
	AttributeValue,
	SpanAttributes,
	SpanLink,
	SpanOptions,
	SpanStatus,
	StoredSpan,
	TelemetryContext,
	TelemetrySpan,
} from "@earendil-works/pi-telemetry";

/**
 * Trace export: a `TelemetryContext` adapter for Pi Durable's typed spans (`pi.durable.phase`, `.submit`, `.entry`, ...)
 * that appends every settled span as one JSON line (`StoredSpan` shape) to a file.
 *
 * Pi Durable's span attributes carry IDs and control metadata only, never messages, tool arguments or results; this
 * exporter additionally drops any oversized attribute value so a misbehaving extension cannot leak content through it.
 * Rotation keeps one previous file (`<path>.1`). Tail the file, or ship it with any log forwarder, into your tracing
 * backend; `renderTrace` from `@earendil-works/pi-telemetry/render` prints a trace from `StoredSpan[]` for debugging.
 */

export interface JsonlTraceExporterOptions {
	readonly filePath: string;
	/** Rotate when the file reaches this size. Default 50 MiB. */
	readonly maxBytes?: number;
	/** Longest attribute string/array element kept (default 256 chars); longer values are dropped, not truncated. */
	readonly maxAttributeChars?: number;
	readonly now?: () => number;
	/** Called when a write fails; export problems must never break the agent. */
	readonly onError?: (error: unknown) => void;
}

const DEFAULT_MAX_BYTES = 50 * 1024 * 1024;

export class JsonlTraceExporter implements TelemetryContext {
	private readonly runId = Math.random().toString(36).slice(2, 10);
	private counter = 0;
	private bytes = 0;
	private fd: number | undefined;
	private readonly maxBytes: number;
	private readonly maxChars: number;
	private readonly now: () => number;
	private spansWritten = 0;
	private closed = false;

	constructor(private readonly options: JsonlTraceExporterOptions) {
		this.maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
		this.maxChars = options.maxAttributeChars ?? 256;
		this.now = options.now ?? (() => Date.now());
		fs.mkdirSync(path.dirname(options.filePath), { recursive: true });
		this.open();
	}

	private open(): void {
		this.fd = fs.openSync(this.options.filePath, "a");
		this.bytes = fs.fstatSync(this.fd).size;
	}

	/** Spans written so far (for health/metrics). */
	written(): number {
		return this.spansWritten;
	}

	nextId(): string {
		this.counter += 1;
		return `${this.runId}:${this.counter}`;
	}

	clean(attributes: SpanAttributes | undefined): SpanAttributes {
		const out: Record<string, AttributeValue> = {};
		for (const [key, value] of Object.entries(attributes ?? {})) {
			if (value === undefined) continue;
			const items: readonly unknown[] = Array.isArray(value) ? value : [value];
			if (items.some((item) => typeof item === "string" && item.length > this.maxChars)) continue;
			out[key] = value;
		}
		return out;
	}

	write(span: StoredSpan): void {
		if (this.closed || this.fd === undefined) return;
		try {
			const line = `${JSON.stringify(span)}\n`;
			if (this.bytes + line.length > this.maxBytes) this.rotate();
			fs.writeSync(this.fd, line);
			this.bytes += line.length;
			this.spansWritten += 1;
		} catch (error) {
			this.options.onError?.(error);
		}
	}

	private rotate(): void {
		if (this.fd !== undefined) fs.closeSync(this.fd);
		try {
			fs.renameSync(this.options.filePath, `${this.options.filePath}.1`);
		} catch {
			// A failed rename only means the next open appends to the same file.
		}
		this.open();
	}

	clock(): number {
		return this.now();
	}

	startSpan<T>(options: SpanOptions, callback: (span: TelemetrySpan) => T | Promise<T>): Promise<T> {
		return runSpan(this, undefined, options, callback);
	}

	close(): void {
		if (this.closed) return;
		this.closed = true;
		if (this.fd !== undefined) fs.closeSync(this.fd);
		this.fd = undefined;
	}
}

class ExportSpan implements TelemetrySpan {
	readonly attributes: Record<string, AttributeValue> = {};
	readonly events: { name: string; timeMs: number; attributes: SpanAttributes }[] = [];
	readonly links: { traceId: string; spanId: string; attributes?: SpanAttributes }[] = [];
	status: SpanStatus = { status: "ok" };
	readonly startedMs: number;

	constructor(
		private readonly exporter: JsonlTraceExporter,
		readonly id: string,
		readonly traceId: string,
		readonly parentId: string | undefined,
		readonly name: string,
	) {
		this.startedMs = exporter.clock();
	}

	addLink(link: SpanLink): void {
		const target = link.context;
		if (target instanceof ExportSpan) {
			this.links.push({
				traceId: target.traceId,
				spanId: target.id,
				...(link.attributes ? { attributes: this.exporter.clean(link.attributes) } : {}),
			});
		}
	}

	addEvent(name: string, attributes?: SpanAttributes): void {
		this.events.push({ name, timeMs: this.exporter.clock(), attributes: this.exporter.clean(attributes) });
	}

	setAttributes(attributes: SpanAttributes): void {
		Object.assign(this.attributes, this.exporter.clean(attributes));
	}

	setStatus(status: SpanStatus): void {
		this.status = status;
	}

	startSpan<T>(options: SpanOptions, callback: (span: TelemetrySpan) => T | Promise<T>): Promise<T> {
		return runSpan(this.exporter, this, options, callback);
	}

	finish(): StoredSpan {
		return {
			id: this.id,
			traceId: this.traceId,
			...(this.parentId ? { parentId: this.parentId } : {}),
			name: this.name,
			startedMs: this.startedMs,
			endedMs: this.exporter.clock(),
			attributes: this.attributes,
			events: this.events,
			links: this.links,
			status: this.status,
		};
	}
}

async function runSpan<T>(
	exporter: JsonlTraceExporter,
	parent: ExportSpan | undefined,
	options: SpanOptions,
	callback: (span: TelemetrySpan) => T | Promise<T>,
): Promise<T> {
	const id = exporter.nextId();
	const span = new ExportSpan(exporter, id, parent?.traceId ?? id, parent?.id, options.name);
	span.setAttributes(options.attributes ?? {});
	for (const link of options.links ?? []) span.addLink(link);
	try {
		return await callback(span);
	} catch (error) {
		span.setStatus({
			status: "error",
			error: { name: error instanceof Error ? error.name : "Error", message: error instanceof Error ? error.message.slice(0, 500) : "error" },
		});
		throw error;
	} finally {
		exporter.write(span.finish());
	}
}
