/**
 * Tiny structured JSON-lines logger. One line per record:
 * `{"time","level","event",...base,...bindings,...fields}`.
 *
 * - Level filtering: records below the configured level are dropped.
 * - Errors are serialized as `{name,message,stack,code?}` only, so custom
 *   error properties (which may carry request bodies or credentials) are not emitted.
 * - Fields whose key looks secret-bearing are replaced with "[REDACTED]".
 * - The sink is injectable so tests can capture lines without touching stdio.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

export type LogFields = Readonly<Record<string, unknown>>;

export type LogSink = (line: string, level: LogLevel) => void;

export interface LoggerOptions {
	/** Minimum level emitted. Default: "info". */
	readonly level?: LogLevel;
	/** Fields attached to every record. */
	readonly base?: LogFields;
	/** Receives each serialized line. Default: stdout for debug/info, stderr for warn/error. */
	readonly sink?: LogSink;
	/** Clock used for the `time` field. Default: Date.now. */
	readonly now?: () => number;
}

export interface Logger {
	readonly level: LogLevel;
	isEnabled(level: LogLevel): boolean;
	debug(event: string, fields?: LogFields): void;
	info(event: string, fields?: LogFields): void;
	warn(event: string, fields?: LogFields): void;
	error(event: string, fields?: LogFields): void;
	/** Returns a logger sharing level/sink/clock whose records also carry `bindings`. */
	child(bindings: LogFields): Logger;
}

const LEVEL_RANK: Readonly<Record<LogLevel, number>> = { debug: 10, info: 20, warn: 30, error: 40 };

const SECRET_KEY_PATTERN = /secret|passw|api[_-]?key|authorization|cookie|credential|private[_-]?key|token$/i;

export function resolveLogLevel(raw: string | undefined, fallback: LogLevel = "info"): LogLevel {
	const normalized = (raw ?? "").trim().toLowerCase();
	return normalized === "debug" || normalized === "info" || normalized === "warn" || normalized === "error"
		? normalized
		: fallback;
}

/** Converts any thrown value into a log-safe object (no custom enumerable properties). */
export function serializeError(error: unknown): Record<string, unknown> {
	if (error instanceof Error) {
		const code = (error as NodeJS.ErrnoException).code;
		return {
			name: error.name,
			message: error.message,
			...(error.stack !== undefined ? { stack: error.stack } : {}),
			...(typeof code === "string" ? { code } : {}),
		};
	}
	return { message: String(error) };
}

function defaultSink(line: string, level: LogLevel): void {
	const stream = level === "warn" || level === "error" ? process.stderr : process.stdout;
	stream.write(`${line}\n`);
}

function toSafeJson(record: Record<string, unknown>): string {
	try {
		return JSON.stringify(record, (key, value: unknown) => {
			if (key !== "" && SECRET_KEY_PATTERN.test(key) && value !== undefined) return "[REDACTED]";
			if (value instanceof Error) return serializeError(value);
			if (typeof value === "bigint") return value.toString();
			return value;
		});
	} catch {
		return JSON.stringify({
			time: record.time,
			level: record.level,
			event: record.event,
			serializationError: true,
		});
	}
}

export function createLogger(options: LoggerOptions = {}): Logger {
	const level: LogLevel = options.level ?? "info";
	const base: LogFields = options.base ?? {};
	const sink: LogSink = options.sink ?? defaultSink;
	const now = options.now ?? (() => Date.now());

	const build = (bindings: LogFields): Logger => {
		const isEnabled = (candidate: LogLevel): boolean => LEVEL_RANK[candidate] >= LEVEL_RANK[level];

		const emit = (candidate: LogLevel, event: string, fields?: LogFields): void => {
			if (!isEnabled(candidate)) return;
			const record: Record<string, unknown> = {
				...base,
				...bindings,
				...fields,
				time: new Date(now()).toISOString(),
				level: candidate,
				event,
			};
			sink(toSafeJson(record), candidate);
		};

		return {
			level,
			isEnabled,
			debug: (event, fields) => emit("debug", event, fields),
			info: (event, fields) => emit("info", event, fields),
			warn: (event, fields) => emit("warn", event, fields),
			error: (event, fields) => emit("error", event, fields),
			child: (childBindings) => build({ ...bindings, ...childBindings }),
		};
	};

	return build({});
}
