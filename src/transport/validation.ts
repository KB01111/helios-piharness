import { ValidationError } from "../contracts/errors.js";

export type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Parses a JSON object body. Empty bodies yield `{}` unless `required`. Malformed JSON -> ValidationError (400). */
export function parseJsonObject(text: string, options: { readonly required?: boolean } = {}): JsonRecord {
	if (text.trim().length === 0) {
		if (options.required) throw new ValidationError("Request body must be a JSON object");
		return {};
	}
	let parsed: unknown;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new ValidationError("Malformed JSON body");
	}
	if (!isRecord(parsed)) throw new ValidationError("Request body must be a JSON object");
	return parsed;
}

export function optionalString(
	body: JsonRecord,
	field: string,
	options: { readonly maxLength?: number; readonly nonEmpty?: boolean } = {},
): string | undefined {
	const value = body[field];
	if (value === undefined || value === null) return undefined;
	if (typeof value !== "string") throw new ValidationError(`${field} must be a string`);
	if (options.nonEmpty && value.trim().length === 0) throw new ValidationError(`${field} must be a non-empty string`);
	if (options.maxLength !== undefined && value.length > options.maxLength) {
		throw new ValidationError(`${field} must be at most ${options.maxLength} characters`);
	}
	return value;
}

export function requiredString(body: JsonRecord, field: string, options: { readonly maxLength?: number } = {}): string {
	const value = optionalString(body, field, { ...options, nonEmpty: true });
	if (value === undefined) throw new ValidationError(`${field} is required and must be a non-empty string`);
	return value;
}

export function optionalBoolean(body: JsonRecord, field: string): boolean | undefined {
	const value = body[field];
	if (value === undefined || value === null) return undefined;
	if (typeof value !== "boolean") throw new ValidationError(`${field} must be a boolean`);
	return value;
}

export function optionalPositiveInt(body: JsonRecord, field: string, max = 10_000): number | undefined {
	const value = body[field];
	if (value === undefined || value === null) return undefined;
	if (typeof value !== "number" || !Number.isInteger(value) || value <= 0 || value > max) {
		throw new ValidationError(`${field} must be an integer between 1 and ${max}`);
	}
	return value;
}

export function optionalEnum<T extends string>(body: JsonRecord, field: string, allowed: readonly T[]): T | undefined {
	const value = body[field];
	if (value === undefined || value === null) return undefined;
	if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
		throw new ValidationError(`${field} must be one of: ${allowed.join(", ")}`);
	}
	return value as T;
}

export function optionalStringArray(
	body: JsonRecord,
	field: string,
	options: { readonly maxItems?: number; readonly maxItemLength?: number } = {},
): string[] | undefined {
	const value = body[field];
	if (value === undefined || value === null) return undefined;
	const maxItems = options.maxItems ?? 100;
	const maxItemLength = options.maxItemLength ?? 200;
	if (!Array.isArray(value) || value.length > maxItems) {
		throw new ValidationError(`${field} must be an array of at most ${maxItems} strings`);
	}
	for (const item of value) {
		if (typeof item !== "string" || item.length === 0 || item.length > maxItemLength) {
			throw new ValidationError(`${field} must contain only non-empty strings (max ${maxItemLength} characters)`);
		}
	}
	return [...value] as string[];
}

export function optionalModelSpec(body: JsonRecord, field: string): { provider: string; modelId: string } | undefined {
	const value = body[field];
	if (value === undefined || value === null) return undefined;
	if (!isRecord(value)) throw new ValidationError(`${field} must be an object with provider and modelId`);
	return {
		provider: requiredString(value, "provider", { maxLength: 200 }),
		modelId: requiredString(value, "modelId", { maxLength: 200 }),
	};
}
