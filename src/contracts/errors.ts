/**
 * Typed errors shared by the harness service and transports. Transports map
 * these to status codes by class (never by message substring).
 */
export class HarnessError extends Error {
	readonly status: number;
	readonly code: string;

	constructor(message: string, status: number, code: string) {
		super(message);
		this.name = new.target.name;
		this.status = status;
		this.code = code;
	}
}

export class ValidationError extends HarnessError {
	constructor(message: string) {
		super(message, 400, "invalid_request");
	}
}

export class UnauthorizedError extends HarnessError {
	constructor(message = "Authentication required") {
		super(message, 401, "unauthorized");
	}
}

export class ForbiddenError extends HarnessError {
	constructor(message = "Forbidden") {
		super(message, 403, "forbidden");
	}
}

/** Also used for resources owned by another tenant, to avoid leaking existence. */
export class NotFoundError extends HarnessError {
	constructor(message: string) {
		super(message, 404, "not_found");
	}
}

export class PayloadTooLargeError extends HarnessError {
	constructor(message = "Request body too large") {
		super(message, 413, "payload_too_large");
	}
}

export class RateLimitError extends HarnessError {
	readonly retryAfterSeconds: number;

	constructor(retryAfterSeconds: number, message = "Rate limit exceeded", code = "rate_limited") {
		super(message, 429, code);
		this.retryAfterSeconds = retryAfterSeconds;
	}
}

/** Too many concurrently open SSE streams (per tenant or process-wide). */
export class TooManyStreamsError extends RateLimitError {
	constructor(retryAfterSeconds = 5) {
		super(retryAfterSeconds, "Too many open streams", "too_many_streams");
	}
}
