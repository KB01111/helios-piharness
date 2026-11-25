/**
 * Helpers that decide which run-failure text may reach API clients (submission results and SSE `run_status`).
 * Free-text failure details can carry provider errors, file paths or SQL, so only stable codes and the
 * deliberately client-visible budget veto message are exposed; everything else is logged server-side.
 */

/** Reasons are stable snake_case codes; anything else (free text) is collapsed so it cannot leak internals. */
const SAFE_REASON_PATTERN = /^[a-z][a-z0-9_.-]{0,63}$/;
/** Budget vetoes are intentionally client-visible (they tell the caller which limit to raise). */
const CLIENT_SAFE_DETAIL_PREFIX = "BUDGET_EXCEEDED ";
const MAX_CLIENT_DETAIL_CHARS = 500;

export function clientSafeReason(reason: string): string {
	return SAFE_REASON_PATTERN.test(reason) ? reason : "unanswered";
}

export function clientSafeDetail(detail: unknown): string | undefined {
	if (typeof detail === "string" && detail.startsWith(CLIENT_SAFE_DETAIL_PREFIX)) {
		return detail.slice(0, MAX_CLIENT_DETAIL_CHARS);
	}
	return undefined;
}
