import { defineExtension, section, type Extension } from "@earendil-works/pi-durable";
import { ConversationMetaDoc } from "../harness/app-documents.js";

export const DEFAULT_STATIC_SYSTEM_PROMPT = [
	"You are a durable, tool-using assistant running inside the Helios Pi Harness.",
	"Follow these core operating principles:",
	"1. Use tools deliberately and prefer idempotent read tools before mutating state.",
	"2. Retrieve long-term user memories on demand with `memory_search`; never assume unverified facts.",
	"3. When a tool output is offloaded to an artifact, use `artifact_read` to inspect relevant slices.",
	"4. For high-impact actions requiring human sign-off, use `domain_request_approval` and wait for the decision.",
	"5. Read relevant skills via `skill_read` only when a listed skill matches the current task.",
	"6. Be concise, accurate, and transparent about errors or authorization blocks.",
	"7. For a task with 3 or more steps, call `plan_set` first, keep it current with `plan_update` (a blocked step needs a reason), and use `plan_view` to re-orient after long tool sequences.",
].join("\n");

export interface CorePromptExtensionOptions {
	readonly staticSystemPrompt?: string;
}

/**
 * Isolated Extension: Static system prompt + session-start frozen Core Memory section.
 *
 * KV-Cache Stability Invariant:
 * - `core_instructions` is strictly constant (no timestamps or random values).
 * - `core_memory` reads ONLY the immutable `frozenCoreMemory` string committed to
 *   `ConversationMetaDoc` at conversation start. Mutating SQLite memories or calling
 *   `memory_search` mid-session NEVER alters this section, keeping the leading
 *   system message prefix byte-identical across turns.
 */
export function createCorePromptExtension(options?: CorePromptExtensionOptions): Extension {
	const staticPrompt = options?.staticSystemPrompt ?? DEFAULT_STATIC_SYSTEM_PROMPT;

	return defineExtension({
		name: "core-prompt",
		sections: [
			section("core_instructions", () => staticPrompt, { tag: true }),
			section(
				"core_memory",
				async (input, context) => {
					const meta = await input.read.snapshot(ConversationMetaDoc, input.conversationId, context);
					const frozen = meta?.frozenCoreMemory?.trim();
					return frozen && frozen.length > 0 ? frozen : undefined;
				},
				{ tag: true },
			),
		],
	});
}
