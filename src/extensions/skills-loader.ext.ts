import fs from "node:fs";
import path from "node:path";
import { defineExtension, section, type Extension } from "@earendil-works/pi-durable";
import { obj, str } from "@earendil-works/pi-schema";
import type { AppCapabilities } from "../contracts/app-capabilities.js";
import { ConversationMetaDoc } from "../harness/app-documents.js";
import { createTenantScopedTool } from "../tools/tool-pack.js";

export interface SkillFrontmatter {
	readonly name: string;
	readonly description: string;
	readonly version?: string;
	readonly tags: readonly string[];
	readonly filePath: string;
}

function parseFrontmatter(raw: string): { metadata: Record<string, string>; body: string } {
	const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
	if (!match) {
		return { metadata: {}, body: raw };
	}
	const yamlBlock = match[1] ?? "";
	const body = match[2] ?? "";
	const metadata: Record<string, string> = {};

	for (const line of yamlBlock.split(/\r?\n/)) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const colon = trimmed.indexOf(":");
		if (colon <= 0) continue;
		const key = trimmed.slice(0, colon).trim();
		const value = trimmed.slice(colon + 1).trim().replace(/^["']|["']$/g, "");
		metadata[key] = value;
	}

	return { metadata, body };
}

function parseInlineTags(raw: string | undefined): string[] {
	if (!raw) return [];
	const stripped = raw.replace(/^\[/, "").replace(/\]$/, "");
	return stripped
		.split(",")
		.map((item) => item.trim().replace(/^["']|["']$/g, ""))
		.filter((item) => item.length > 0);
}

/**
 * Discovers progressive-disclosure `SKILL.md` files under `skillsDir` and returns
 * frontmatter-only descriptors, filtering out any skills disabled by Skill Dojo.
 */
export function loadSkillCatalog(skillsDir: string, disabledSkills: readonly string[] = []): SkillFrontmatter[] {
	if (!skillsDir || !fs.existsSync(skillsDir)) {
		return [];
	}

	const disabledSet = new Set(disabledSkills);
	const found: SkillFrontmatter[] = [];

	const entries = fs.readdirSync(skillsDir, { withFileTypes: true });
	for (const entry of entries) {
		let skillFile: string | undefined;
		let fallbackName = entry.name;

		if (entry.isDirectory()) {
			const candidate = path.join(skillsDir, entry.name, "SKILL.md");
			if (fs.existsSync(candidate)) {
				skillFile = candidate;
			}
		} else if (entry.isFile() && entry.name.toLowerCase().endsWith(".skill.md")) {
			skillFile = path.join(skillsDir, entry.name);
			fallbackName = entry.name.replace(/\.skill\.md$/i, "");
		}

		if (!skillFile) continue;
		const raw = fs.readFileSync(skillFile, "utf8");
		const { metadata } = parseFrontmatter(raw);
		const name = metadata.name || fallbackName;
		if (disabledSet.has(name)) continue;

		found.push({
			name,
			description: metadata.description || "No description provided.",
			...(metadata.version ? { version: metadata.version } : {}),
			tags: parseInlineTags(metadata.tags),
			filePath: skillFile,
		});
	}

	found.sort((a, b) => a.name.localeCompare(b.name));
	return found;
}

export function formatFrozenSkillsIndex(skills: readonly SkillFrontmatter[]): string {
	if (skills.length === 0) return "";
	const lines = skills.map((s) => {
		const tags = s.tags.length > 0 ? ` [${s.tags.join(", ")}]` : "";
		return `- ${s.name}: ${s.description}${tags}`;
	});
	return [
		"Available skills (progressive disclosure — call `skill_read` to load full instructions when relevant):",
		...lines,
	].join("\n");
}

export function readSkillBody(skillsDir: string, skillName: string): { meta: SkillFrontmatter; body: string } | undefined {
	const catalog = loadSkillCatalog(skillsDir, []);
	const match = catalog.find((s) => s.name === skillName);
	if (!match) return undefined;
	const raw = fs.readFileSync(match.filePath, "utf8");
	const { body } = parseFrontmatter(raw);
	return { meta: match, body: body.trim() };
}

const SkillReadParams = obj({
	name: str({ description: "Exact skill name from the skills_index section." }),
});

/**
 * Isolated Extension: Progressive-disclosure SKILL.md index + health-aware disabling.
 *
 * The `skills_index` section reads the session-start frozen index from
 * `ConversationMetaDoc` for KV-cache stability. `skill_read` dynamically checks
 * Skill Dojo disabled status before revealing a skill's body.
 */
export function createSkillsLoaderExtension(capabilities: AppCapabilities): Extension {
	const skillReadTool = createTenantScopedTool<
		typeof SkillReadParams,
		{ skillName: string; disabled: boolean; version: string }
	>({
		name: "skill_read",
		description: "Load the full instructions of an enabled SKILL.md by name. Safe to replay.",
		parameters: SkillReadParams,
		replay: "safe",
		execute: async (args, tenant) => {
			const disabled = await capabilities.telemetry.listDisabledSkills(tenant.tenantId);
			if (disabled.includes(args.name)) {
				return {
					isError: true,
					output: [
						{
							type: "text",
							text: `Skill "${args.name}" is currently disabled by Skill Dojo health policy.`,
						},
					],
					details: { skillName: args.name, disabled: true, version: "" },
				};
			}

			const loaded = readSkillBody(capabilities.skillsDir, args.name);
			if (!loaded) {
				return {
					isError: true,
					output: [{ type: "text", text: `Skill "${args.name}" not found in "${capabilities.skillsDir}".` }],
					details: { skillName: args.name, disabled: false, version: "" },
				};
			}

			return {
				output: [{ type: "text", text: `# Skill: ${loaded.meta.name}\n\n${loaded.body}` }],
				details: { skillName: loaded.meta.name, disabled: false, version: loaded.meta.version ?? "1" },
			};
		},
	});

	return defineExtension({
		name: "skills-loader",
		sections: [
			section(
				"skills_index",
				async (input, context) => {
					const meta = await input.read.snapshot(ConversationMetaDoc, input.conversationId, context);
					const frozen = meta?.frozenSkillsIndex?.trim();
					return frozen && frozen.length > 0 ? frozen : undefined;
				},
				{ tag: true },
			),
		],
		tools: [skillReadTool],
	});
}
