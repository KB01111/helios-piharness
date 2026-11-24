import fs from "node:fs";
import path from "node:path";
import { loadSkillCatalog } from "../extensions/skills-loader.ext.js";

/** Directory (inside the skills dir) that holds draft skills awaiting approval. Ignored by `loadSkillCatalog`. */
export const PROPOSED_SKILLS_DIRNAME = "_proposed";

const MAX_SKILL_NAME_LENGTH = 64;

export interface SkillDraft {
	readonly suggestedSkillName: string;
	readonly draftSkillMarkdown: string;
	readonly id?: string;
	readonly support?: number;
}

export interface SkillProposalProvenance {
	readonly tenantId: string;
	readonly source: string;
	readonly proposedAt: number;
	readonly taskId?: string;
}

export interface ProposedSkillRecord {
	readonly name: string;
	readonly workflowId?: string;
	/** Absolute path of the written SKILL.md. */
	readonly filePath: string;
}

export interface ProposeSkillsOptions {
	readonly provenance: SkillProposalProvenance;
	/** Maximum number of NEW drafts written per call. Default: 5 */
	readonly maxNew?: number;
	/** Names that must never be proposed (e.g. skills disabled by Skill Dojo). */
	readonly excludeNames?: readonly string[];
}

export interface ProposeSkillsResult {
	readonly created: readonly ProposedSkillRecord[];
	readonly skipped: readonly { readonly name: string; readonly reason: string }[];
}

/**
 * Reduces an arbitrary suggestion to a safe single path segment: lowercase
 * `[a-z0-9-]`, no leading/trailing dashes, at most 64 chars. Returns
 * `undefined` if nothing usable remains. This makes path traversal
 * (`../x`, absolute paths, `_proposed` shadowing) impossible by construction.
 */
export function sanitizeSkillName(raw: string): string | undefined {
	const cleaned = raw
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, MAX_SKILL_NAME_LENGTH)
		.replace(/-+$/g, "");
	return cleaned.length > 0 ? cleaned : undefined;
}

function oneLine(value: string): string {
	return value.replace(/[\r\n]+/g, " ").trim();
}

function proposedRoot(skillsDir: string): string {
	return path.join(skillsDir, PROPOSED_SKILLS_DIRNAME);
}

/** Resolves `<skillsDir>/_proposed/<name>` and asserts it cannot escape the proposed root. */
function proposedDirFor(skillsDir: string, safeName: string): string {
	const root = path.resolve(proposedRoot(skillsDir));
	const dir = path.resolve(root, safeName);
	if (path.dirname(dir) !== root) {
		throw new Error(`Refusing skill path outside proposal root: "${safeName}"`);
	}
	return dir;
}

function activeDirFor(skillsDir: string, safeName: string): string {
	const root = path.resolve(skillsDir);
	const dir = path.resolve(root, safeName);
	if (path.dirname(dir) !== root) {
		throw new Error(`Refusing skill path outside skills dir: "${safeName}"`);
	}
	return dir;
}

/**
 * Inserts provenance + `status: proposed` into the draft's YAML frontmatter
 * (keeping `---` as the very first line so the file stays loadable once approved).
 */
function withProvenance(draftMarkdown: string, name: string, provenance: SkillProposalProvenance): string {
	const lines = [
		"status: proposed",
		`provenance: ${oneLine(provenance.source)}`,
		`proposed-tenant: ${oneLine(provenance.tenantId)}`,
		`proposed-at: ${new Date(provenance.proposedAt).toISOString()}`,
		...(provenance.taskId ? [`proposed-by-task: ${oneLine(provenance.taskId)}`] : []),
	];
	const normalized = draftMarkdown.replace(/\r\n/g, "\n");
	const match = normalized.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
	if (!match) {
		return ["---", `name: ${name}`, "description: Proposed skill draft.", ...lines, "---", "", normalized].join("\n");
	}
	const frontmatter = (match[1] ?? "")
		.split("\n")
		.map((line) => (/^name\s*:/.test(line.trim()) ? `name: ${name}` : line))
		.join("\n");
	return ["---", frontmatter, ...lines, "---", match[2] ?? ""].join("\n");
}

function existingSkillNames(skillsDir: string): Set<string> {
	const names = new Set<string>();
	for (const skill of loadSkillCatalog(skillsDir, [])) {
		names.add(skill.name);
		const safe = sanitizeSkillName(skill.name);
		if (safe) names.add(safe);
	}
	return names;
}

/**
 * Writes mined workflow drafts to `<skillsDir>/_proposed/<name>/SKILL.md`.
 *
 * - Never touches the active catalog (approval via `promoteProposedSkill` is required).
 * - Idempotent: an already proposed or already active skill is never rewritten or duplicated.
 * - Names are sanitized to a single safe path segment.
 */
export function proposeSkillDrafts(
	skillsDir: string,
	drafts: readonly SkillDraft[],
	options: ProposeSkillsOptions,
): ProposeSkillsResult {
	const created: ProposedSkillRecord[] = [];
	const skipped: { name: string; reason: string }[] = [];
	if (!skillsDir) return { created, skipped };

	const maxNew = Math.max(0, options.maxNew ?? 5);
	const excluded = new Set(options.excludeNames ?? []);
	const active = existingSkillNames(skillsDir);
	const seen = new Set<string>();

	for (const draft of drafts) {
		const name = sanitizeSkillName(draft.suggestedSkillName);
		if (!name) {
			skipped.push({ name: draft.suggestedSkillName, reason: "invalid-name" });
			continue;
		}
		if (seen.has(name)) {
			skipped.push({ name, reason: "duplicate-in-batch" });
			continue;
		}
		seen.add(name);
		if (excluded.has(name) || excluded.has(draft.suggestedSkillName)) {
			skipped.push({ name, reason: "excluded" });
			continue;
		}
		if (active.has(name) || fs.existsSync(activeDirFor(skillsDir, name))) {
			skipped.push({ name, reason: "already-active" });
			continue;
		}
		const dir = proposedDirFor(skillsDir, name);
		const filePath = path.join(dir, "SKILL.md");
		// Test the artifacts, not the directory: an empty/leftover directory (e.g. a crash right after mkdir)
		// must not block the proposal forever. A REJECTED tombstone keeps a rejected draft from returning.
		if (fs.existsSync(filePath) || fs.existsSync(path.join(dir, "REJECTED"))) {
			skipped.push({ name, reason: "already-proposed" });
			continue;
		}
		if (created.length >= maxNew) {
			skipped.push({ name, reason: "cycle-limit" });
			continue;
		}

		fs.mkdirSync(dir, { recursive: true });
		if (!writeFileExclusiveAtomic(filePath, withProvenance(draft.draftSkillMarkdown, name, options.provenance))) {
			skipped.push({ name, reason: "already-proposed" });
			continue;
		}
		created.push({ name, ...(draft.id ? { workflowId: draft.id } : {}), filePath });
	}

	return { created, skipped };
}

/**
 * Writes `content` to `filePath` via a temp file in the same directory so readers never observe a partial
 * SKILL.md. Never overwrites: returns false when `filePath` already exists (a concurrent writer won).
 */
function writeFileExclusiveAtomic(filePath: string, content: string): boolean {
	const tmpPath = `${filePath}.${process.pid}.${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}.tmp`;
	try {
		fs.writeFileSync(tmpPath, content, { encoding: "utf8", flag: "wx" });
		try {
			// A hard link is atomic AND exclusive (fails with EEXIST instead of replacing the target).
			fs.linkSync(tmpPath, filePath);
			return true;
		} catch (error) {
			const code = (error as NodeJS.ErrnoException).code;
			if (code === "EEXIST") return false;
			// Filesystem without hard links: fall back to rename (atomic replace) after an existence check.
			if (fs.existsSync(filePath)) return false;
			fs.renameSync(tmpPath, filePath);
			return true;
		}
	} finally {
		fs.rmSync(tmpPath, { force: true });
	}
}

/** Names of skills currently waiting for approval. */
export function listProposedSkills(skillsDir: string): string[] {
	const root = proposedRoot(skillsDir);
	if (!skillsDir || !fs.existsSync(root)) return [];
	return fs
		.readdirSync(root, { withFileTypes: true })
		.filter((entry) => entry.isDirectory() && fs.existsSync(path.join(root, entry.name, "SKILL.md")))
		.map((entry) => entry.name)
		.sort();
}

/**
 * Approves a proposed skill by moving it into the active catalog. This is an
 * explicit, operator-invoked step and is intentionally NOT exposed over HTTP.
 *
 * Throws if the name is invalid, no proposal exists, or an active skill with
 * that name already exists (never overwrites).
 */
export function promoteProposedSkill(skillsDir: string, name: string): { readonly name: string; readonly filePath: string } {
	const safe = sanitizeSkillName(name);
	if (!safe || safe !== name) {
		throw new Error(`Invalid proposed skill name "${name}"`);
	}
	const source = proposedDirFor(skillsDir, safe);
	const sourceFile = path.join(source, "SKILL.md");
	if (!fs.existsSync(sourceFile)) {
		throw new Error(`No proposed skill named "${safe}"`);
	}
	const target = activeDirFor(skillsDir, safe);
	if (fs.existsSync(target) || existingSkillNames(skillsDir).has(safe)) {
		throw new Error(`Active skill "${safe}" already exists; refusing to overwrite`);
	}

	const approved = fs
		.readFileSync(sourceFile, "utf8")
		.replace(/^status:\s*proposed\s*$/m, `status: approved\napproved-at: ${new Date().toISOString()}`);
	const approvedTmp = `${sourceFile}.${process.pid}.${Date.now().toString(36)}.tmp`;
	try {
		fs.writeFileSync(approvedTmp, approved, "utf8");
		fs.renameSync(approvedTmp, sourceFile);
	} finally {
		fs.rmSync(approvedTmp, { force: true });
	}
	fs.renameSync(source, target);
	return { name: safe, filePath: path.join(target, "SKILL.md") };
}

/**
 * Rejects a proposal: removes its SKILL.md but leaves a tombstone so the same
 * draft is not re-proposed by the next sleep cycle. Returns whether it existed.
 */
export function rejectProposedSkill(skillsDir: string, name: string): boolean {
	const safe = sanitizeSkillName(name);
	if (!safe || safe !== name) return false;
	const dir = proposedDirFor(skillsDir, safe);
	const file = path.join(dir, "SKILL.md");
	if (!fs.existsSync(file)) return false;
	fs.rmSync(file, { force: true });
	fs.writeFileSync(path.join(dir, "REJECTED"), `rejected-at: ${new Date().toISOString()}\n`, "utf8");
	return true;
}
