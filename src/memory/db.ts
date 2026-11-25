import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

export interface AppDatabaseOptions {
	readonly dbPath: string;
}

interface Migration {
	readonly version: number;
	readonly name: string;
	readonly sql: string;
	readonly run?: (db: DatabaseSync) => void;
}

function ensureColumn(db: DatabaseSync, table: string, column: string, definition: string): void {
	const cols = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
	if (cols.some((c) => c.name === column)) return;
	db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition};`);
}

const MIGRATIONS: readonly Migration[] = [
	{
		version: 1,
		name: "initial_app_tables",
		sql: `
			CREATE TABLE IF NOT EXISTS midterm_episodes (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				run_id TEXT NOT NULL,
				user_message TEXT NOT NULL,
				assistant_summary TEXT NOT NULL,
				tool_trace_json TEXT NOT NULL,
				friction_signals_json TEXT NOT NULL,
				friction_score REAL NOT NULL DEFAULT 0,
				consolidated INTEGER NOT NULL DEFAULT 0,
				created_at INTEGER NOT NULL,
				UNIQUE(tenant_id, user_id, conversation_id, run_id)
			);

			CREATE INDEX IF NOT EXISTS idx_midterm_episodes_lookup
				ON midterm_episodes (tenant_id, user_id, consolidated, created_at);

			CREATE TABLE IF NOT EXISTS longterm_memories (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				kind TEXT NOT NULL,
				mem_key TEXT NOT NULL,
				content TEXT NOT NULL,
				confidence REAL NOT NULL,
				pinned INTEGER NOT NULL DEFAULT 0,
				status TEXT NOT NULL DEFAULT 'active',
				superseded_by_id TEXT,
				evidence_count INTEGER NOT NULL DEFAULT 1,
				distinct_conversations INTEGER NOT NULL DEFAULT 1,
				source_conversation_id TEXT,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_longterm_memories_scope
				ON longterm_memories (tenant_id, user_id, status, mem_key);

			CREATE VIRTUAL TABLE IF NOT EXISTS longterm_memories_fts USING fts5(
				memory_id UNINDEXED,
				tenant_id UNINDEXED,
				user_id UNINDEXED,
				mem_key,
				content,
				kind
			);

			CREATE TABLE IF NOT EXISTS artifacts (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				content_type TEXT NOT NULL,
				byte_length INTEGER NOT NULL,
				source_tool TEXT,
				conversation_id TEXT,
				file_path TEXT,
				inline_content TEXT,
				created_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_artifacts_scope
				ON artifacts (tenant_id, user_id, created_at);

			CREATE TABLE IF NOT EXISTS tool_invocations (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				call_id TEXT NOT NULL,
				tool_name TEXT NOT NULL,
				skill_name TEXT,
				is_error INTEGER NOT NULL DEFAULT 0,
				diagnostic_code TEXT,
				duration_ms REAL,
				created_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_tool_invocations_scope
				ON tool_invocations (tenant_id, tool_name, created_at);

			CREATE TABLE IF NOT EXISTS run_telemetry (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				run_id TEXT NOT NULL,
				tool_names_json TEXT NOT NULL,
				friction_score REAL NOT NULL DEFAULT 0,
				friction_kinds_json TEXT NOT NULL,
				created_at INTEGER NOT NULL,
				UNIQUE(tenant_id, user_id, conversation_id, run_id)
			);

			CREATE INDEX IF NOT EXISTS idx_run_telemetry_scope
				ON run_telemetry (tenant_id, user_id, created_at);

			CREATE TABLE IF NOT EXISTS skill_metrics (
				tenant_id TEXT NOT NULL,
				name TEXT NOT NULL,
				kind TEXT NOT NULL,
				calls INTEGER NOT NULL DEFAULT 0,
				successes INTEGER NOT NULL DEFAULT 0,
				errors INTEGER NOT NULL DEFAULT 0,
				total_duration_ms REAL NOT NULL DEFAULT 0,
				last_used_at INTEGER NOT NULL,
				PRIMARY KEY (tenant_id, name, kind)
			);

			CREATE TABLE IF NOT EXISTS disabled_skills (
				tenant_id TEXT NOT NULL,
				skill_name TEXT NOT NULL,
				reason TEXT NOT NULL,
				disabled_at INTEGER NOT NULL,
				PRIMARY KEY (tenant_id, skill_name)
			);

			CREATE TABLE IF NOT EXISTS sim_log (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				target_module TEXT NOT NULL,
				hypothesis TEXT NOT NULL,
				patch_summary TEXT NOT NULL,
				outcome TEXT NOT NULL,
				evolve_delta_json TEXT NOT NULL,
				held_out_delta_json TEXT NOT NULL,
				regression_detected INTEGER NOT NULL DEFAULT 0,
				notes TEXT,
				created_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_sim_log_module
				ON sim_log (tenant_id, target_module, created_at);
		`,
	},
	{
		version: 2,
		name: "memory_evidence_and_embeddings",
		sql: `
			-- Evidence ledger: which episodes contributed to which memory (idempotent evidence counting).
			CREATE TABLE IF NOT EXISTS memory_evidence (
				memory_id TEXT NOT NULL,
				episode_id TEXT NOT NULL,
				created_at INTEGER NOT NULL,
				PRIMARY KEY (memory_id, episode_id)
			);

			-- Optional vector store for hybrid retrieval (filled lazily when an EmbeddingProvider is injected).
			CREATE TABLE IF NOT EXISTS memory_embeddings (
				memory_id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				model TEXT NOT NULL,
				dim INTEGER NOT NULL,
				content_hash TEXT NOT NULL,
				vector BLOB NOT NULL,
				updated_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_memory_embeddings_scope
				ON memory_embeddings (tenant_id, user_id, model);
		`,
	},
	{
		version: 3,
		name: "memory_graph_temporal_and_utility",
		sql: `
			-- Tenant-scoped entity vocabulary extracted deterministically from memory keys & content.
			CREATE TABLE IF NOT EXISTS memory_entities (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				canonical_name TEXT NOT NULL,
				display_name TEXT NOT NULL,
				entity_type TEXT NOT NULL DEFAULT 'entity',
				updated_at INTEGER NOT NULL,
				UNIQUE(tenant_id, user_id, canonical_name)
			);

			CREATE INDEX IF NOT EXISTS idx_memory_entities_scope
				ON memory_entities (tenant_id, user_id, canonical_name);

			-- Bipartite links connecting memories to their canonical entities (used for multi-hop PPR).
			CREATE TABLE IF NOT EXISTS memory_entity_links (
				memory_id TEXT NOT NULL,
				entity_id TEXT NOT NULL,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				weight REAL NOT NULL DEFAULT 1.0,
				PRIMARY KEY (memory_id, entity_id)
			);

			CREATE INDEX IF NOT EXISTS idx_memory_entity_links_scope
				ON memory_entity_links (tenant_id, user_id, entity_id);
			CREATE INDEX IF NOT EXISTS idx_memory_entity_links_memory
				ON memory_entity_links (memory_id);

			-- Lightweight bi-temporal entity-to-entity relations backed by supporting memories.
			CREATE TABLE IF NOT EXISTS memory_relations (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				source_entity_id TEXT NOT NULL,
				target_entity_id TEXT NOT NULL,
				relation TEXT NOT NULL,
				memory_id TEXT NOT NULL,
				valid_from INTEGER,
				valid_until INTEGER,
				weight REAL NOT NULL DEFAULT 1.0
			);

			CREATE INDEX IF NOT EXISTS idx_memory_relations_scope
				ON memory_relations (tenant_id, user_id, source_entity_id);
			CREATE INDEX IF NOT EXISTS idx_memory_relations_memory
				ON memory_relations (memory_id);

			-- Explicit calendar/date intervals extracted from memory content for temporal-window search.
			CREATE TABLE IF NOT EXISTS memory_temporal_anchors (
				id TEXT PRIMARY KEY,
				memory_id TEXT NOT NULL,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				start_ms INTEGER NOT NULL,
				end_ms INTEGER NOT NULL,
				precision TEXT NOT NULL,
				label TEXT NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_memory_temporal_scope
				ON memory_temporal_anchors (tenant_id, user_id, start_ms, end_ms);
			CREATE INDEX IF NOT EXISTS idx_memory_temporal_memory
				ON memory_temporal_anchors (memory_id);

			-- FTS5 index over recent unconsolidated mid-term episodes for immediate session recall.
			CREATE VIRTUAL TABLE IF NOT EXISTS midterm_episodes_fts USING fts5(
				episode_id UNINDEXED,
				tenant_id UNINDEXED,
				user_id UNINDEXED,
				user_message,
				assistant_summary
			);
		`,
		run: (db: DatabaseSync) => {
			ensureColumn(db, "longterm_memories", "valid_from", "INTEGER");
			ensureColumn(db, "longterm_memories", "valid_until", "INTEGER");
			ensureColumn(db, "longterm_memories", "access_count", "INTEGER NOT NULL DEFAULT 0");
			ensureColumn(db, "longterm_memories", "last_accessed_at", "INTEGER");
			ensureColumn(db, "longterm_memories", "feedback_weight", "REAL NOT NULL DEFAULT 0.5");

			db.exec(`
				UPDATE longterm_memories SET valid_from = created_at WHERE valid_from IS NULL;
				UPDATE longterm_memories SET valid_until = updated_at WHERE status = 'superseded' AND valid_until IS NULL;
				INSERT INTO midterm_episodes_fts (episode_id, tenant_id, user_id, user_message, assistant_summary)
				SELECT e.id, e.tenant_id, e.user_id, e.user_message, e.assistant_summary
				FROM midterm_episodes e
				WHERE e.consolidated = 0
				  AND NOT EXISTS (SELECT 1 FROM midterm_episodes_fts f WHERE f.episode_id = e.id);
			`);
		},
	},
	{
		version: 4,
		name: "permission_audit_and_tool_gate_approvals",
		sql: `
			CREATE TABLE IF NOT EXISTS permission_audit_log (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				task_id TEXT NOT NULL,
				call_id TEXT NOT NULL,
				tool_name TEXT NOT NULL,
				tier TEXT NOT NULL,
				decision TEXT NOT NULL,
				reason TEXT,
				reviewer TEXT,
				created_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_permission_audit_tenant
				ON permission_audit_log (tenant_id, conversation_id, id DESC);

			CREATE TABLE IF NOT EXISTS pending_tool_gate_approvals (
				task_id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				conversation_id TEXT NOT NULL,
				call_id TEXT NOT NULL,
				tool_name TEXT NOT NULL,
				action TEXT NOT NULL,
				summary TEXT NOT NULL,
				signal_name TEXT NOT NULL,
				signal_key TEXT NOT NULL,
				created_at INTEGER NOT NULL
			);

			CREATE INDEX IF NOT EXISTS idx_pending_tool_gate_scope
				ON pending_tool_gate_approvals (tenant_id, user_id, conversation_id, created_at);
		`,
	},
];

/**
 * Application SQLite database for memory, artifacts, telemetry, and learning loops.
 *
 * Uses Node's built-in `node:sqlite` (`DatabaseSync`) with disjoint table names
 * from Pi Durable's `pi_*` schema.
 */
export class AppDatabase {
	readonly db: DatabaseSync;
	readonly dbPath: string;
	private closed = false;
	private txDepth = 0;

	private constructor(db: DatabaseSync, dbPath: string) {
		this.db = db;
		this.dbPath = dbPath;
	}

	static open(options: AppDatabaseOptions): AppDatabase {
		const { dbPath } = options;
		if (dbPath !== ":memory:") {
			fs.mkdirSync(path.dirname(dbPath), { recursive: true });
		}
		const db = new DatabaseSync(dbPath);
		try {
			db.exec("PRAGMA foreign_keys = ON;");
			db.exec("PRAGMA busy_timeout = 5000;");
			if (dbPath !== ":memory:") {
				db.exec("PRAGMA journal_mode = WAL;");
				db.exec("PRAGMA synchronous = NORMAL;");
			}

			const instance = new AppDatabase(db, dbPath);
			instance.migrate();
			return instance;
		} catch (error) {
			// Do not leak the file handle (and its WAL lock) when setup or a migration fails.
			try {
				db.close();
			} catch {
				// Already closed / never fully opened.
			}
			throw error;
		}
	}

	private migrate(): void {
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS app_migrations (
				version INTEGER PRIMARY KEY,
				name TEXT NOT NULL,
				applied_at INTEGER NOT NULL
			);
		`);

		const appliedRows = this.db
			.prepare("SELECT version FROM app_migrations ORDER BY version ASC")
			.all() as Array<{ version: number }>;
		const applied = new Set(appliedRows.map((row) => Number(row.version)));

		for (const migration of MIGRATIONS) {
			if (applied.has(migration.version)) continue;
			this.transaction(() => {
				// Another process may have applied this version between the read above and acquiring the write
				// lock (BEGIN IMMEDIATE): re-check inside the transaction, and make the insert itself idempotent.
				const already = this.db
					.prepare("SELECT 1 AS ok FROM app_migrations WHERE version = ?")
					.get(migration.version);
				if (already) return;
				this.db.exec(migration.sql);
				migration.run?.(this.db);
				this.db
					.prepare("INSERT OR IGNORE INTO app_migrations (version, name, applied_at) VALUES (?, ?, ?)")
					.run(migration.version, migration.name, Date.now());
			});
		}
	}

	/**
	 * Runs `fn` atomically. Re-entrant: the outermost call opens `BEGIN IMMEDIATE`, nested calls use
	 * SAVEPOINTs so an inner failure rolls back only the inner work while the outer transaction can
	 * still commit (or roll back everything if the error propagates).
	 */
	transaction<T>(fn: () => T): T {
		this.assertOpen();
		if (this.txDepth === 0) {
			this.db.exec("BEGIN IMMEDIATE");
			this.txDepth = 1;
			try {
				const result = fn();
				this.db.exec("COMMIT");
				return result;
			} catch (error) {
				try {
					this.db.exec("ROLLBACK");
				} catch {
					// Ignore rollback errors after failed statement.
				}
				throw error;
			} finally {
				this.txDepth = 0;
			}
		}

		const name = `sp_${this.txDepth}`;
		this.db.exec(`SAVEPOINT ${name}`);
		this.txDepth += 1;
		try {
			const result = fn();
			this.db.exec(`RELEASE SAVEPOINT ${name}`);
			return result;
		} catch (error) {
			try {
				this.db.exec(`ROLLBACK TO SAVEPOINT ${name}`);
				this.db.exec(`RELEASE SAVEPOINT ${name}`);
			} catch {
				// Ignore rollback errors after failed statement.
			}
			throw error;
		} finally {
			this.txDepth -= 1;
		}
	}

	/** True while inside `transaction()`. */
	get inTransaction(): boolean {
		return this.txDepth > 0;
	}

	close(): void {
		if (this.closed) return;
		this.closed = true;
		this.db.close();
	}

	isClosed(): boolean {
		return this.closed;
	}

	private assertOpen(): void {
		if (this.closed) {
			throw new Error("AppDatabase is closed");
		}
	}
}
