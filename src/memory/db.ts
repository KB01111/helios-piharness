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
