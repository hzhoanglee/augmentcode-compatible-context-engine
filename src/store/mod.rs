pub mod ops;
pub mod schema;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, SurrealKv};
use tokio::sync::RwLock;

use crate::store::schema::SCHEMA_DDL;

/// Shared, process-wide map of one open SurrealDB handle per repo path.
///
/// A single datastore instance must back every read and write for a repo:
/// SurrealKV resolves its MVCC view when the datastore is opened, so two
/// instances on the same on-disk path do not observe each other's commits.
/// Indexer and server therefore share one handle per repo via this map.
pub type RepoDbMap = Arc<RwLock<HashMap<String, Surreal<Db>>>>;

/// Sanitize a repo path to a safe directory name (max 64 chars).
pub fn sanitize_repo_name(repo_path: &str) -> String {
    let sanitized: String = repo_path
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = sanitized.trim_matches('_');
    if trimmed.len() > 64 {
        trimmed[trimmed.len() - 64..].to_string()
    } else {
        trimmed.to_string()
    }
}

/// Return the SurrealDB data directory for a given repo.
pub fn db_path(home_dir: &Path, repo_path: &str) -> PathBuf {
    let name = sanitize_repo_name(repo_path);
    home_dir
        .join(".vibervn")
        .join("context-engine")
        .join("surreal")
        .join(name)
}

/// Open (or create) a SurrealDB RocksDB database for the given repo.
/// Runs schema DDL to ensure all tables/indexes exist.
pub async fn open_db(home_dir: &Path, repo_path: &str) -> Result<Surreal<Db>> {
    let path = db_path(home_dir, repo_path);
    std::fs::create_dir_all(&path).with_context(|| format!("create db dir {:?}", path))?;

    let db = Surreal::new::<SurrealKv>(path.to_str().unwrap())
        .await
        .context("open surrealdb")?;

    db.use_ns("context_engine")
        .use_db(sanitize_repo_name(repo_path))
        .await
        .context("select ns/db")?;

    // Apply schema DDL.
    //
    // `DEFINE FIELD OVERWRITE` is used for all fields so that re-running this DDL
    // on an existing database actively REPLACES the persisted field definition with
    // the current one. A plain `DEFINE FIELD` (without OVERWRITE) is a no-op on a
    // field that already exists — the old type stays in the datastore and any write
    // that uses the corrected type is rejected at runtime with a FieldCheck error,
    // silently rolling back the whole transaction.
    //
    // Tables use `IF NOT EXISTS` (safe — does not drop existing rows).
    // Indexes use `IF NOT EXISTS` (avoids unnecessary index rebuilds on re-open).
    //
    // `.check()` is called on the response so that any DDL statement error is
    // surfaced immediately rather than swallowed.
    db.query(SCHEMA_DDL)
        .await
        .context("apply schema DDL")?
        .check()
        .context("schema DDL contained errors")?;

    Ok(db)
}

/// Return the shared `Surreal<Db>` handle for `repo`, opening and caching it on
/// first use. All callers share one datastore instance per repo so reads see
/// writes (see [`RepoDbMap`]).
///
/// The fast path takes only a read lock. On a miss we open the DB *before*
/// taking the write lock (open is the slow part and must not block readers),
/// then re-check under the write lock so a racing opener never replaces a
/// handle that is already cached.
pub async fn get_or_open(
    repo_dbs: &RepoDbMap,
    home_dir: &Path,
    repo: &str,
) -> Result<Surreal<Db>> {
    if let Some(db) = repo_dbs.read().await.get(repo) {
        return Ok(db.clone());
    }
    let db = open_db(home_dir, repo).await?;
    let mut map = repo_dbs.write().await;
    if let Some(existing) = map.get(repo) {
        return Ok(existing.clone());
    }
    map.insert(repo.to_string(), db.clone());
    Ok(db)
}

#[cfg(test)]
mod isolation_repro {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn concurrent_handles_isolation_then_shared_fix() {
        // ── PART 1: does isolation exist? ────────────────────────────────────
        let home = TempDir::new().unwrap();
        let repo = "/proj/repo_iso";

        // Open handle A (fresh DB).
        let a = open_db(home.path(), repo).await.expect("open A");
        assert_eq!(
            ops::count_chunks(&a).await.unwrap(),
            0,
            "fresh DB must be empty"
        );

        // Open handle B on the SAME on-disk path while A is still alive.
        let b = open_db(home.path(), repo).await.expect("open B");

        // Write+commit one chunk through B.
        b.query(
            "CREATE chunk SET file = '/x/f.rs', line_start = 1, line_end = 2, \
             content = 'x', embedding = [0.1, 0.2, 0.3, 0.4], symbol_ref = NONE;",
        )
        .await
        .expect("write chunk via B");

        // B must see its own write — if not, the experiment is invalid.
        assert_eq!(
            ops::count_chunks(&b).await.unwrap(),
            1,
            "B must see its own write (write failed if this is 0)"
        );

        // Re-query the still-open A — does it see B's commit?
        let a_after = ops::count_chunks(&a).await.unwrap();
        println!(
            "ISOLATION PROBE: A reads {a_after} after B committed 1 \
             (0 => isolation confirmed, 1 => no isolation / hypothesis WRONG)"
        );

        // ── PART 2: does the shared handle (the proposed fix) work? ──────────
        let map: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));

        // Both calls resolve to the SAME cached instance.
        let sa = get_or_open(&map, home.path(), repo).await.expect("shared A");
        let sb = get_or_open(&map, home.path(), repo).await.expect("shared B");

        // Write another chunk through sb (total on disk will be 2 after this).
        sb.query(
            "CREATE chunk SET file = '/x/f.rs', line_start = 3, line_end = 4, \
             content = 'y', embedding = [0.5, 0.6, 0.7, 0.8], symbol_ref = NONE;",
        )
        .await
        .expect("write chunk via shared B");

        let sa_after = ops::count_chunks(&sa).await.unwrap();
        println!(
            "SHARED PROBE: shared-A reads {sa_after} after shared-B committed another chunk"
        );

        // ── Assertions ───────────────────────────────────────────────────────
        assert_eq!(
            a_after,
            0,
            "EXPECTED isolation: a still-open separate handle must NOT see another instance's \
             commit. If this fails (A == 1), the isolation hypothesis is WRONG and the real bug \
             is elsewhere (read-path ns/db context)."
        );

        // sa and sb are the same cached instance; part 1 left 1 chunk on disk,
        // part 2 wrote 1 more through the shared instance → shared-A should see 2.
        assert_eq!(
            sa_after,
            2,
            "shared handle must see writes made through the same cached instance"
        );
    }
}

// ─── Stale-schema regression ──────────────────────────────────────────────
//
// This module proves that `DEFINE FIELD OVERWRITE` correctly migrates an existing
// database whose field was created with the OLD type (`option<record<symbol>>`).
//
// WITHOUT the OVERWRITE fix (plain `DEFINE FIELD`):
//   - Re-applying the corrected DDL is a no-op: the on-disk type stays as
//     `option<record<symbol>>`.
//   - Attempting to write a quoted-string `symbol_ref` value fails with:
//       "Found '<string>' for field `symbol_ref`, ... but expected a
//       option<record<symbol>>"
//   - The whole transaction rolls back silently.
//
// WITH the OVERWRITE fix:
//   - Re-applying the DDL updates the persisted type to `option<string>`.
//   - The same quoted-string write commits successfully (count = 1).
//
// This is the exact scenario for every on-disk SurrealKV database that was
// created before the `parent`/`symbol_ref` type correction — which is why the
// bug only appeared on existing deployments, not on fresh installs.
#[cfg(test)]
mod stale_schema {
    use surrealdb::Surreal;
    use surrealdb::engine::local::{Db, SurrealKv};
    use tempfile::TempDir;

    use crate::store::schema::SCHEMA_DDL;
    use crate::store::ops::count_chunks;

    /// Open a raw SurrealKV DB (no DDL applied) on a TempDir.
    /// The caller is responsible for applying whatever schema it needs.
    async fn open_raw_db(dir: &std::path::Path, name: &str) -> Surreal<Db> {
        let path = dir.join(name);
        std::fs::create_dir_all(&path).unwrap();
        let db = Surreal::new::<SurrealKv>(path.to_str().unwrap())
            .await
            .expect("open raw db");
        db.use_ns("context_engine").use_db(name).await.expect("ns/db");
        db
    }

    /// Retrieve the INFO FOR TABLE result for `table` as a raw JSON string.
    /// Used to inspect the persisted field definition before and after DDL re-application.
    async fn info_for_table(db: &Surreal<Db>, table: &str) -> String {
        let result: Option<serde_json::Value> = db
            .query(format!("INFO FOR TABLE {table};"))
            .await
            .expect("INFO FOR TABLE")
            .take(0)
            .ok()
            .flatten();
        format!("{result:?}")
    }

    /// STEP 1 (RED → GREEN):
    ///
    /// 1. Force the datastore into the STALE state: apply OLD DDL declaring
    ///    `symbol_ref` and `parent` as `option<record<symbol>>`.
    /// 2. Inspect the persisted type via `INFO FOR TABLE` — confirms old type is in place.
    /// 3. Re-apply the CURRENT corrected `SCHEMA_DDL` (with OVERWRITE).
    /// 4. Inspect again — with OVERWRITE the type MUST now read `option<string>`.
    /// 5. Attempt the real writer's statement (quoted-string `symbol_ref` inside a txn).
    /// 6. Assert the write COMMITS and count = 1.
    ///
    /// This test FAILS without `DEFINE FIELD OVERWRITE` (plain re-DEFINE is a no-op,
    /// the FieldCheck error still triggers) and PASSES with OVERWRITE.
    #[tokio::test]
    async fn overwrite_migrates_stale_schema_and_write_commits() {
        let home = TempDir::new().unwrap();
        let db = open_raw_db(home.path(), "stale_repro").await;

        // ── 1. Install the OLD (stale) schema ────────────────────────────────
        // This mirrors what every pre-fix on-disk database had: both critical
        // fields declared as `option<record<symbol>>`.
        let old_ddl = "\
            DEFINE TABLE chunk SCHEMAFULL;\
            DEFINE FIELD symbol_ref ON chunk TYPE option<record<symbol>>;\
            DEFINE TABLE symbol SCHEMAFULL;\
            DEFINE FIELD parent ON symbol TYPE option<record<symbol>>;";
        db.query(old_ddl)
            .await
            .expect("install old stale DDL")
            .check()
            .expect("old DDL must not err");

        // ── 2. Confirm the old type is persisted ─────────────────────────────
        let before = info_for_table(&db, "chunk").await;
        println!("STALE-SCHEMA INFO BEFORE re-apply:\n  chunk: {before}");
        // The persisted definition must contain `record<symbol>` or `record` to
        // confirm the stale state is actually in place.
        assert!(
            before.to_lowercase().contains("record"),
            "before re-apply, the stale type must contain 'record' — got: {before}"
        );

        // ── 3. Re-apply the corrected SCHEMA_DDL (with DEFINE FIELD OVERWRITE) ──
        db.query(SCHEMA_DDL)
            .await
            .expect("corrected DDL must not return transport error")
            .check()
            .expect("corrected DDL must have no per-statement errors");

        // ── 4. Confirm the type has been updated ─────────────────────────────
        let after = info_for_table(&db, "chunk").await;
        println!("STALE-SCHEMA INFO AFTER re-apply:\n  chunk: {after}");
        // After OVERWRITE, `record<symbol>` must be gone from symbol_ref's definition.
        // The field info from SurrealDB contains the type string; we check that
        // the old record-type reference is no longer present.
        assert!(
            !after.contains("record<symbol>"),
            "after re-apply with OVERWRITE, 'record<symbol>' must be gone from the \
             field definition — OVERWRITE did not update the persisted type. Got: {after}"
        );

        // ── 5. Attempt the real writer's statement (mirroring pipeline.rs) ───
        let txn = "BEGIN TRANSACTION;\n\
            CREATE chunk SET \
              file = '/x/config.rs', \
              line_start = 1, \
              line_end = 10, \
              content = 'impl EmbeddingConfig {}', \
              embedding = [0.0, 1.0, 0.5], \
              symbol_ref = 'symbol:⟨config.rs::impl_EmbeddingConfig⟩';\n\
            COMMIT TRANSACTION;\n";

        let mut resp = db.query(txn).await.expect(".await must not fail");
        let errors = resp.take_errors();
        println!(
            "STALE-SCHEMA WRITE RESULT: errors = {errors:?}"
        );

        const GENERIC: &str = "The query was not executed due to a failed transaction";
        let real_error: Vec<_> = errors
            .iter()
            .filter(|(_, e)| !e.to_string().contains(GENERIC))
            .collect();
        println!("STALE-SCHEMA WRITE: non-generic errors = {real_error:?}");

        // ── 6. Assert commit succeeded ────────────────────────────────────────
        assert!(
            real_error.is_empty(),
            "transaction must commit after OVERWRITE migration — FieldCheck still firing: {real_error:?}\n\
             This means DEFINE FIELD OVERWRITE did NOT update the persisted type. \
             The stale 'option<record<symbol>>' definition is still enforced."
        );

        let count = count_chunks(&db).await.unwrap();
        println!("STALE-SCHEMA WRITE: chunk count after commit = {count}");
        assert_eq!(
            count,
            1,
            "chunk must persist after OVERWRITE migration (got {count}); \
             transaction is still rolling back due to stale field type"
        );
    }

    /// Verify that `DEFINE TABLE IF NOT EXISTS` does NOT drop existing rows.
    /// This confirms the table DDL form we chose is safe to re-run on a live database.
    #[tokio::test]
    async fn table_redefine_does_not_drop_rows() {
        let home = TempDir::new().unwrap();
        let db = open_raw_db(home.path(), "table_redef").await;

        // Set up a minimal chunk table and insert a sentinel row.
        db.query(
            "DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;\
             DEFINE FIELD OVERWRITE file ON chunk TYPE string;\
             DEFINE FIELD OVERWRITE line_start ON chunk TYPE int;\
             DEFINE FIELD OVERWRITE line_end ON chunk TYPE int;\
             DEFINE FIELD OVERWRITE content ON chunk TYPE string;\
             DEFINE FIELD OVERWRITE embedding ON chunk TYPE array<float>;\
             DEFINE FIELD OVERWRITE symbol_ref ON chunk TYPE option<string>;\
             CREATE chunk SET file='/sentinel', line_start=1, line_end=1, \
               content='sentinel', embedding=[], symbol_ref=NONE;",
        )
        .await
        .expect("setup")
        .check()
        .expect("setup check");

        let before = count_chunks(&db).await.unwrap();
        assert_eq!(before, 1, "sentinel row must exist before re-DDL");

        // Re-run the full SCHEMA_DDL (simulating a server restart).
        db.query(SCHEMA_DDL)
            .await
            .expect("re-apply DDL")
            .check()
            .expect("re-apply check");

        let after = count_chunks(&db).await.unwrap();
        println!("TABLE-REDEF: rows before={before}, after={after}");
        assert_eq!(
            after,
            before,
            "DEFINE TABLE IF NOT EXISTS must not drop existing rows (before={before}, after={after})"
        );
    }
}
