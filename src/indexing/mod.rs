pub mod pipeline;
pub mod tracker;
pub mod walker;
pub mod watcher;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};

use crate::config::Settings;
use crate::embedding::voyage::VoyageClient;
use crate::indexing::pipeline::IndexPipeline;
use crate::indexing::tracker::FileChange;
use crate::indexing::watcher::start_watcher;
use crate::store::{self, RepoDbMap};
use crate::vector::{SearchResult, VectorIndex};

// ─── Repo indexing status ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IndexState {
    Idle,
    Indexing,
    Error,
}

/// Per-repo status snapshot returned by `GET /api/index-status`.
///
/// Dual-meaning fields (state-gated contract):
/// - `state == Indexing`: `total_files` = workset size for this run (denominator);
///   `indexed_files` = files whose chunks have been embedded so far (numerator).
///   Progress % = indexed_files / total_files (guard against total_files == 0).
/// - `state == Idle` / `Error`: `indexed_files` = files in the index;
///   `total_files` = total files in the repo on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoStatus {
    pub state: IndexState,
    pub indexed_files: u64,
    pub total_files: u64,
    pub last_indexed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

impl Default for RepoStatus {
    fn default() -> Self {
        Self {
            state: IndexState::Idle,
            indexed_files: 0,
            total_files: 0,
            last_indexed_at: None,
            error: None,
        }
    }
}

// ─── ProgressHandle ───────────────────────────────────────────────────────

/// Lightweight handle passed into `IndexPipeline::run` so the pipeline can
/// write live progress into the shared status map without knowing the full engine.
#[derive(Clone)]
pub struct ProgressHandle {
    statuses: Arc<RwLock<HashMap<String, RepoStatus>>>,
    repo: String,
}

impl ProgressHandle {
    /// Set the denominator once the parsed file set is known.
    pub async fn set_run_total(&self, total: u64) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        s.total_files = total;
    }

    /// Advance the numerator. Monotonic — never decreases.
    pub async fn set_processed(&self, processed: u64) {
        let mut map = self.statuses.write().await;
        let s = map.entry(self.repo.clone()).or_default();
        if processed > s.indexed_files {
            s.indexed_files = processed;
        }
    }
}

// ─── IndexEngine ──────────────────────────────────────────────────────────

/// Central orchestrator for all indexing operations.
/// Stored in `AppState` and shared via `Arc`.
pub struct IndexEngine {
    pub home_dir: PathBuf,
    /// Per-repo status map, keyed by repo path string.
    /// Wrapped in Arc so `ProgressHandle` can hold a reference without borrowing self.
    pub statuses: Arc<RwLock<HashMap<String, RepoStatus>>>,
    /// Serialises concurrent pipeline runs per repo.
    repo_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Channel sender for triggering index runs (manual or watcher-driven).
    trigger_tx: tokio::sync::mpsc::Sender<IndexTrigger>,
    /// In-process vector index for fast cosine similarity search.
    pub vector_index: Arc<RwLock<VectorIndex>>,
    /// Shared per-repo DB handles — the same map the server reads through, so
    /// indexer writes are visible to explorer/query reads (one instance per repo).
    repo_dbs: RepoDbMap,
}

#[derive(Debug)]
pub struct IndexTrigger {
    pub repo: String,
    pub changes: Option<Vec<FileChange>>, // None = full incremental scan
    /// Force a full rebuild (clear + re-embed everything) regardless of staleness.
    pub rebuild: bool,
}

/// Load and merge the vector index from all configured repo DBs into one index.
///
/// Each repo is opened and loaded independently; a failure to open or load any
/// single repo is logged and skipped so the remaining repos still load. This is
/// the startup path that makes every indexed repo searchable (not just the first).
pub(crate) async fn load_repos_vector_index(
    repo_dbs: &RepoDbMap,
    home_dir: &std::path::Path,
    repos: &[String],
) -> VectorIndex {
    let mut merged = VectorIndex::new();
    for repo in repos {
        match store::get_or_open(repo_dbs, home_dir, repo).await {
            Ok(db) => match VectorIndex::load_from_db(&db).await {
                Ok(vi) => {
                    let count = vi.len();
                    if count > 0 {
                        merged.merge(vi);
                        info!(repo = %repo, count, "loaded repo vectors into VectorIndex");
                    }
                }
                Err(e) => {
                    warn!(repo = %repo, error = %e, "failed to load VectorIndex from DB; skipping repo");
                }
            },
            Err(e) => {
                warn!(repo = %repo, error = %e, "failed to open DB for VectorIndex load; skipping repo");
            }
        }
    }
    info!(total = merged.len(), "VectorIndex loaded from DB");
    merged
}

impl IndexEngine {
    /// Create the engine and spawn the watcher background task.
    ///
    /// `repo_dbs` is the shared handle map (also held by the server); the
    /// indexer writes through these same handles so reads observe its commits.
    pub async fn start(home_dir: PathBuf, settings: &Settings, repo_dbs: RepoDbMap) -> Arc<Self> {
        let (trigger_tx, trigger_rx) = tokio::sync::mpsc::channel::<IndexTrigger>(256);

        // Load the vector index from ALL configured repo DBs, merging into one index.
        let vector_index = Arc::new(RwLock::new(
            load_repos_vector_index(&repo_dbs, &home_dir, &settings.repos).await,
        ));

        let engine = Arc::new(IndexEngine {
            home_dir: home_dir.clone(),
            statuses: Arc::new(RwLock::new(HashMap::new())),
            repo_locks: Mutex::new(HashMap::new()),
            trigger_tx: trigger_tx.clone(),
            vector_index,
            repo_dbs,
        });

        // Initialise status entries.
        {
            let mut statuses = engine.statuses.write().await;
            for repo in &settings.repos {
                statuses.insert(repo.clone(), RepoStatus::default());
            }
        }

        // Start watcher for each repo.
        for repo in settings.repos.clone() {
            let tx = trigger_tx.clone();
            let repo_path = repo.clone();
            tokio::spawn(async move {
                start_watcher(repo_path, tx).await;
            });
        }

        // Spawn the single consumer task.
        let engine_clone = engine.clone();
        let settings_clone = settings.clone();
        tokio::spawn(async move {
            run_consumer(engine_clone, trigger_rx, settings_clone).await;
        });

        engine
    }

    /// Send a manual trigger to index a single repo.
    pub async fn trigger_index(&self, repo: &str) -> Result<()> {
        self.trigger_tx
            .send(IndexTrigger {
                repo: repo.to_string(),
                changes: None,
                rebuild: false,
            })
            .await
            .map_err(|e| anyhow::anyhow!("trigger channel closed: {e}"))?;
        Ok(())
    }

    /// Send a manual trigger to fully rebuild a single repo's index.
    pub async fn trigger_rebuild(&self, repo: &str) -> Result<()> {
        self.trigger_tx
            .send(IndexTrigger {
                repo: repo.to_string(),
                changes: None,
                rebuild: true,
            })
            .await
            .map_err(|e| anyhow::anyhow!("trigger channel closed: {e}"))?;
        Ok(())
    }

    /// Send a manual trigger to index all repos.
    pub async fn trigger_index_all(&self, repos: &[String]) -> Result<()> {
        for repo in repos {
            self.trigger_index(repo).await?;
        }
        Ok(())
    }

    /// Return per-repo status snapshot.
    pub async fn all_statuses(&self) -> HashMap<String, RepoStatus> {
        self.statuses.read().await.clone()
    }

    /// Return status for a single repo.
    pub async fn repo_status(&self, repo: &str) -> Option<RepoStatus> {
        self.statuses.read().await.get(repo).cloned()
    }

    async fn get_repo_lock(&self, repo: &str) -> Arc<Mutex<()>> {
        let mut locks = self.repo_locks.lock().await;
        locks
            .entry(repo.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Search the in-memory vector index for the top-k most similar chunks.
    ///
    /// This is a read-only, lock-free (read lock) operation. No DB call is
    /// made — all work happens in-process.
    pub async fn vector_search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<SearchResult> {
        let index = self.vector_index.read().await;
        index.search(query_embedding, top_k)
    }
}

// ─── Consumer task ────────────────────────────────────────────────────────

async fn run_consumer(
    engine: Arc<IndexEngine>,
    mut rx: tokio::sync::mpsc::Receiver<IndexTrigger>,
    settings: Settings,
) {
    while let Some(trigger) = rx.recv().await {
        let repo = trigger.repo.clone();
        let force_rebuild = trigger.rebuild;
        let engine_ref = engine.clone();
        let settings_ref = settings.clone();

        // Acquire per-repo serialisation lock.
        let lock = engine_ref.get_repo_lock(&repo).await;
        let _guard = lock.lock().await;

        // Mark indexing. Reset progress counters so the UI shows the
        // indeterminate pulse (total_files == 0) until the pipeline reports a total.
        {
            let mut statuses = engine_ref.statuses.write().await;
            let status = statuses.entry(repo.clone()).or_default();
            status.state = IndexState::Indexing;
            status.error = None;
            status.indexed_files = 0;
            status.total_files = 0;
        }

        // Build embedding client — skip if no keys configured.
        let voyage_client = if settings_ref.embedding.api_keys.is_empty() {
            info!(repo = %repo, "no embedding API keys configured, skipping embed");
            None
        } else {
            match VoyageClient::new(
                settings_ref.embedding.model.clone(),
                settings_ref.embedding.api_keys.clone(),
            ) {
                Ok(c) => Some(c),
                Err(e) => {
                    error!(repo = %repo, error = %e, "failed to create voyage client");
                    let mut statuses = engine_ref.statuses.write().await;
                    let s = statuses.entry(repo.clone()).or_default();
                    s.state = IndexState::Error;
                    s.error = Some(e.to_string());
                    continue;
                }
            }
        };

        // Build a progress handle that lets the pipeline write live counts.
        let progress = ProgressHandle {
            statuses: Arc::clone(&engine_ref.statuses),
            repo: repo.clone(),
        };

        // Acquire the SHARED repo DB handle — the same instance the server reads
        // through, so the explorer/query layer sees these writes immediately.
        let db = match store::get_or_open(&engine_ref.repo_dbs, &engine_ref.home_dir, &repo).await {
            Ok(db) => db,
            Err(e) => {
                error!(repo = %repo, error = %e, "failed to open repo DB");
                let mut statuses = engine_ref.statuses.write().await;
                let s = statuses.entry(repo.clone()).or_default();
                s.state = IndexState::Error;
                s.error = Some(e.to_string());
                continue;
            }
        };

        let pipeline = IndexPipeline::new(repo.clone(), voyage_client);

        match pipeline
            .run(&db, trigger.changes, force_rebuild, Some(&engine_ref.vector_index), Some(progress))
            .await
        {
            Ok(stats) => {
                info!(repo = %repo, indexed = stats.indexed_files, "indexing complete");
                let mut statuses = engine_ref.statuses.write().await;
                let s = statuses.entry(repo.clone()).or_default();
                s.state = IndexState::Idle;
                s.indexed_files = stats.indexed_files;
                s.total_files = stats.total_files;
                s.last_indexed_at = Some(Utc::now());
                s.error = None;
            }
            Err(e) => {
                // `{e:#}` prints the full anyhow context chain on one line
                // (outer context + the underlying SurrealDB per-statement error),
                // not just the outermost message — essential for diagnosing rollbacks.
                error!(repo = %repo, error = %format!("{e:#}"), "indexing failed");
                let mut statuses = engine_ref.statuses.write().await;
                let s = statuses.entry(repo.clone()).or_default();
                s.state = IndexState::Error;
                s.error = Some(format!("{e:#}"));
            }
        }
    }
}

#[cfg(test)]
mod load_repos_tests {
    use super::*;
    use tempfile::TempDir;

    /// Seed `n` chunk rows (each with a non-empty 4-d embedding) into `repo`'s DB.
    async fn seed_repo(home: &std::path::Path, repo: &str, n: usize) {
        let db = store::open_db(home, repo).await.expect("open_db");
        for i in 0..n {
            let q = format!(
                "CREATE chunk SET file = '{repo}/f{i}.rs', line_start = 1, line_end = 2, \
                 content = 'x', embedding = [0.1, 0.2, 0.3, 0.4], symbol_ref = NONE;"
            );
            db.query(&q).await.expect("seed chunk");
        }
    }

    /// Startup must load EVERY configured repo, not just the first.
    /// Two repos seeded with 1 and 2 chunks → merged index must hold all 3.
    /// Fails under the `.first()` / `take(1)` regression (would yield 1).
    #[tokio::test]
    async fn loads_all_repos_not_just_first() {
        let home = TempDir::new().expect("tempdir");
        let repo_one = "/proj/repo_one".to_string();
        let repo_two = "/proj/repo_two".to_string();

        seed_repo(home.path(), &repo_one, 1).await;
        seed_repo(home.path(), &repo_two, 2).await;

        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        let index =
            load_repos_vector_index(&repo_dbs, home.path(), &[repo_one, repo_two]).await;

        // 1 (repo_one) + 2 (repo_two) = 3. Under take(1) this would be 1.
        assert_eq!(index.len(), 3, "expected all repos merged, not just the first");
    }
}
