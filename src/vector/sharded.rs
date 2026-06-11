//! Per-repo sharded vector index with a resident-byte cap and repo-keyed LRU.
//!
//! Memory axis 2 fix: instead of one merged flat `Vec<f32>` holding every repo's
//! embeddings (resident for the lifetime of the process), the index is split into
//! one [`VectorIndex`] shard per repo. A resident-byte cap bounds total RAM; when
//! an insert/warm would exceed it, the least-recently-used **non-active** shards
//! are evicted (their `VectorIndex` dropped wholesale — no O(n) swap-remove scan).
//!
//! ## Concurrency contract (held by the engine)
//! - The whole `ShardedVectorIndex` lives behind a single `RwLock`. `search` takes
//!   a READ guard (concurrent searches run in parallel); mutation (`install_shard`,
//!   `apply_incremental`, `evict_*`, `remove_*`) takes a WRITE guard.
//! - `search` is `&self`: it bumps per-shard recency via an `AtomicU64` last-touched
//!   stamp (interior mutability), so updating LRU order does NOT require a write lock
//!   and never serializes the read-heavy hot path.
//! - Lock order is ALWAYS `repo_dbs` → `vector_index`, never nested the other way.
//! - Warm/eviction loads into a temp `VectorIndex` OUTSIDE any write lock, then
//!   installs the finished shard under a short write lock — the DB scan never
//!   happens while the write lock is held.
//!
//! ## Ranking correctness across shards
//! Every stored vector and every query vector is L2-normalized (see
//! [`VectorIndex::insert`] / [`VectorIndex::search`]), so each shard returns true
//! cosine scores in `[-1, 1]`. A global top-k over per-shard top-k candidates is
//! therefore exact — partitioning is mathematically transparent to ranking.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vector::{SearchResult, VectorIndex};

/// A resident repo shard plus its LRU recency stamp.
///
/// `last_touched` is an `AtomicU64` so `search` (which holds only `&self`) can bump
/// it without a write lock. Recency order is derived by comparing stamps against a
/// monotonic global clock — no `Vec` reshuffling, no `&mut self` for a touch.
struct Shard {
    index: VectorIndex,
    last_touched: AtomicU64,
}

impl Shard {
    fn new(index: VectorIndex, stamp: u64) -> Self {
        Self {
            index,
            last_touched: AtomicU64::new(stamp),
        }
    }
}

/// Outcome of a `search` over the sharded index.
///
/// `cold_repos` lists repos that were requested (in scope for the query) but are
/// not currently resident. The caller treats results as PARTIAL and should enqueue
/// a background warm for each cold repo — it must never block the query to load them.
#[derive(Debug, Default)]
pub struct ShardedSearch {
    pub results: Vec<SearchResult>,
    pub cold_repos: Vec<String>,
}

/// Per-repo sharded vector index. Not internally synchronized — the engine wraps
/// the whole struct in one `tokio::sync::RwLock` (see module docs). `search` only
/// needs a read guard; recency is tracked via per-shard atomic stamps.
pub struct ShardedVectorIndex {
    /// One vector shard (+ recency stamp) per repo, keyed by repo path string.
    shards: HashMap<String, Shard>,
    /// Monotonic logical clock. Each touch fetch-adds 1 and stamps the shard, so a
    /// higher stamp == more recently used. Eviction picks the lowest stamp (LRU).
    clock: AtomicU64,
    /// Resident-byte cap. Total `byte_size()` across shards is kept at or below
    /// this after each install (best-effort: a single shard larger than the cap
    /// is still kept — we never evict a repo to below usefulness, and never evict
    /// the repo just installed).
    cap_bytes: usize,
}

impl ShardedVectorIndex {
    /// Create an empty sharded index with the given resident-byte cap.
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            shards: HashMap::new(),
            clock: AtomicU64::new(0),
            cap_bytes,
        }
    }

    /// True if no shard holds any vector.
    pub fn is_empty(&self) -> bool {
        self.shards.values().all(|s| s.index.is_empty())
    }

    /// Total resident vector bytes across all shards.
    pub fn resident_bytes(&self) -> usize {
        self.shards.values().map(|s| s.index.byte_size()).sum()
    }

    /// Number of resident (loaded) repo shards.
    pub fn resident_repo_count(&self) -> usize {
        self.shards.len()
    }

    /// The configured resident-byte cap (0 = disabled).
    #[inline]
    pub fn resident_cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    /// List of currently-resident repo keys.
    pub fn resident_repos(&self) -> Vec<String> {
        self.shards.keys().cloned().collect()
    }

    /// True if `repo` currently has a resident shard.
    pub fn is_resident(&self, repo: &str) -> bool {
        self.shards.contains_key(repo)
    }

    // ── Recency bookkeeping ──────────────────────────────────────────────────

    /// Next monotonic recency stamp.
    #[inline]
    fn next_stamp(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Bump `repo`'s recency to most-recently-used. Takes only `&self` — the stamp
    /// is an atomic, so this is callable from the read-locked search path without
    /// serializing concurrent searches.
    fn touch(&self, repo: &str) {
        if let Some(shard) = self.shards.get(repo) {
            shard
                .last_touched
                .store(self.next_stamp(), Ordering::Relaxed);
        }
    }

    // ── Installation + eviction ──────────────────────────────────────────────

    /// Install (replace) the shard for `repo` with a fully-built `VectorIndex`,
    /// touch its recency, then evict LRU non-active shards until under cap.
    ///
    /// `active` is the set of repos that must never be evicted by this call
    /// (the just-installed repo is always implicitly protected).
    pub fn install_shard(&mut self, repo: &str, shard: VectorIndex, active: &[String]) {
        if shard.is_empty() {
            // An empty shard carries no information; drop it.
            self.shards.remove(repo);
            return;
        }
        let stamp = self.next_stamp();
        self.shards
            .insert(repo.to_string(), Shard::new(shard, stamp));
        self.evict_to_cap(repo, active);
    }

    /// Evict least-recently-used shards until resident bytes are at or below the
    /// cap. Never evicts `protected` (the just-installed repo) or any repo in
    /// `active`. Correctness against in-flight reads is guaranteed by the outer
    /// `RwLock` (this runs under a write guard; searches hold a read guard and
    /// return owned results, so a shard is never dropped mid-read).
    fn evict_to_cap(&mut self, protected: &str, active: &[String]) {
        if self.cap_bytes == 0 {
            return; // cap disabled
        }
        while self.resident_bytes() > self.cap_bytes {
            // Find the evictable shard with the lowest recency stamp (true LRU).
            let victim = self
                .shards
                .iter()
                .filter(|(repo, _)| {
                    repo.as_str() != protected && !active.iter().any(|a| a == *repo)
                })
                .min_by_key(|(_, s)| s.last_touched.load(Ordering::Relaxed))
                .map(|(repo, _)| repo.clone());
            match victim {
                Some(repo) => {
                    self.shards.remove(&repo);
                }
                None => break, // nothing left that may be evicted
            }
        }
    }

    /// Explicitly evict a single repo's shard (e.g. when its DB handle is evicted
    /// by the synchronized repo_dbs LRU). Idempotent.
    pub fn evict_repo(&mut self, repo: &str) {
        self.shards.remove(repo);
    }

    // ── Incremental write helpers (used by the pipeline) ─────────────────────

    /// Apply incremental changes to a repo's shard: remove vectors for
    /// `removed_files`, then insert `new_vectors`. Touches recency and re-checks
    /// the cap. If the repo has no resident shard yet, one is created.
    pub fn apply_incremental(
        &mut self,
        repo: &str,
        removed_files: &[String],
        new_vectors: &[(crate::vector::ChunkId, Vec<f32>)],
        active: &[String],
    ) {
        let stamp = self.next_stamp();
        let shard = self
            .shards
            .entry(repo.to_string())
            .or_insert_with(|| Shard::new(VectorIndex::new(), stamp));
        for file in removed_files {
            shard.index.remove_file(file);
        }
        shard.index.insert(new_vectors);
        shard.last_touched.store(stamp, Ordering::Relaxed);
        // An emptied shard is dropped to free its (now-zero) slot.
        if shard.index.is_empty() {
            self.evict_repo(repo);
            return;
        }
        self.evict_to_cap(repo, active);
    }

    /// Replace a repo's shard contents wholesale (full-rebuild path): clear the
    /// existing shard, insert the freshly-built vectors. Touches recency + cap.
    pub fn replace_repo(
        &mut self,
        repo: &str,
        new_vectors: &[(crate::vector::ChunkId, Vec<f32>)],
        active: &[String],
    ) {
        let mut shard = VectorIndex::new();
        shard.insert(new_vectors);
        self.install_shard(repo, shard, active);
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Fan-out search over resident shards, merge to a global top-k.
    ///
    /// Takes `&self` — runs under a READ guard, so concurrent searches proceed in
    /// parallel. Recency is bumped via per-shard atomic stamps (no write lock).
    ///
    /// - `repo_filter = Some(repo)`: search only that repo's shard. If it is not
    ///   resident, returns empty results with the repo flagged cold.
    /// - `repo_filter = None`: search every resident shard; `scope` lists the
    ///   repos that SHOULD be searched (the configured set) so non-resident ones
    ///   can be reported cold for background warming.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        repo_filter: Option<&str>,
        scope: &[String],
    ) -> ShardedSearch {
        let mut cold_repos: Vec<String> = Vec::new();
        let mut merged: Vec<SearchResult> = Vec::new();

        match repo_filter {
            Some(repo) => {
                if let Some(shard) = self.shards.get(repo) {
                    merged.extend(shard.index.search(query, top_k));
                    self.touch(repo);
                } else {
                    cold_repos.push(repo.to_string());
                }
            }
            None => {
                // Search every resident shard.
                for (repo, shard) in self.shards.iter() {
                    if shard.index.is_empty() {
                        continue;
                    }
                    merged.extend(shard.index.search(query, top_k));
                    self.touch(repo);
                }
                // Any in-scope repo that is not resident is cold → warm in background.
                for repo in scope {
                    if !self.shards.contains_key(repo) {
                        cold_repos.push(repo.clone());
                    }
                }
            }
        }

        // Global top-k over the per-shard candidates. Scores are comparable across
        // shards (all L2-normalized cosine), so a plain sort + truncate is exact.
        merged.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(top_k);

        ShardedSearch {
            results: merged,
            cold_repos,
        }
    }

    /// Remove all vectors belonging to `repo` across shards (repo deletion path).
    /// Boundary-safe: also drops any shard whose key path is inside `repo`.
    pub fn remove_repo(&mut self, repo: &str) {
        self.shards.remove(repo);
        // Defensive: also clear vectors physically inside the repo from any shard
        // (covers nested-path edge cases consistent with the old merged-index
        // `remove_repo` semantics at vector/mod.rs).
        for shard in self.shards.values_mut() {
            shard.index.remove_repo(repo);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{ChunkId, VectorIndex};

    /// Build a vector whose cosine against the query `[1,0,0,...]` is STRICTLY
    /// determined by `gid` (globally unique id): v = normalize([gid+1, 1, 0...]).
    /// cosine = (gid+1)/sqrt((gid+1)^2 + 1), strictly increasing in gid → every
    /// vector gets a distinct score, so top-k ordering is unambiguous (no ties).
    fn vec_for_gid(gid: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 8];
        v[0] = (gid + 1) as f32;
        v[1] = 1.0;
        v
    }

    /// Build a shard for `repo` from an explicit list of global ids.
    fn shard_from_gids(repo: &str, gids: &[usize]) -> VectorIndex {
        let mut vi = VectorIndex::new();
        let pairs: Vec<(ChunkId, Vec<f32>)> = gids
            .iter()
            .map(|&g| {
                (
                    ChunkId {
                        file: format!("{repo}/g{g}.rs"),
                        line_start: 1,
                        line_end: 2,
                    },
                    vec_for_gid(g),
                )
            })
            .collect();
        vi.insert(&pairs);
        vi
    }

    fn bytes_for(n_vectors: usize, dim: usize) -> usize {
        n_vectors * dim * std::mem::size_of::<f32>()
    }

    /// CONTRACT TEST 1 — cross-shard top-k == merged-index top-k.
    ///
    /// Builds the same vector population two ways: (a) one merged `VectorIndex`
    /// (the OLD design), (b) three per-repo shards. A global top-k over the shards
    /// must return EXACTLY the same chunk_ids in the same order as the merged
    /// index. Proves sharding is mathematically transparent to ranking (all
    /// vectors L2-normalized → cosine scores comparable across shards). Vectors are
    /// constructed with strictly distinct scores so ordering is unambiguous.
    #[test]
    fn cross_shard_topk_equals_merged_topk() {
        // Distinct global ids partitioned across three repos (interleaved sizes).
        let repo_gids: [(&str, Vec<usize>); 3] = [
            ("/r/a", vec![0, 3, 6, 9, 12]),
            ("/r/b", vec![1, 4, 7, 10, 13, 15, 16]),
            ("/r/c", vec![2, 5, 8, 11]),
        ];

        // (a) Merged index.
        let mut merged = VectorIndex::new();
        for (repo, gids) in &repo_gids {
            merged.merge(shard_from_gids(repo, gids));
        }

        // (b) Sharded index, generous cap (nothing evicted).
        let mut sharded = ShardedVectorIndex::new(bytes_for(1000, 8));
        let scope: Vec<String> = repo_gids.iter().map(|(r, _)| r.to_string()).collect();
        for (repo, gids) in &repo_gids {
            sharded.install_shard(repo, shard_from_gids(repo, gids), &scope);
        }

        // Query points at slot 0 — score strictly increases with gid.
        let mut query = vec![0.0f32; 8];
        query[0] = 1.0;
        let top_k = 6;

        let merged_results = merged.search(&query, top_k);
        let sharded_out = sharded.search(&query, top_k, None, &scope);

        assert_eq!(
            merged_results.len(),
            sharded_out.results.len(),
            "merged and sharded must return the same number of results"
        );
        for (m, s) in merged_results.iter().zip(sharded_out.results.iter()) {
            assert_eq!(
                m.chunk_id.file, s.chunk_id.file,
                "ranking order must match merged index"
            );
            assert_eq!(m.chunk_id.line_start, s.chunk_id.line_start);
            assert!(
                (m.score - s.score).abs() < 1e-6,
                "scores must match across designs: merged={} sharded={}",
                m.score,
                s.score
            );
        }
        assert!(sharded_out.cold_repos.is_empty(), "no repo should be cold");
    }

    /// CONTRACT TEST 2 — eviction respects the cap AND never evicts an active repo.
    ///
    /// Cap fits exactly two 10-vector shards. Install three repos; the LRU must
    /// evict down to the cap. We pass repo "/r/a" as ACTIVE on every install, so
    /// even though it is the least-recently-used by insertion order, it must NEVER
    /// be evicted — proving the active-set protection. Total resident bytes must
    /// end at or below the cap.
    #[test]
    fn eviction_respects_cap_and_never_evicts_active() {
        let dim = 8;
        let per_shard = 10usize;
        // Cap = room for exactly 2 shards.
        let cap = bytes_for(per_shard * 2, dim);
        let mut idx = ShardedVectorIndex::new(cap);

        let active = vec!["/r/a".to_string()];

        // Install a, b, c — each 10 vectors. After c, resident would be 3 shards
        // (> cap) → must evict one. "/r/a" is active so it is protected; the LRU
        // victim must be "/r/b" (oldest non-active).
        idx.install_shard(
            "/r/a",
            shard_from_gids("/r/a", &(0..10).collect::<Vec<_>>()),
            &active,
        );
        idx.install_shard(
            "/r/b",
            shard_from_gids("/r/b", &(10..20).collect::<Vec<_>>()),
            &active,
        );
        idx.install_shard(
            "/r/c",
            shard_from_gids("/r/c", &(20..30).collect::<Vec<_>>()),
            &active,
        );

        assert!(
            idx.resident_bytes() <= cap,
            "resident bytes ({}) must be at or below cap ({})",
            idx.resident_bytes(),
            cap
        );
        assert!(
            idx.is_resident("/r/a"),
            "active repo /r/a must never be evicted"
        );
        assert!(
            idx.is_resident("/r/c"),
            "most-recently-installed /r/c must stay resident"
        );
        assert!(
            !idx.is_resident("/r/b"),
            "LRU non-active /r/b must be evicted to honor the cap"
        );
        assert_eq!(
            idx.resident_repo_count(),
            2,
            "exactly two shards fit under the cap"
        );
    }

    /// CONTRACT TEST 3 — a cold repo yields PARTIAL results and is flagged for warm.
    ///
    /// Two repos are in scope but only one is resident. A `None`-filter search must:
    /// (a) return results only from the resident shard (partial),
    /// (b) report the non-resident repo in `cold_repos` so the engine can spawn a
    ///     background warm. It must NOT block or error on the cold repo.
    /// A subsequent `install_shard` (simulating the completed warm) then makes the
    /// previously-cold repo searchable with no repo reported cold.
    #[test]
    fn cold_repo_returns_partial_and_flags_warm() {
        let mut idx = ShardedVectorIndex::new(bytes_for(1000, 8));
        let scope = vec!["/r/hot".to_string(), "/r/cold".to_string()];

        // Only "/r/hot" is resident.
        idx.install_shard("/r/hot", shard_from_gids("/r/hot", &[0, 1, 2]), &scope);

        let mut query = vec![0.0f32; 8];
        query[0] = 1.0;

        let out = idx.search(&query, 10, None, &scope);

        // Partial: results only from the hot shard.
        assert!(
            !out.results.is_empty(),
            "resident shard must contribute results"
        );
        assert!(
            out.results
                .iter()
                .all(|r| r.chunk_id.file.starts_with("/r/hot")),
            "results must come only from the resident shard"
        );
        // Cold repo flagged for background warm — not silently dropped, not blocking.
        assert_eq!(
            out.cold_repos,
            vec!["/r/cold".to_string()],
            "cold repo must be flagged for warm"
        );

        // Simulate the background warm completing.
        idx.install_shard("/r/cold", shard_from_gids("/r/cold", &[3, 4]), &scope);
        let out2 = idx.search(&query, 10, None, &scope);
        assert!(
            out2.cold_repos.is_empty(),
            "after warm, no repo should be cold"
        );
        assert!(
            out2.results
                .iter()
                .any(|r| r.chunk_id.file.starts_with("/r/cold")),
            "warmed repo must now contribute results"
        );
    }

    /// A repo-filtered search to a cold repo returns empty + flags only that repo.
    #[test]
    fn filtered_search_to_cold_repo_flags_only_that_repo() {
        let mut idx = ShardedVectorIndex::new(bytes_for(1000, 8));
        let scope = vec!["/r/hot".to_string(), "/r/cold".to_string()];
        idx.install_shard("/r/hot", shard_from_gids("/r/hot", &[0, 1, 2]), &scope);

        let mut query = vec![0.0f32; 8];
        query[0] = 1.0;

        // Filter to the cold repo — must return empty and flag it.
        let out = idx.search(&query, 10, Some("/r/cold"), &["/r/cold".to_string()]);
        assert!(
            out.results.is_empty(),
            "filtered search to cold repo yields no results"
        );
        assert_eq!(out.cold_repos, vec!["/r/cold".to_string()]);
    }

    /// CONTRACT TEST 4 — `search` runs on `&self` and a search TOUCH updates LRU
    /// recency (via the atomic stamp), so eviction picks the genuinely least-
    /// recently-USED shard, not merely the oldest-installed one.
    ///
    /// Install a, b in that order (b newer). Then search `a` via `&self` — this
    /// must bump a's recency above b WITHOUT a write lock. Installing c then
    /// overflows the 2-shard cap: the victim must be `b` (now the LRU), proving the
    /// search touch took effect through the shared-reference path.
    #[test]
    fn search_touch_updates_recency_on_shared_ref() {
        let dim = 8;
        let per_shard = 10usize;
        let cap = bytes_for(per_shard * 2, dim);
        let mut idx = ShardedVectorIndex::new(cap);

        idx.install_shard(
            "/r/a",
            shard_from_gids("/r/a", &(0..10).collect::<Vec<_>>()),
            &[],
        );
        idx.install_shard(
            "/r/b",
            shard_from_gids("/r/b", &(10..20).collect::<Vec<_>>()),
            &[],
        );

        // Search `a` through a SHARED reference (&self) — bumps a to MRU via the
        // atomic stamp. If this compiles with `&idx` (not `&mut`), the hot path is
        // genuinely on a read-capable signature.
        let shared: &ShardedVectorIndex = &idx;
        let mut query = vec![0.0f32; 8];
        query[0] = 1.0;
        let _ = shared.search(&query, 5, Some("/r/a"), &["/r/a".to_string()]);

        // Now install c → overflow. Victim must be b (LRU after a's touch).
        idx.install_shard(
            "/r/c",
            shard_from_gids("/r/c", &(20..30).collect::<Vec<_>>()),
            &[],
        );

        assert!(idx.resident_bytes() <= cap, "must honor cap");
        assert!(
            idx.is_resident("/r/a"),
            "a was just searched → MRU → must survive"
        );
        assert!(idx.is_resident("/r/c"), "c just installed → must survive");
        assert!(
            !idx.is_resident("/r/b"),
            "b became LRU after a's search-touch → must be the eviction victim"
        );
    }
}
