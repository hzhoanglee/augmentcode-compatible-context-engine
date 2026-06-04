use anyhow::{Context, Result};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tracing::info;

use crate::path_in_repo;

// ─── Public types ─────────────────────────────────────────────────────────

/// Identifies a chunk by its location in the source tree.
/// Used to map VectorIndex results back to SurrealDB rows.
#[derive(Debug, Clone)]
pub struct ChunkId {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// A single result returned by [`VectorIndex::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: ChunkId,
    /// Cosine similarity in [0, 1] (vectors are pre-normalized).
    pub score: f32,
}

// ─── VectorIndex ─────────────────────────────────────────────────────────

/// In-memory flat cosine-similarity index.
///
/// All vectors are L2-normalized at insert time, so cosine similarity reduces
/// to a plain dot product at query time (no division per candidate).
///
/// At 500 K chunks × 1024 dims, LLVM auto-vectorizes the inner loop to SIMD,
/// giving ~50-100 ms per query on a modern CPU — acceptable until an HNSW
/// implementation is available.
pub struct VectorIndex {
    /// Row-major storage: entry i holds the normalized embedding for chunk i.
    embeddings: Vec<Vec<f32>>,
    /// Parallel array: chunk_ids[i] corresponds to embeddings[i].
    chunk_ids: Vec<ChunkId>,
    /// Dimensionality of the first inserted vector; all subsequent inserts
    /// must match. `None` until the first insert.
    dimension: Option<usize>,
}

impl VectorIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            chunk_ids: Vec::new(),
            dimension: None,
        }
    }

    /// Insert a batch of (ChunkId, embedding) pairs.
    ///
    /// Each embedding is L2-normalized before storage. Zero-length or
    /// zero-magnitude vectors are stored as-is (they will score 0 against
    /// everything, which is correct).
    pub fn insert(&mut self, chunks: &[(ChunkId, Vec<f32>)]) {
        for (id, raw_emb) in chunks {
            if raw_emb.is_empty() {
                // Skip zero-length embeddings — they carry no information.
                continue;
            }
            // Record dimension on first insert; skip mismatches.
            match self.dimension {
                None => self.dimension = Some(raw_emb.len()),
                Some(d) if d != raw_emb.len() => {
                    tracing::warn!(
                        expected = d,
                        got = raw_emb.len(),
                        file = %id.file,
                        "embedding dimension mismatch — skipping chunk"
                    );
                    continue;
                }
                _ => {}
            }

            let normalized = l2_normalize(raw_emb);
            self.embeddings.push(normalized);
            self.chunk_ids.push(id.clone());
        }
    }

    /// Remove all embeddings whose `file` field matches `file`.
    ///
    /// Uses swap-remove to avoid O(n) shifts; rebuilds both parallel arrays.
    pub fn remove_file(&mut self, file: &str) {
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if self.chunk_ids[i].file == file {
                self.chunk_ids.swap_remove(i);
                self.embeddings.swap_remove(i);
                // Don't advance i — the swapped element now lives at i.
            } else {
                i += 1;
            }
        }
    }

    /// Remove all embeddings belonging to a repo.
    ///
    /// Uses [`path_in_repo`] for boundary-safe matching (no `/foo` vs `/foobar`
    /// collision). O(n) swap-remove pass over the parallel arrays — same pattern
    /// as [`remove_file`].
    pub fn remove_repo(&mut self, repo: &str) {
        let mut i = 0;
        while i < self.chunk_ids.len() {
            if path_in_repo(&self.chunk_ids[i].file, repo) {
                self.chunk_ids.swap_remove(i);
                self.embeddings.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Merge another `VectorIndex` into this one, consuming it.
    ///
    /// O(m) where m = other.len(). Vectors from `other` are already normalized
    /// (they went through `insert` or `load_from_db`), so no re-normalization
    /// is needed — they are moved directly into the parallel arrays.
    ///
    /// Dimension compatibility: if `self` has no dimension yet, adopts `other`'s.
    /// If both have a dimension and they differ, logs a warning and skips the merge.
    pub fn merge(&mut self, other: VectorIndex) {
        if other.is_empty() {
            return;
        }
        match (self.dimension, other.dimension) {
            (None, dim) => self.dimension = dim,
            (Some(a), Some(b)) if a != b => {
                tracing::warn!(
                    self_dim = a,
                    other_dim = b,
                    "VectorIndex merge dimension mismatch — skipping"
                );
                return;
            }
            _ => {}
        }
        self.embeddings.extend(other.embeddings);
        self.chunk_ids.extend(other.chunk_ids);
    }

    /// Search for the top-k most similar chunks to `query`.
    ///
    /// `query` is normalized internally so the caller need not pre-normalize.
    /// Returns results sorted by descending score, capped at `top_k`.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if self.embeddings.is_empty() || query.is_empty() || top_k == 0 {
            return vec![];
        }

        let q_norm = l2_normalize(query);

        // Score every vector.
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, dot_product(&q_norm, emb)))
            .collect();

        // Partial sort: bring the top-k largest scores to the front.
        let k = top_k.min(scored.len());
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        scored
            .into_iter()
            .map(|(i, score)| SearchResult {
                chunk_id: self.chunk_ids[i].clone(),
                score,
            })
            .collect()
    }

    /// Load all embeddings from SurrealDB on startup.
    ///
    /// Only loads rows that have a non-empty embedding vector.
    pub async fn load_from_db(db: &Surreal<Db>) -> Result<Self> {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct Row {
            file: String,
            line_start: i64,
            line_end: i64,
            embedding: Vec<f32>,
        }

        let rows: Vec<Row> = db
            .query("SELECT file, line_start, line_end, embedding FROM chunk WHERE embedding != []")
            .await
            .context("load embeddings from chunk table")?
            .take(0)?;

        let mut index = VectorIndex::new();
        let pairs: Vec<(ChunkId, Vec<f32>)> = rows
            .into_iter()
            .map(|r| {
                (
                    ChunkId {
                        file: r.file,
                        line_start: r.line_start as u32,
                        line_end: r.line_end as u32,
                    },
                    r.embedding,
                )
            })
            .collect();

        let count = pairs.len();
        index.insert(&pairs);
        info!(count, "loaded embeddings into VectorIndex");

        Ok(index)
    }

    /// Remove all entries from the index.
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.chunk_ids.clear();
        self.dimension = None;
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns `true` if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Return a copy of `v` normalized to unit L2 length.
/// Returns `v` unchanged if its magnitude is zero (avoids NaN).
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / mag).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a ChunkId for a given file and line range.
    fn chunk(file: &str, start: u32, end: u32) -> ChunkId {
        ChunkId {
            file: file.to_string(),
            line_start: start,
            line_end: end,
        }
    }

    /// Helper: create a simple non-zero embedding of the given dimension.
    fn emb(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| seed + i as f32 * 0.1).collect()
    }

    #[test]
    fn remove_repo_no_prefix_collision() {
        // Two repos where one path PREFIXES the other:
        // /foo and /foobar — removing /foo must NOT evict /foobar entries.
        let mut index = VectorIndex::new();

        let foo_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/foo/a.rs", 1, 10), emb(4, 1.0)),
            (chunk("/foo/b.rs", 1, 5), emb(4, 2.0)),
        ];
        let foobar_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/foobar/c.rs", 1, 20), emb(4, 3.0)),
            (chunk("/foobar/d.rs", 5, 15), emb(4, 4.0)),
        ];

        index.insert(&foo_chunks);
        index.insert(&foobar_chunks);
        assert_eq!(index.len(), 4);

        // Remove /foo — only /foo entries should be evicted.
        index.remove_repo("/foo");
        assert_eq!(index.len(), 2);

        // Verify the remaining entries all belong to /foobar.
        for cid in &index.chunk_ids {
            assert!(
                cid.file.starts_with("/foobar/"),
                "unexpected file after remove_repo: {}",
                cid.file
            );
        }
    }

    #[test]
    fn remove_repo_windows_paths_no_collision() {
        // Windows variant: D:\proj\foo vs D:\proj\foobar
        let mut index = VectorIndex::new();

        let foo_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk(r"D:\proj\foo\x.rs", 1, 10), emb(4, 1.0)),
        ];
        let foobar_chunks: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk(r"D:\proj\foobar\y.rs", 1, 10), emb(4, 2.0)),
        ];

        index.insert(&foo_chunks);
        index.insert(&foobar_chunks);
        assert_eq!(index.len(), 2);

        index.remove_repo(r"D:\proj\foo");
        assert_eq!(index.len(), 1);
        assert_eq!(index.chunk_ids[0].file, r"D:\proj\foobar\y.rs");
    }

    #[test]
    fn full_rebuild_one_repo_preserves_other() {
        // Simulate: index has repo A + repo B vectors.
        // Full rebuild of A: remove_repo(A) then insert(new A vectors).
        // Assert B is untouched and A is refreshed.
        let mut index = VectorIndex::new();

        let repo_a_old: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/repo_a/old1.rs", 1, 10), emb(4, 1.0)),
            (chunk("/repo_a/old2.rs", 5, 20), emb(4, 2.0)),
        ];
        let repo_b: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/repo_b/file1.rs", 1, 5), emb(4, 3.0)),
            (chunk("/repo_b/file2.rs", 10, 30), emb(4, 4.0)),
            (chunk("/repo_b/file3.rs", 1, 100), emb(4, 5.0)),
        ];

        index.insert(&repo_a_old);
        index.insert(&repo_b);
        assert_eq!(index.len(), 5);

        // Simulate full rebuild of repo A.
        index.remove_repo("/repo_a");
        assert_eq!(index.len(), 3); // Only B remains.

        let repo_a_new: Vec<(ChunkId, Vec<f32>)> = vec![
            (chunk("/repo_a/new1.rs", 1, 15), emb(4, 6.0)),
            (chunk("/repo_a/new2.rs", 1, 8), emb(4, 7.0)),
            (chunk("/repo_a/new3.rs", 1, 50), emb(4, 8.0)),
        ];
        index.insert(&repo_a_new);
        assert_eq!(index.len(), 6); // 3 B + 3 new A

        // Verify B is untouched.
        let b_files: Vec<&str> = index
            .chunk_ids
            .iter()
            .filter(|c| path_in_repo(&c.file, "/repo_b"))
            .map(|c| c.file.as_str())
            .collect();
        assert_eq!(b_files.len(), 3);
        assert!(b_files.contains(&"/repo_b/file1.rs"));
        assert!(b_files.contains(&"/repo_b/file2.rs"));
        assert!(b_files.contains(&"/repo_b/file3.rs"));

        // Verify A is refreshed.
        let a_files: Vec<&str> = index
            .chunk_ids
            .iter()
            .filter(|c| path_in_repo(&c.file, "/repo_a"))
            .map(|c| c.file.as_str())
            .collect();
        assert_eq!(a_files.len(), 3);
        assert!(a_files.contains(&"/repo_a/new1.rs"));
        assert!(a_files.contains(&"/repo_a/new2.rs"));
        assert!(a_files.contains(&"/repo_a/new3.rs"));
    }

    #[test]
    fn merge_combines_two_indexes() {
        let mut a = VectorIndex::new();
        a.insert(&[(chunk("/a/f.rs", 1, 10), emb(4, 1.0))]);

        let mut b = VectorIndex::new();
        b.insert(&[(chunk("/b/g.rs", 1, 5), emb(4, 2.0))]);

        a.merge(b);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn merge_dimension_mismatch_skips() {
        let mut a = VectorIndex::new();
        a.insert(&[(chunk("/a/f.rs", 1, 10), emb(4, 1.0))]);

        let mut b = VectorIndex::new();
        b.insert(&[(chunk("/b/g.rs", 1, 5), emb(8, 2.0))]);

        a.merge(b);
        // Merge skipped — a should still have only 1 entry.
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn remove_repo_with_trailing_sep() {
        let mut index = VectorIndex::new();
        index.insert(&[(chunk("/repo/file.rs", 1, 10), emb(4, 1.0))]);
        assert_eq!(index.len(), 1);

        // Repo path with trailing slash — should still match.
        index.remove_repo("/repo/");
        assert_eq!(index.len(), 0);
    }
}
