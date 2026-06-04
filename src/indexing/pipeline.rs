use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use tracing::{debug, info, warn};

use crate::embedding::InputType;
use crate::embedding::voyage::{MAX_BATCH_SIZE, VoyageClient};
use crate::indexing::ProgressHandle;
use crate::indexing::tracker::{ChangeKind, FileChange, stat_file};
use crate::indexing::walker::walk_repo;
use crate::parsing::parse_file;
use crate::parsing::relations::{EdgeKind, EdgeTarget, RawEdge};
use crate::parsing::symbols::{QualifiedSymbol, Symbol};
use crate::store;
use crate::store::ops::get_all_file_meta;
use crate::vector::{ChunkId, VectorIndex};

pub struct IndexPipelineStats {
    pub indexed_files: u64,
    pub total_files: u64,
}

/// Runs the parse → embed → store pipeline for one repo.
pub struct IndexPipeline {
    home_dir: PathBuf,
    repo: String,
    voyage: Option<VoyageClient>,
}

impl IndexPipeline {
    pub fn new(home_dir: PathBuf, repo: String, voyage: Option<VoyageClient>) -> Self {
        Self { home_dir, repo, voyage }
    }

    /// Run the pipeline.
    /// - `changes = None` → incremental scan (detect changes from mtime).
    /// - `changes = Some(list)` → process only the given file changes.
    /// - `force_rebuild = true` → clear and re-embed everything, ignoring staleness.
    /// - `progress` → optional handle for reporting live progress to the status map.
    pub async fn run(
        &self,
        changes: Option<Vec<FileChange>>,
        force_rebuild: bool,
        vector_index: Option<&tokio::sync::RwLock<VectorIndex>>,
        progress: Option<ProgressHandle>,
    ) -> Result<IndexPipelineStats> {
        let db = store::open_db(&self.home_dir, &self.repo).await?;

        // Check if first run (no file_meta at all).
        let stored_meta = get_all_file_meta(&db, &self.repo).await?;
        let is_first_run = stored_meta.is_empty();

        let total_files = walk_repo(&self.repo).len() as u64;

        if is_first_run || force_rebuild {
            if force_rebuild && !is_first_run {
                info!(repo = %self.repo, "forced full rebuild");
            } else {
                info!(repo = %self.repo, "first run — full rebuild");
            }
            let new_vectors = self.full_rebuild(&db, progress.as_ref()).await?;
            if let Some(vi) = vector_index {
                let mut guard = vi.write().await;
                guard.remove_repo(&self.repo);
                guard.insert(&new_vectors);
            }
            let indexed = get_all_file_meta(&db, &self.repo).await?.len() as u64;
            return Ok(IndexPipelineStats { indexed_files: indexed, total_files });
        }

        // Incremental run.
        let file_changes = match changes {
            Some(explicit) => explicit,
            None => {
                // Detect via mtime comparison.
                let all_files = walk_repo(&self.repo);
                let meta_map: HashMap<String, (i64, i64)> = stored_meta
                    .iter()
                    .map(|m| (m.path.clone(), (m.mtime, m.size)))
                    .collect();
                crate::indexing::tracker::detect_changes(&all_files, &meta_map)
            }
        };

        if file_changes.is_empty() {
            debug!(repo = %self.repo, "no changes detected");
            let indexed = stored_meta.len() as u64;
            return Ok(IndexPipelineStats { indexed_files: indexed, total_files });
        }

        info!(repo = %self.repo, changes = file_changes.len(), "incremental index");
        let (removed_files, new_vectors) = self.incremental_run(&db, file_changes, progress.as_ref()).await?;

        if let Some(vi) = vector_index {
            let mut guard = vi.write().await;
            for file in &removed_files {
                guard.remove_file(file);
            }
            guard.insert(&new_vectors);
        }

        let indexed = get_all_file_meta(&db, &self.repo).await?.len() as u64;
        Ok(IndexPipelineStats { indexed_files: indexed, total_files })
    }

    // ─── Full rebuild ─────────────────────────────────────────────────────

    /// Returns (chunk_id, embedding) pairs for VectorIndex insertion.
    async fn full_rebuild(&self, db: &Surreal<Db>, progress: Option<&ProgressHandle>) -> Result<Vec<(ChunkId, Vec<f32>)>> {
        // 1. Walk all files.
        let all_files = walk_repo(&self.repo);
        info!(repo = %self.repo, file_count = all_files.len(), "walking repo for full rebuild");

        // 2. Parse all files.
        let parse_results = parse_all_files_parallel(&all_files);

        // 3. Collect symbols, chunks, edges.
        let mut all_symbols: Vec<Symbol> = Vec::new();
        let mut all_chunks_by_file: Vec<(String, Vec<crate::parsing::chunker::Chunk>)> = Vec::new();
        let mut all_edges: Vec<RawEdge> = Vec::new();

        for (file, pr) in &parse_results {
            all_symbols.extend(pr.symbols.iter().cloned());
            all_chunks_by_file.push((file.clone(), pr.chunks.clone()));
            all_edges.extend(pr.edges.iter().cloned());
        }

        // 4. Build symbol index for cross-file resolution.
        let symbol_index = build_symbol_index(&all_symbols);

        // 5. Embed all chunks (outside transaction — network I/O).
        let embeddings = self.embed_all_chunks(&all_chunks_by_file, progress).await?;

        // 6. Resolve edges.
        let resolved_edges = resolve_edges(&all_edges, &symbol_index);

        // 7. Compute file stats before the transaction.
        let file_stats: Vec<(String, i64, i64)> = all_files
            .iter()
            .filter_map(|f| stat_file(f).map(|s| (f.clone(), s.mtime, s.size)))
            .collect();

        // 8. Build and execute a single transaction query.
        let mut txn = String::from("BEGIN TRANSACTION;\n");

        // Delete everything.
        txn.push_str("DELETE FROM calls;\n");
        txn.push_str("DELETE FROM uses;\n");
        txn.push_str("DELETE FROM imports;\n");
        txn.push_str("DELETE FROM contains;\n");
        txn.push_str("DELETE FROM implements;\n");
        txn.push_str("DELETE FROM symbol;\n");
        txn.push_str("DELETE FROM chunk;\n");
        txn.push_str("DELETE FROM file_meta;\n");

        // Upsert symbols.
        for sym in &all_symbols {
            append_upsert_symbol(&mut txn, sym);
        }

        // Insert edges.
        for (from, to, kind, line) in &resolved_edges {
            append_insert_edge(&mut txn, from, to, kind, *line);
        }

        // Insert chunks with embeddings.
        let mut emb_iter = embeddings.iter();
        let mut chunk_vectors: Vec<(ChunkId, Vec<f32>)> = Vec::new();

        for (_file, chunks) in &all_chunks_by_file {
            for chunk in chunks {
                let emb = emb_iter.next().cloned().unwrap_or_default();
                append_insert_chunk(&mut txn, chunk, &emb);
                chunk_vectors.push((
                    ChunkId {
                        file: chunk.file.clone(),
                        line_start: chunk.line_start,
                        line_end: chunk.line_end,
                    },
                    emb,
                ));
            }
        }

        // Upsert file_meta.
        for (path, mtime, size) in &file_stats {
            append_upsert_file_meta(&mut txn, path, *mtime, *size, &self.repo);
        }

        txn.push_str("COMMIT TRANSACTION;\n");

        db.query(&txn).await.context("full_rebuild: transaction failed")?;

        Ok(chunk_vectors)
    }

    // ─── Incremental run ──────────────────────────────────────────────────

    /// Returns (files_removed, new_chunk_vectors) for VectorIndex update.
    async fn incremental_run(
        &self,
        db: &Surreal<Db>,
        changes: Vec<FileChange>,
        progress: Option<&ProgressHandle>,
    ) -> Result<(Vec<String>, Vec<(ChunkId, Vec<f32>)>)> {
        // Separate added/modified from deleted.
        let to_process: Vec<String> = changes
            .iter()
            .filter(|c| c.kind != ChangeKind::Deleted)
            .map(|c| c.path.clone())
            .collect();
        let to_delete: Vec<String> = changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Deleted)
            .map(|c| c.path.clone())
            .collect();

        // All files whose old data must be purged.
        let all_affected: Vec<String> = to_delete
            .iter()
            .chain(to_process.iter())
            .cloned()
            .collect();

        // Parse changed files.
        let parse_results = parse_all_files_parallel(&to_process);

        let mut all_symbols: Vec<Symbol> = Vec::new();
        let mut all_chunks_by_file: Vec<(String, Vec<crate::parsing::chunker::Chunk>)> = Vec::new();
        let mut all_edges: Vec<RawEdge> = Vec::new();

        for (file, pr) in &parse_results {
            all_symbols.extend(pr.symbols.iter().cloned());
            all_chunks_by_file.push((file.clone(), pr.chunks.clone()));
            all_edges.extend(pr.edges.iter().cloned());
        }

        // Build symbol index from new symbols + existing DB symbols.
        let mut symbol_index = build_symbol_index(&all_symbols);
        let db_symbols = query_all_symbols_from_db(db).await?;
        for sym in db_symbols {
            symbol_index.entry(sym.name.clone()).or_default().push(sym);
        }

        // Embed chunks outside transaction.
        let embeddings = self.embed_all_chunks(&all_chunks_by_file, progress).await?;

        // Resolve edges.
        let resolved_edges = resolve_edges(&all_edges, &symbol_index);

        // Compute file stats before the transaction.
        let file_stats: Vec<(String, i64, i64)> = to_process
            .iter()
            .filter_map(|f| stat_file(f).map(|s| (f.clone(), s.mtime, s.size)))
            .collect();

        // Build the transaction.
        let mut txn = String::from("BEGIN TRANSACTION;\n");

        // Delete old data for all affected files.
        for file in &all_affected {
            append_delete_file_data(&mut txn, file);
        }

        // Insert new symbols.
        for sym in &all_symbols {
            append_upsert_symbol(&mut txn, sym);
        }

        // Insert edges.
        for (from, to, kind, line) in &resolved_edges {
            append_insert_edge(&mut txn, from, to, kind, *line);
        }

        // Insert chunks.
        let mut emb_iter = embeddings.iter();
        let mut chunk_vectors: Vec<(ChunkId, Vec<f32>)> = Vec::new();

        for (_file, chunks) in &all_chunks_by_file {
            for chunk in chunks {
                let emb = emb_iter.next().cloned().unwrap_or_default();
                append_insert_chunk(&mut txn, chunk, &emb);
                chunk_vectors.push((
                    ChunkId {
                        file: chunk.file.clone(),
                        line_start: chunk.line_start,
                        line_end: chunk.line_end,
                    },
                    emb,
                ));
            }
        }

        // Upsert file_meta for added/modified files.
        for (path, mtime, size) in &file_stats {
            append_upsert_file_meta(&mut txn, path, *mtime, *size, &self.repo);
        }

        // Delete file_meta for deleted files.
        for file in &to_delete {
            let escaped = escape_surreal(file);
            txn.push_str(&format!(
                "DELETE FROM file_meta WHERE path = '{escaped}';\n"
            ));
        }

        txn.push_str("COMMIT TRANSACTION;\n");

        db.query(&txn).await.context("incremental_run: transaction failed")?;

        Ok((all_affected, chunk_vectors))
    }

    // ─── Embedding helper ─────────────────────────────────────────────────

    /// Embed all chunks, reporting per-batch progress via `progress`.
    ///
    /// Progress advances at embedding batch boundaries (every `MAX_BATCH_SIZE`
    /// chunks). The numerator counts files whose last chunk has been embedded,
    /// using a per-file cumulative prefix over the flattened chunk list so the
    /// denominator and numerator always use the same file set and the bar
    /// reaches exactly 100%.
    async fn embed_all_chunks(
        &self,
        chunks_by_file: &[(String, Vec<crate::parsing::chunker::Chunk>)],
        progress: Option<&ProgressHandle>,
    ) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = chunks_by_file
            .iter()
            .flat_map(|(_, chunks)| chunks.iter().map(|c| c.content.clone()))
            .collect();

        if texts.is_empty() {
            // Nothing to embed — report total immediately so the bar completes.
            if let Some(ph) = progress {
                let total = chunks_by_file.len() as u64;
                ph.set_run_total(total).await;
                ph.set_processed(total).await;
            }
            return Ok(vec![]);
        }

        // Precompute cumulative chunk-end index for each file so we can map
        // "chunks done so far" → "files fully embedded".
        // cumulative[i] = index of the last chunk of file i in the flat list (exclusive end).
        let mut cumulative: Vec<usize> = Vec::with_capacity(chunks_by_file.len());
        let mut running = 0usize;
        for (_, chunks) in chunks_by_file {
            running += chunks.len();
            cumulative.push(running);
        }
        let total_files = chunks_by_file.len() as u64;

        // Report the denominator once the file set is known, before any I/O.
        if let Some(ph) = progress {
            ph.set_run_total(total_files).await;
        }

        match &self.voyage {
            Some(client) => {
                info!(count = texts.len(), "embedding chunks");
                let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
                let mut done: usize = 0;

                for batch in texts.chunks(MAX_BATCH_SIZE) {
                    let batch_vec: Vec<String> = batch.to_vec();
                    let embeddings = client.embed_batch(&batch_vec, InputType::Document).await?;
                    done += embeddings.len();
                    all_embeddings.extend(embeddings);

                    // Count how many files are fully embedded (all their chunks done).
                    if let Some(ph) = progress {
                        // Binary-search for the rightmost file whose cumulative end <= done.
                        let completed_files = cumulative.partition_point(|&end| end <= done) as u64;
                        ph.set_processed(completed_files).await;
                    }
                }

                Ok(all_embeddings)
            }
            None => {
                warn!("no embedding client configured; storing empty embeddings");
                // No network I/O — mark everything complete immediately.
                if let Some(ph) = progress {
                    ph.set_processed(total_files).await;
                }
                Ok(vec![vec![]; texts.len()])
            }
        }
    }
}

// ─── Parallel parsing ─────────────────────────────────────────────────────

fn parse_all_files_parallel(
    files: &[String],
) -> Vec<(String, crate::parsing::ParseResult)> {
    use rayon::prelude::*;

    files
        .par_iter()
        .filter_map(|file| {
            let source = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => {
                    warn!(file = %file, error = %e, "failed to read file");
                    return None;
                }
            };
            let result = parse_file(file, &source);
            Some((file.clone(), result))
        })
        .collect()
}

// ─── Symbol index helpers ─────────────────────────────────────────────────

fn build_symbol_index(symbols: &[Symbol]) -> HashMap<String, Vec<QualifiedSymbol>> {
    let mut index: HashMap<String, Vec<QualifiedSymbol>> = HashMap::new();
    for sym in symbols {
        index
            .entry(sym.qualified.name.clone())
            .or_default()
            .push(sym.qualified.clone());
    }
    index
}

async fn query_all_symbols_from_db(
    db: &Surreal<Db>,
) -> Result<Vec<QualifiedSymbol>> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Row {
        file: String,
        name: String,
    }

    let rows: Vec<Row> = db
        .query("SELECT file, name FROM symbol")
        .await
        .context("query all symbols")?
        .take(0)?;

    Ok(rows
        .into_iter()
        .map(|r| QualifiedSymbol {
            file: r.file,
            scope_path: vec![],
            name: r.name,
        })
        .collect())
}

// ─── Edge resolution ──────────────────────────────────────────────────────

fn resolve_edges(
    edges: &[RawEdge],
    symbol_index: &HashMap<String, Vec<QualifiedSymbol>>,
) -> Vec<(QualifiedSymbol, QualifiedSymbol, EdgeKind, u32)> {
    let mut resolved = Vec::new();

    for edge in edges {
        let to = match &edge.to {
            EdgeTarget::Resolved(qs) => qs.clone(),
            EdgeTarget::Unresolved { name, .. } => {
                match symbol_index.get(name) {
                    Some(candidates) if !candidates.is_empty() => {
                        let same_file = candidates
                            .iter()
                            .find(|c| c.file == edge.from.file);
                        same_file
                            .or_else(|| candidates.first())
                            .cloned()
                            .unwrap()
                    }
                    _ => {
                        debug!(name = %name, "dropping unresolved edge");
                        continue;
                    }
                }
            }
        };

        resolved.push((edge.from.clone(), to, edge.kind.clone(), edge.line));
    }

    resolved
}

// ─── SurrealQL escaping ───────────────────────────────────────────────────

/// Escape a string for safe embedding in a SurrealQL single-quoted literal.
/// Handles backslashes (must be first) and single quotes.
fn escape_surreal(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Format a Vec<f32> as a SurrealQL array literal: `[0.1, 0.2, ...]`.
fn format_embedding(emb: &[f32]) -> String {
    if emb.is_empty() {
        return "[]".to_string();
    }
    let inner: Vec<String> = emb.iter().map(|v| format!("{v}")).collect();
    format!("[{}]", inner.join(","))
}

// ─── Transaction query builders ───────────────────────────────────────────

/// Append DELETE statements for all data owned by `file`.
fn append_delete_file_data(txn: &mut String, file: &str) {
    let f = escape_surreal(file);
    txn.push_str(&format!("DELETE FROM calls WHERE in_file = '{f}' OR out_file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM uses WHERE in_file = '{f}' OR out_file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM imports WHERE in_file = '{f}' OR out_file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM contains WHERE in_file = '{f}' OR out_file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM implements WHERE in_file = '{f}' OR out_file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM symbol WHERE file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM chunk WHERE file = '{f}';\n"));
    txn.push_str(&format!("DELETE FROM file_meta WHERE path = '{f}';\n"));
}

/// Append an UPSERT statement for `sym`.
fn append_upsert_symbol(txn: &mut String, sym: &Symbol) {
    use crate::store::ops::kind_to_str;

    let fqn = escape_surreal(&sym.qualified.fqn());
    let name = escape_surreal(&sym.qualified.name);
    let kind = kind_to_str(&sym.kind);
    let file = escape_surreal(&sym.qualified.file);
    let ls = sym.line_start as i64;
    let le = sym.line_end as i64;
    let sig = sym
        .signature
        .as_deref()
        .map(|s| format!("'{}'", escape_surreal(s)))
        .unwrap_or_else(|| "NONE".to_string());
    let parent = sym
        .parent_fqn
        .as_deref()
        .map(|p| format!("'symbol:⟨{}⟩'", escape_surreal(p)))
        .unwrap_or_else(|| "NONE".to_string());

    txn.push_str(&format!(
        "UPSERT symbol:`⟨{fqn}⟩` SET \
         name = '{name}', kind = '{kind}', file = '{file}', \
         line_start = {ls}, line_end = {le}, \
         signature = {sig}, parent = {parent};\n"
    ));
}

/// Append a RELATE statement for an edge.
fn append_insert_edge(
    txn: &mut String,
    from: &QualifiedSymbol,
    to: &QualifiedSymbol,
    kind: &EdgeKind,
    line: u32,
) {
    let from_fqn = escape_surreal(&from.fqn());
    let to_fqn = escape_surreal(&to.fqn());
    let in_file = escape_surreal(&from.file);
    let out_file = escape_surreal(&to.file);
    let table = match kind {
        EdgeKind::Calls => "calls",
        EdgeKind::Uses => "uses",
        EdgeKind::Imports => "imports",
        EdgeKind::Contains => "contains",
        EdgeKind::Implements => "implements",
    };

    if matches!(kind, EdgeKind::Calls) {
        txn.push_str(&format!(
            "RELATE symbol:`⟨{from_fqn}⟩`->{table}->symbol:`⟨{to_fqn}⟩` \
             SET line = {line}, in_file = '{in_file}', out_file = '{out_file}';\n"
        ));
    } else {
        txn.push_str(&format!(
            "RELATE symbol:`⟨{from_fqn}⟩`->{table}->symbol:`⟨{to_fqn}⟩` \
             SET in_file = '{in_file}', out_file = '{out_file}';\n"
        ));
    }
}

/// Append a CREATE chunk statement.
fn append_insert_chunk(txn: &mut String, chunk: &crate::parsing::chunker::Chunk, emb: &[f32]) {
    let file = escape_surreal(&chunk.file);
    let content = escape_surreal(&chunk.content);
    let ls = chunk.line_start as i64;
    let le = chunk.line_end as i64;
    let embedding = format_embedding(emb);
    let sym_ref = chunk
        .symbol_ref
        .as_deref()
        .map(|s| format!("'symbol:⟨{}⟩'", escape_surreal(s)))
        .unwrap_or_else(|| "NONE".to_string());

    txn.push_str(&format!(
        "CREATE chunk SET file = '{file}', line_start = {ls}, line_end = {le}, \
         content = '{content}', embedding = {embedding}, symbol_ref = {sym_ref};\n"
    ));
}

/// Append an UPSERT file_meta statement.
fn append_upsert_file_meta(txn: &mut String, path: &str, mtime: i64, size: i64, repo: &str) {
    let p = escape_surreal(path);
    let r = escape_surreal(repo);
    txn.push_str(&format!(
        "UPSERT file_meta SET path = '{p}', mtime = {mtime}, size = {size}, repo = '{r}' \
         WHERE path = '{p}';\n"
    ));
}
