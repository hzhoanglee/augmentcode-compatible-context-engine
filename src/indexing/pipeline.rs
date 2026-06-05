use std::collections::HashMap;

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
use crate::store::ops::get_all_file_meta;
use crate::vector::{ChunkId, VectorIndex};

pub struct IndexPipelineStats {
    pub indexed_files: u64,
    pub total_files: u64,
}

/// Runs the parse → embed → store pipeline for one repo.
pub struct IndexPipeline {
    repo: String,
    voyage: Option<VoyageClient>,
}

impl IndexPipeline {
    pub fn new(repo: String, voyage: Option<VoyageClient>) -> Self {
        Self { repo, voyage }
    }

    /// Run the pipeline against the shared `db` handle.
    /// - `changes = None` → incremental scan (detect changes from mtime).
    /// - `changes = Some(list)` → process only the given file changes.
    /// - `force_rebuild = true` → clear and re-embed everything, ignoring staleness.
    /// - `progress` → optional handle for reporting live progress to the status map.
    pub async fn run(
        &self,
        db: &Surreal<Db>,
        changes: Option<Vec<FileChange>>,
        force_rebuild: bool,
        vector_index: Option<&tokio::sync::RwLock<VectorIndex>>,
        progress: Option<ProgressHandle>,
    ) -> Result<IndexPipelineStats> {
        // Check if first run (no file_meta at all).
        let stored_meta = get_all_file_meta(db, &self.repo).await?;
        let is_first_run = stored_meta.is_empty();

        let total_files = walk_repo(&self.repo).len() as u64;

        if is_first_run || force_rebuild {
            if force_rebuild && !is_first_run {
                info!(repo = %self.repo, "forced full rebuild");
            } else {
                info!(repo = %self.repo, "first run — full rebuild");
            }
            let new_vectors = self.full_rebuild(db, progress.as_ref()).await?;
            if let Some(vi) = vector_index {
                let mut guard = vi.write().await;
                guard.remove_repo(&self.repo);
                guard.insert(&new_vectors);
            }
            let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
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
        let (removed_files, new_vectors) = self.incremental_run(db, file_changes, progress.as_ref()).await?;

        if let Some(vi) = vector_index {
            let mut guard = vi.write().await;
            for file in &removed_files {
                guard.remove_file(file);
            }
            guard.insert(&new_vectors);
        }

        let indexed = get_all_file_meta(db, &self.repo).await?.len() as u64;
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

        let resp = db.query(&txn).await.context("full_rebuild: transaction failed")?;
        check_transaction(resp).context("full_rebuild")?;

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

        let resp = db.query(&txn).await.context("incremental_run: transaction failed")?;
        check_transaction(resp).context("incremental_run")?;

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

/// Inspect a transaction response and return the FIRST meaningful per-statement
/// error, or `Ok(())` if every statement succeeded.
///
/// `Response::check()` is not usable here: when a SurrealDB transaction rolls
/// back, EVERY statement is annotated with the same generic "The query was not
/// executed due to a failed transaction" message, and `check()` surfaces only
/// the first of those — hiding the one statement whose real error (e.g. a type
/// violation) actually triggered the rollback. `take_errors()` returns the full
/// `index → error` map, so we skip the generic cascade messages and report the
/// true culprit with its statement index.
fn check_transaction(mut resp: surrealdb::Response) -> Result<()> {
    let errors = resp.take_errors();
    if errors.is_empty() {
        return Ok(());
    }
    const GENERIC: &str = "The query was not executed due to a failed transaction";
    // Prefer the first error that is NOT the generic rollback-cascade message.
    let culprit = errors
        .iter()
        .filter(|(_, e)| !e.to_string().contains(GENERIC))
        .min_by_key(|(idx, _)| **idx);
    match culprit {
        Some((idx, e)) => {
            anyhow::bail!("transaction rolled back — statement #{idx} failed: {e}")
        }
        None => {
            // Only generic cascade messages present (rare): report the lowest index.
            let (idx, e) = errors.iter().min_by_key(|(idx, _)| **idx).unwrap();
            anyhow::bail!("transaction rolled back — statement #{idx}: {e}")
        }
    }
}

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
    let inner: Vec<String> = emb.iter().map(|v| format!("{v:?}")).collect();
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

// ─── format_embedding correctness tests ──────────────────────────────────────
//
// Regression suite for the format_embedding fix.
// Root cause investigation showed that `format!("{v}")` for f32 emits bare integer
// tokens ("0", "1", "-0") for whole-number floats. SurrealDB's SCHEMAFULL
// `array<float>` field accepts those via coercion today but the correct fix is to
// always emit a float-form literal so the stored type is unambiguous.
// The fix uses `{v:?}` (Rust Debug format for f32) which always includes either a
// decimal point (`0.0`, `1.0`) or scientific-notation exponent (`1e-7`) — making
// every element parseable as a SurrealQL float literal without coercion.
#[cfg(test)]
mod format_embedding_tests {
    use super::*;
    use crate::store::open_db;
    use crate::store::ops::count_chunks;
    use tempfile::TempDir;

    /// After the fix, format_embedding must never emit bare integer tokens.
    /// A bare integer token is a value with neither '.' nor 'e'/'E'.
    #[test]
    fn format_embedding_always_emits_float_literals() {
        let cases: &[&[f32]] = &[
            &[0.0, 1.0, -1.0, 0.5],
            &[-0.0, 2.0, -2.0, 100.0],
            &[0.023445, -0.987654, 0.5],
            &[1.0e-7, -1.0e7, f32::MIN_POSITIVE],
        ];
        for input in cases {
            let result = format_embedding(input);
            println!("format_embedding({input:?}) = {result}");
            // Each comma-separated element must contain '.' or 'e'/'E'.
            let inner = result.trim_start_matches('[').trim_end_matches(']');
            for token in inner.split(',') {
                let t = token.trim();
                assert!(
                    t.contains('.') || t.contains('e') || t.contains('E'),
                    "token {t:?} in {result:?} is not a float literal \
                     (no '.' or 'e'); would be parsed as integer in SurrealQL"
                );
            }
        }
    }

    /// format_embedding with whole-number floats must produce `0.0`, `1.0` etc.
    /// (specifically checking the Debug format gives correct output).
    #[test]
    fn whole_number_floats_get_decimal_point() {
        let result = format_embedding(&[0.0, 1.0, -1.0, -0.0]);
        println!("whole-number floats: {result}");
        assert!(result.contains("0.0"), "0.0 must appear as '0.0', got: {result}");
        assert!(result.contains("1.0"), "1.0 must appear as '1.0', got: {result}");
        assert!(result.contains("-1.0"), "-1.0 must appear as '-1.0', got: {result}");
    }

    /// Chunks with whole-number embedding components (0.0, 1.0) must commit and
    /// persist when the embedding is generated via format_embedding.
    /// This is the key regression test: before the fix, the formatted embedding
    /// would produce integer tokens that could cause issues; after the fix,
    /// the chunk persists correctly with proper float literals.
    #[tokio::test]
    async fn chunk_with_whole_number_embedding_persists() {
        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), "/test/embed_regression").await.expect("open db");

        // Use format_embedding with whole-number components — the same path as production.
        let embedding_str = format_embedding(&[0.0, 1.0, -1.0, 0.5, -0.5]);
        println!("REGRESSION: embedding literal = {embedding_str}");

        let txn = format!(
            "BEGIN TRANSACTION;\n\
             CREATE chunk SET file = '/test/foo.rs', line_start = 1, line_end = 5, \
             content = 'fn foo() {{}}', embedding = {embedding_str}, symbol_ref = NONE;\n\
             COMMIT TRANSACTION;\n"
        );

        db.query(&txn).await.expect(".await must not err")
            .check().expect("chunk with formatted embedding must commit without per-statement error");

        let count = count_chunks(&db).await.unwrap();
        assert_eq!(count, 1,
            "chunk must persist after format_embedding fix (got {count}); \
             integer token in embedding may have triggered a rollback");
    }

    /// format_embedding round-trips float precision: values stored and retrieved
    /// should be numerically equivalent to the original f32 (within f32 precision).
    #[tokio::test]
    async fn format_embedding_round_trips_precision() {
        use serde::Deserialize;

        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), "/test/embed_roundtrip").await.expect("open db");

        let original: Vec<f32> = vec![0.023445, -0.987654, 0.5, 0.0, 1.0, -1.0, 1.234567e-7];
        let embedding_str = format_embedding(&original);
        println!("ROUNDTRIP: embedding_str = {embedding_str}");

        let txn = format!(
            "BEGIN TRANSACTION;\n\
             CREATE chunk SET file = '/test/rt.rs', line_start = 1, line_end = 1, \
             content = 'x', embedding = {embedding_str}, symbol_ref = NONE;\n\
             COMMIT TRANSACTION;\n"
        );
        db.query(&txn).await.expect(".await").check().expect(".check");

        #[derive(Deserialize)]
        struct Row { embedding: Vec<f64> }
        let rows: Vec<Row> = db
            .query("SELECT embedding FROM chunk LIMIT 1")
            .await.expect("select").take(0).expect("take");

        let stored = &rows.first().expect("must have a row").embedding;
        assert_eq!(stored.len(), original.len(), "embedding length must round-trip");
        for (i, (&orig, &stored_v)) in original.iter().zip(stored.iter()).enumerate() {
            let diff = (orig as f64 - stored_v).abs();
            assert!(
                diff < 1e-6,
                "element {i}: original={orig}, stored={stored_v}, diff={diff}; \
                 float precision must survive format_embedding round-trip"
            );
        }
        println!("ROUNDTRIP: all {} elements match within 1e-6", original.len());
    }
}

// ─── STEP 3 regression test ───────────────────────────────────────────────
//
// Drives the real full_rebuild write path end-to-end (voyage = None so no
// network) and asserts that chunks, files, and symbols all persist.
// Also includes a voyage-scale probe for 1024-dim embeddings.
#[cfg(test)]
mod end_to_end_persist {
    use super::*;
    use crate::store::open_db;
    use crate::store::ops::{count_chunks, count_indexed_files, count_symbols};
    use tempfile::TempDir;

    /// Write a tiny Rust source file into `dir` and return its absolute path.
    fn write_test_file(dir: &std::path::Path) -> String {
        let path = dir.join("sample.rs");
        std::fs::write(
            &path,
            "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nfn subtract(a: i32, b: i32) -> i32 {\n    a - b\n}\n",
        )
        .expect("write test file");
        path.to_str().unwrap().replace('\\', "/")
    }

    /// Full-rebuild of the real context-engine-rs source tree (voyage=None).
    /// This exercises the SAME code path and file set as the live failing run.
    #[tokio::test]
    async fn full_rebuild_real_source_tree_voyage_none() {
        let home = TempDir::new().unwrap();
        let repo = env!("CARGO_MANIFEST_DIR").replace('\\', "/");
        println!("REAL-TREE PROBE: repo = {repo}");

        let db = open_db(home.path(), &repo).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let result = pipeline.run(&db, None, true, None, None).await;
        println!("REAL-TREE PROBE: result = {:?}", result.as_ref().map(|s| (s.indexed_files, s.total_files)));

        let chunks = count_chunks(&db).await.unwrap();
        let symbols = count_symbols(&db).await.unwrap();
        let files = count_indexed_files(&db, &repo).await.unwrap();
        println!("REAL-TREE PROBE: chunks={chunks}, symbols={symbols}, files={files}");

        assert!(result.is_ok(), "full_rebuild of real source tree must succeed (got: {:?})", result.err());
        assert!(chunks > 0, "must have chunks after full_rebuild of real source tree");
        assert!(files > 0, "must have indexed files");
    }

    /// VOYAGE-SCALE probe — simulate a full transaction with 1024-dim embeddings
    /// using format_embedding, to reproduce the production rollback condition.
    #[tokio::test]
    async fn voyage_scale_embedding_transaction_probe() {
        let home = TempDir::new().unwrap();
        let db = open_db(home.path(), "/test/voyage_scale").await.expect("open db");

        // Simulate Voyage AI embeddings: normalized vectors, values in (-1, 1).
        // Use a deterministic pattern that includes whole-number components (0.0, 1.0, -1.0)
        // alongside fractional values, matching what Voyage would actually return.
        let make_emb = |seed: usize| -> Vec<f32> {
            (0..1024usize).map(|i| {
                let v = ((seed * 7 + i * 13) % 2001) as f32 / 1000.0 - 1.0;
                v.max(-1.0_f32).min(1.0_f32)
            }).collect()
        };

        let mut txn = String::from("BEGIN TRANSACTION;\n");

        txn.push_str("UPSERT symbol:`⟨/test/s.rs::/test/s.rs::test_fn⟩` SET \
            name = 'test_fn', kind = 'function', file = '/test/s.rs', \
            line_start = 1, line_end = 10, signature = NONE, parent = NONE;\n");

        for i in 0..50usize {
            let emb = make_emb(i);
            let emb_str = format_embedding(&emb);
            txn.push_str(&format!(
                "CREATE chunk SET file = '/test/s.rs', line_start = {ls}, line_end = {le}, \
                 content = 'fn chunk_{i}() {{}}', embedding = {emb_str}, \
                 symbol_ref = 'symbol:⟨/test/s.rs::/test/s.rs::test_fn⟩';\n",
                ls = i * 10, le = i * 10 + 9
            ));
        }

        txn.push_str("UPSERT file_meta SET path = '/test/s.rs', mtime = 12345, size = 1000, \
            repo = '/test' WHERE path = '/test/s.rs';\n");
        txn.push_str("COMMIT TRANSACTION;\n");

        println!("VOYAGE-SCALE PROBE: txn length = {} bytes", txn.len());

        let mut resp = db.query(&txn).await.expect(".await must not err");
        let errors: Vec<_> = resp.take_errors().into_iter().collect();
        println!("VOYAGE-SCALE PROBE: per-statement errors = {:?}", errors);

        let chunk_count = count_chunks(&db).await.unwrap();
        let symbol_count = count_symbols(&db).await.unwrap();
        println!("VOYAGE-SCALE PROBE: chunk_count={chunk_count}, symbol_count={symbol_count}");
        println!("VOYAGE-SCALE PROBE: RESULT — 1024-dim embedding transaction {}",
            if errors.is_empty() && chunk_count == 50 { "COMMITS OK" }
            else if errors.is_empty() { "NO ERROR BUT WRONG COUNTS" }
            else { "FAILS" });
    }

    /// Full-rebuild through the real IndexPipeline (voyage=None) must persist
    /// chunks, indexed files, and symbols — proving the transaction no longer
    /// rolls back silently.
    #[tokio::test]
    async fn full_rebuild_persists_chunks_files_symbols() {
        let home = TempDir::new().unwrap();
        let repo_dir = TempDir::new().unwrap();

        let _file_path = write_test_file(repo_dir.path());
        let repo = repo_dir.path().to_str().unwrap().replace('\\', "/");

        let db = open_db(home.path(), &repo).await.expect("open db");
        let pipeline = IndexPipeline::new(repo.clone(), None);

        let stats = pipeline
            .run(&db, None, true, None, None)
            .await
            .expect("full_rebuild must succeed");

        let chunks = count_chunks(&db).await.unwrap();
        let files = count_indexed_files(&db, &repo).await.unwrap();
        let symbols = count_symbols(&db).await.unwrap();

        println!("STEP3 — indexed_files={}, total_files={}", stats.indexed_files, stats.total_files);
        println!("STEP3 — chunks={chunks}, files={files}, symbols={symbols}");

        assert!(chunks > 0,
            "chunks must be > 0 after full_rebuild (got {chunks}); transaction still rolling back");
        assert!(files > 0,
            "indexed files must be > 0 after full_rebuild (got {files})");
        assert!(symbols > 0,
            "symbols must be > 0 after full_rebuild (got {symbols})");
        assert_eq!(stats.indexed_files, files,
            "stats.indexed_files must match count_indexed_files");
    }
}
