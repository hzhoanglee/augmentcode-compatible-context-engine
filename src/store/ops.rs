use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use crate::parsing::symbols::{QualifiedSymbol, Symbol, SymbolKind};
use crate::parsing::relations::EdgeKind;
use crate::parsing::chunker::Chunk;

// ─── FileMeta ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMeta {
    pub path: String,
    pub mtime: i64,
    pub size: i64,
    pub repo: String,
}

// ─── IndexMeta ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub key: String,
    pub value: String,
}

// ─── DB row types for queries ─────────────────────────────────────────────

pub fn kind_to_str(k: &SymbolKind) -> &'static str {
    match k {
        SymbolKind::Function => "function",
        SymbolKind::Method => "method",
        SymbolKind::Struct => "struct",
        SymbolKind::Trait => "trait",
        SymbolKind::Impl => "impl",
        SymbolKind::Class => "class",
        SymbolKind::Module => "module",
        SymbolKind::Interface => "interface",
    }
}

// ─── Delete operations (used in transactions) ────────────────────────────

/// Delete all edges, symbols, chunks, and file_meta for a given file path.
/// Edge deletion happens first (while symbol IDs still exist for traversal).
pub async fn delete_file_data(db: &Surreal<Db>, file_path: &str) -> Result<()> {
    // 1. Delete edges first (all relation tables by in_file or out_file).
    let path = file_path.to_string();

    db.query("DELETE FROM calls WHERE in_file = $path OR out_file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete calls")?;

    db.query("DELETE FROM uses WHERE in_file = $path OR out_file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete uses")?;

    db.query("DELETE FROM imports WHERE in_file = $path OR out_file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete imports")?;

    db.query("DELETE FROM contains WHERE in_file = $path OR out_file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete contains")?;

    db.query("DELETE FROM implements WHERE in_file = $path OR out_file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete implements")?;

    // 2. Delete symbols.
    db.query("DELETE FROM symbol WHERE file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete symbols")?;

    // 3. Delete chunks.
    db.query("DELETE FROM chunk WHERE file = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete chunks")?;

    // 4. Delete file_meta.
    db.query("DELETE FROM file_meta WHERE path = $path")
        .bind(("path", path.clone()))
        .await
        .context("delete file_meta")?;

    Ok(())
}

/// Delete ALL data — used for full rebuild.
pub async fn delete_all_data(db: &Surreal<Db>) -> Result<()> {
    // Edges first.
    db.query("DELETE FROM calls").await.context("delete all calls")?;
    db.query("DELETE FROM uses").await.context("delete all uses")?;
    db.query("DELETE FROM imports").await.context("delete all imports")?;
    db.query("DELETE FROM contains").await.context("delete all contains")?;
    db.query("DELETE FROM implements").await.context("delete all implements")?;
    // Then symbols, chunks, file_meta.
    db.query("DELETE FROM symbol").await.context("delete all symbols")?;
    db.query("DELETE FROM chunk").await.context("delete all chunks")?;
    db.query("DELETE FROM file_meta").await.context("delete all file_meta")?;
    Ok(())
}

// ─── Insert operations ────────────────────────────────────────────────────

/// Upsert a symbol using its deterministic record ID.
pub async fn upsert_symbol(db: &Surreal<Db>, sym: &Symbol) -> Result<()> {
    let record_id = sym.qualified.record_id();
    let kind_str = kind_to_str(&sym.kind);
    let parent_id = sym.parent_fqn.as_ref().map(|fqn| {
        format!("symbol:⟨{}⟩", fqn)
    });

    db.query(
        "UPSERT type::thing($id) SET \
         name = $name, kind = $kind, file = $file, \
         line_start = $line_start, line_end = $line_end, \
         signature = $signature, parent = $parent",
    )
    .bind(("id", record_id))
    .bind(("name", sym.qualified.name.clone()))
    .bind(("kind", kind_str.to_string()))
    .bind(("file", sym.qualified.file.clone()))
    .bind(("line_start", sym.line_start as i64))
    .bind(("line_end", sym.line_end as i64))
    .bind(("signature", sym.signature.clone()))
    .bind(("parent", parent_id))
    .await
    .context("upsert symbol")?;

    Ok(())
}

/// Insert a resolved edge using RELATE.
pub async fn insert_edge(
    db: &Surreal<Db>,
    from: &QualifiedSymbol,
    to: &QualifiedSymbol,
    kind: &EdgeKind,
    line: u32,
) -> Result<()> {
    let from_id = from.record_id();
    let to_id = to.record_id();
    let in_file = from.file.clone();
    let out_file = to.file.clone();

    match kind {
        EdgeKind::Calls => {
            db.query(
                "RELATE type::thing($from)->calls->type::thing($to) \
                 SET line = $line, in_file = $in_file, out_file = $out_file",
            )
            .bind(("from", from_id))
            .bind(("to", to_id))
            .bind(("line", line as i64))
            .bind(("in_file", in_file))
            .bind(("out_file", out_file))
            .await
            .context("insert calls edge")?;
        }
        EdgeKind::Uses => {
            db.query(
                "RELATE type::thing($from)->uses->type::thing($to) \
                 SET in_file = $in_file, out_file = $out_file",
            )
            .bind(("from", from_id))
            .bind(("to", to_id))
            .bind(("in_file", in_file))
            .bind(("out_file", out_file))
            .await
            .context("insert uses edge")?;
        }
        EdgeKind::Imports => {
            db.query(
                "RELATE type::thing($from)->imports->type::thing($to) \
                 SET in_file = $in_file, out_file = $out_file",
            )
            .bind(("from", from_id))
            .bind(("to", to_id))
            .bind(("in_file", in_file))
            .bind(("out_file", out_file))
            .await
            .context("insert imports edge")?;
        }
        EdgeKind::Contains => {
            db.query(
                "RELATE type::thing($from)->contains->type::thing($to) \
                 SET in_file = $in_file, out_file = $out_file",
            )
            .bind(("from", from_id))
            .bind(("to", to_id))
            .bind(("in_file", in_file))
            .bind(("out_file", out_file))
            .await
            .context("insert contains edge")?;
        }
        EdgeKind::Implements => {
            db.query(
                "RELATE type::thing($from)->implements->type::thing($to) \
                 SET in_file = $in_file, out_file = $out_file",
            )
            .bind(("from", from_id))
            .bind(("to", to_id))
            .bind(("in_file", in_file))
            .bind(("out_file", out_file))
            .await
            .context("insert implements edge")?;
        }
    }

    Ok(())
}

/// Insert a chunk with its embedding.
pub async fn insert_chunk(db: &Surreal<Db>, chunk: &Chunk, embedding: Vec<f32>) -> Result<()> {
    let symbol_ref = chunk.symbol_ref.as_ref().map(|fqn| format!("symbol:⟨{}⟩", fqn));

    db.query(
        "CREATE chunk SET \
         file = $file, line_start = $line_start, line_end = $line_end, \
         content = $content, embedding = $embedding, symbol_ref = $symbol_ref",
    )
    .bind(("file", chunk.file.clone()))
    .bind(("line_start", chunk.line_start as i64))
    .bind(("line_end", chunk.line_end as i64))
    .bind(("content", chunk.content.clone()))
    .bind(("embedding", embedding))
    .bind(("symbol_ref", symbol_ref))
    .await
    .context("insert chunk")?;

    Ok(())
}

/// Upsert file metadata.
pub async fn upsert_file_meta(db: &Surreal<Db>, meta: &FileMeta) -> Result<()> {
    db.query(
        "UPSERT file_meta SET path = $path, mtime = $mtime, size = $size, repo = $repo \
         WHERE path = $path",
    )
    .bind(("path", meta.path.clone()))
    .bind(("mtime", meta.mtime))
    .bind(("size", meta.size))
    .bind(("repo", meta.repo.clone()))
    .await
    .context("upsert file_meta")?;

    Ok(())
}

// ─── Query operations ─────────────────────────────────────────────────────

/// Fetch all file_meta rows for a given repo.
pub async fn get_all_file_meta(db: &Surreal<Db>, repo: &str) -> Result<Vec<FileMeta>> {
    let rows: Vec<FileMeta> = db
        .query("SELECT path, mtime, size, repo FROM file_meta WHERE repo = $repo")
        .bind(("repo", repo.to_string()))
        .await
        .context("get all file_meta")?
        .take(0)?;
    Ok(rows)
}

/// Get a single index_meta value by key.
pub async fn get_meta(db: &Surreal<Db>, key: &str) -> Result<Option<String>> {
    let rows: Vec<IndexMeta> = db
        .query("SELECT key, value FROM index_meta WHERE key = $key")
        .bind(("key", key.to_string()))
        .await
        .context("get index_meta")?
        .take(0)?;
    Ok(rows.into_iter().next().map(|r| r.value))
}

/// Set an index_meta key/value.
pub async fn set_meta(db: &Surreal<Db>, key: &str, value: &str) -> Result<()> {
    db.query(
        "UPSERT index_meta SET key = $key, value = $value WHERE key = $key",
    )
    .bind(("key", key.to_string()))
    .bind(("value", value.to_string()))
    .await
    .context("set index_meta")?;
    Ok(())
}

/// Get all symbols from a given file (used for edge resolution).
pub async fn get_symbols_for_file(
    db: &Surreal<Db>,
    file: &str,
) -> Result<Vec<QualifiedSymbol>> {
    #[derive(Deserialize)]
    struct Row {
        file: String,
        name: String,
    }
    let rows: Vec<Row> = db
        .query("SELECT file, name FROM symbol WHERE file = $file")
        .bind(("file", file.to_string()))
        .await
        .context("get symbols for file")?
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

/// Find a symbol by name across all files (for cross-file edge resolution).
pub async fn find_symbol_by_name(
    db: &Surreal<Db>,
    name: &str,
) -> Result<Vec<QualifiedSymbol>> {
    #[derive(Deserialize)]
    struct Row {
        file: String,
        name: String,
    }
    let rows: Vec<Row> = db
        .query("SELECT file, name FROM symbol WHERE name = $name")
        .bind(("name", name.to_string()))
        .await
        .context("find symbol by name")?
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

/// Count indexed files for a repo.
pub async fn count_indexed_files(db: &Surreal<Db>, repo: &str) -> Result<u64> {
    #[derive(Deserialize)]
    struct Row {
        count: i64,
    }
    let rows: Vec<Row> = db
        .query("SELECT count() AS count FROM file_meta WHERE repo = $repo GROUP ALL")
        .bind(("repo", repo.to_string()))
        .await
        .context("count indexed files")?
        .take(0)?;
    Ok(rows.first().map(|r| r.count as u64).unwrap_or(0))
}

// ─── Index Explorer queries (read-only, bounded) ──────────────────────────
//
// Every helper below is capped with LIMIT or a count aggregate so a
// Linux-kernel-scale index never streams unbounded rows into the HTTP layer
// or the browser. Embeddings are reduced to their length server-side
// (`array::len`) so the float vectors never cross the wire.

#[derive(Deserialize)]
struct CountRow {
    count: i64,
}

/// Total chunk rows stored (whole DB — one DB per repo, so this is per-repo).
pub async fn count_chunks(db: &Surreal<Db>) -> Result<u64> {
    let rows: Vec<CountRow> = db
        .query("SELECT count() AS count FROM chunk GROUP ALL")
        .await
        .context("count chunks")?
        .take(0)?;
    Ok(rows.first().map(|r| r.count as u64).unwrap_or(0))
}

/// Total symbol rows stored.
pub async fn count_symbols(db: &Surreal<Db>) -> Result<u64> {
    let rows: Vec<CountRow> = db
        .query("SELECT count() AS count FROM symbol GROUP ALL")
        .await
        .context("count symbols")?
        .take(0)?;
    Ok(rows.first().map(|r| r.count as u64).unwrap_or(0))
}

/// Sample one stored embedding's dimensionality, or 0 if none embedded yet.
pub async fn sample_embedding_dim(db: &Surreal<Db>) -> Result<u64> {
    let rows: Vec<CountRow> = db
        .query(
            "SELECT array::len(embedding) AS count FROM chunk \
             WHERE embedding != [] LIMIT 1",
        )
        .await
        .context("sample embedding dim")?
        .take(0)?;
    Ok(rows.first().map(|r| r.count.max(0) as u64).unwrap_or(0))
}

/// One row in the file browser: path, language-agnostic metadata, and chunk count.
#[derive(Debug, Clone, Serialize)]
pub struct FileBrowserRow {
    pub path: String,
    pub mtime: i64,
    pub size: i64,
    pub chunks: u64,
}

/// Return a bounded, alphabetically-ordered page of indexed files for a repo,
/// each annotated with its chunk count. `limit` is hard-capped by the caller.
pub async fn files_page(
    db: &Surreal<Db>,
    repo: &str,
    limit: usize,
) -> Result<Vec<FileBrowserRow>> {
    #[derive(Deserialize)]
    struct MetaRow {
        path: String,
        mtime: i64,
        size: i64,
    }

    let metas: Vec<MetaRow> = db
        .query(
            "SELECT path, mtime, size FROM file_meta \
             WHERE repo = $repo ORDER BY path LIMIT $limit",
        )
        .bind(("repo", repo.to_string()))
        .bind(("limit", limit as i64))
        .await
        .context("files_page: file_meta")?
        .take(0)?;

    // Chunk counts grouped by file in a single query, then joined in memory.
    #[derive(Deserialize)]
    struct GroupRow {
        file: String,
        count: i64,
    }
    let groups: Vec<GroupRow> = db
        .query("SELECT file, count() AS count FROM chunk GROUP BY file")
        .await
        .context("files_page: chunk counts")?
        .take(0)?;

    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for g in groups {
        counts.insert(g.file, g.count.max(0) as u64);
    }

    Ok(metas
        .into_iter()
        .map(|m| FileBrowserRow {
            chunks: counts.get(&m.path).copied().unwrap_or(0),
            path: m.path,
            mtime: m.mtime,
            size: m.size,
        })
        .collect())
}

/// A chunk detail row (no embedding floats — only the dimension count).
#[derive(Debug, Clone, Serialize)]
pub struct ChunkDetailRow {
    pub line_start: i64,
    pub line_end: i64,
    pub content: String,
    pub embedding_dim: u64,
    pub symbol: Option<String>,
}

/// Return the chunks for a single file, ordered by line, bounded by `limit`.
pub async fn chunks_for_file(
    db: &Surreal<Db>,
    file: &str,
    limit: usize,
) -> Result<Vec<ChunkDetailRow>> {
    #[derive(Deserialize)]
    struct Row {
        line_start: i64,
        line_end: i64,
        content: String,
        embedding_dim: i64,
        symbol_ref: Option<String>,
    }
    let rows: Vec<Row> = db
        .query(
            "SELECT line_start, line_end, content, \
             array::len(embedding) AS embedding_dim, symbol_ref \
             FROM chunk WHERE file = $file ORDER BY line_start LIMIT $limit",
        )
        .bind(("file", file.to_string()))
        .bind(("limit", limit as i64))
        .await
        .context("chunks_for_file")?
        .take(0)?;

    Ok(rows
        .into_iter()
        .map(|r| ChunkDetailRow {
            line_start: r.line_start,
            line_end: r.line_end,
            content: r.content,
            embedding_dim: r.embedding_dim.max(0) as u64,
            symbol: r.symbol_ref.as_deref().and_then(strip_symbol_ref),
        })
        .collect())
}

/// A node in the call graph (one symbol).
#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line_start: i64,
    pub line_end: i64,
}

/// An edge in the call graph (caller → callee).
#[derive(Debug, Clone, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
}

/// The call graph payload: nodes + edges, both bounded.
#[derive(Debug, Clone, Serialize)]
pub struct CallGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// True if the result was capped (more edges/symbols exist in the index).
    pub truncated: bool,
}

/// Build a bounded node-link view of the `calls` relation.
///
/// Strategy: take up to `edge_limit` call edges, collect the symbols they touch
/// (capped at `node_limit`), then emit the induced subgraph. Edges referencing a
/// symbol that was dropped by the node cap are themselves dropped, so the graph
/// is always internally consistent.
pub async fn call_graph(
    db: &Surreal<Db>,
    edge_limit: usize,
    node_limit: usize,
) -> Result<CallGraph> {
    // Symbol endpoints are stored as record links; fetch FQN + metadata via the
    // relation's in/out, plus the file fields already denormalized on the edge.
    #[derive(Deserialize)]
    struct EdgeRow {
        #[serde(rename = "in_name")]
        in_name: Option<String>,
        #[serde(rename = "out_name")]
        out_name: Option<String>,
        in_file: String,
        out_file: String,
    }

    let edge_rows: Vec<EdgeRow> = db
        .query(
            "SELECT in.name AS in_name, out.name AS out_name, in_file, out_file \
             FROM calls LIMIT $limit",
        )
        .bind(("limit", edge_limit as i64))
        .await
        .context("call_graph: edges")?
        .take(0)?;

    let total_edges = edge_rows.len();

    // Pull symbol metadata for nodes (bounded). Keyed by file::name.
    #[derive(Deserialize)]
    struct SymRow {
        name: String,
        kind: String,
        file: String,
        line_start: i64,
        line_end: i64,
    }
    let sym_rows: Vec<SymRow> = db
        .query(
            "SELECT name, kind, file, line_start, line_end FROM symbol LIMIT $limit",
        )
        .bind(("limit", node_limit as i64))
        .await
        .context("call_graph: symbols")?
        .take(0)?;

    let mut nodes: Vec<GraphNode> = Vec::with_capacity(sym_rows.len());
    let mut node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for s in sym_rows {
        let id = format!("{}::{}", s.file, s.name);
        if node_ids.insert(id.clone()) {
            nodes.push(GraphNode {
                id,
                name: s.name,
                kind: s.kind,
                file: s.file,
                line_start: s.line_start,
                line_end: s.line_end,
            });
        }
    }

    let mut edges: Vec<GraphEdge> = Vec::new();
    for e in edge_rows {
        let (in_name, out_name) = match (e.in_name, e.out_name) {
            (Some(i), Some(o)) => (i, o),
            _ => continue,
        };
        let source = format!("{}::{}", e.in_file, in_name);
        let target = format!("{}::{}", e.out_file, out_name);
        if node_ids.contains(&source) && node_ids.contains(&target) {
            edges.push(GraphEdge { source, target });
        }
    }

    let truncated = total_edges >= edge_limit || nodes.len() >= node_limit;
    Ok(CallGraph { nodes, edges, truncated })
}

/// Strip the stored `symbol:⟨fqn⟩` wrapper and return just the symbol name.
fn strip_symbol_ref(s: &str) -> Option<String> {
    s.strip_prefix("symbol:⟨")
        .and_then(|s| s.strip_suffix("⟩"))
        .map(|fqn| fqn.rsplit("::").next().unwrap_or(fqn).to_string())
}
