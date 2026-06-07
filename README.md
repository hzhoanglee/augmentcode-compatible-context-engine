# vibervn-context-engine

## Install & Run

Run the latest release directly with npx — no manual download, the correct
prebuilt binary for your platform is fetched automatically. The `@latest`
tag forces npx to fetch the newest published version instead of reusing a
stale cached one:

```bash
npx vibervn-context-engine@latest
```

This boots the HTTP server on port 6699 (web UI at http://127.0.0.1:6699,
MCP endpoint at `/mcp`). Any CLI flags are forwarded to the binary:

```bash
npx vibervn-context-engine@latest --port 8080 --bind 0.0.0.0
```

Or install it globally to get a persistent `vibervn-context-engine` command:

```bash
npm install -g vibervn-context-engine@latest
vibervn-context-engine --port 6699
```

Supported platforms: Linux x64/arm64, macOS arm64, Windows x64.

## Features

| Feature | Description |
|---------|-------------|
| Semantic code search | Finds code by meaning via embeddings, not literal text matching |
| Multi-language parsing | Tree-sitter symbol extraction for Python, JavaScript, TypeScript, Rust, Go, Java, C, and C++ |
| Call-graph expansion | Resolves caller/callee edges and BFS-expands matched symbols at query time |
| Incremental indexing | Re-indexes only changed files (mtime + watcher), crash-safe via per-file commit markers |
| Real-time file watching | `notify` (debounced) triggers re-index automatically on file changes |
| Voyage AI embeddings | HTTP embedding client with an on-disk cache to avoid redundant API calls |
| LLM reranking | Reorders candidate chunks with an LLM (OpenAI / Google); optional, can be disabled |
| Embedded SurrealDB | Stores chunks, symbols, and edges; one datastore per repo |
| HTTP API + Web UI | Settings management, index explorer, and a query test console |
| MCP server | Exposes a single `codebase-retrieval` tool over streamable HTTP |
| SSE progress stream | Streams live indexing progress events to the UI |
| Large-repo scaling | Bounded memory and no O(n²) paths — built for Linux/Chromium-scale codebases |

## Supported Languages

Tree-sitter symbol extraction (functions, classes, methods, and call edges) is
implemented per language. File extensions are mapped in `detect_language`
(`src/parsing/mod.rs`).

| Language | Extensions | Grammar |
|----------|------------|---------|
| Python | `.py` | `tree-sitter-python` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | `tree-sitter-javascript` |
| TypeScript | `.ts` | `tree-sitter-typescript` |
| TSX | `.tsx` | `tree-sitter-javascript` |
| Rust | `.rs` | `tree-sitter-rust` |
| Go | `.go` | `tree-sitter-go` |
| Java | `.java` | `tree-sitter-java` |
| C | `.c` | `tree-sitter-c` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hxx`, `.hh` | `tree-sitter-cpp` |

Files with any other extension are chunked and embedded for semantic search,
but no symbols or call edges are extracted from them.

## How It Works

```mermaid
flowchart TD
    Boot([Server boot]) --> Engine[IndexEngine starts]
    Engine --> Watchers[Spawn per-repo file watchers]
    Engine --> LoadVec[Load vector index from SurrealDB]

    Watchers -->|file change / manual / MCP| Trigger[Index trigger]
    Trigger --> Detect[Walk repo + detect changed files]
    Detect --> Parse[Parse: tree-sitter symbols, chunks, raw edges]
    Parse --> Embed[Embed chunks: Voyage AI + on-disk cache]
    Embed --> Store[Store chunks/symbols in SurrealDB]
    Store --> Phase2[Phase 2: resolve raw edges into call graph]
    Phase2 --> VecIndex[(In-memory vector index)]
    LoadVec --> VecIndex

    subgraph Clients
        WebUI[Web UI]
        MCP[MCP tool: codebase-retrieval]
    end

    Clients --> Q1[Embed query]
    Q1 --> Q2[Vector search: top-k cosine]
    Q2 -.reads.-> VecIndex
    Q2 --> Q3[Graph expand: BFS callers/callees]
    Q3 --> Q4[Merge + dedup adjacent ranges]
    Q4 --> Q5[LLM rerank]
    Q5 --> Q6[Format: path#Lstart-end + numbered lines]
    Q6 --> Result([Results])
```
