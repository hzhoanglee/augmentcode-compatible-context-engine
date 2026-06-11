# augment-ce-server

Augment Code SDK–compatible context API server, backed by a locally running
[vibervn-context-engine](../vibervn-context-engine/). Point the Augment context
SDK (`client-dist`) at this server and `DirectContext` works unmodified:
`addToIndex`, `waitForIndexing`, `search`, `removeFromIndex`.

## How it works

The Augment client uploads files as blobs (`sha256(path+content)` → path + content).
This server materializes them into a per-token workspace directory
(`DATA_DIR/workspaces/<sha256(token)[:16]>/files/`), registers that directory as a
repo with the context engine (which indexes it), and answers
`agents/codebase-retrieval` by calling the engine's `/api/mcp-tool` funnel.

| Augment endpoint | Behavior |
|---|---|
| `POST /find-missing` | Classifies blobs against workspace state + CE index status (`last_indexed_at` vs upload/trigger times) |
| `POST /batch-upload` | Writes files into the workspace, registers repo with CE, triggers indexing |
| `POST /checkpoint-blobs` | Applies deletions to disk, issues a checkpoint UUID |
| `POST /agents/codebase-retrieval` | Syncs disk to the client's active blob set (handles deletions), calls CE `/api/mcp-tool`, returns `formatted_retrieval` |
| `POST /chat-stream` | 501 (not implemented) |

Auth: any non-empty Bearer token is accepted; the token doubles as the workspace key.

## Run

```bash
# 1. Start the context engine (configure an embedding backend in its web UI at
#    http://127.0.0.1:6699 — either a Voyage AI key, or an OpenAI-compatible
#    /v1/embeddings endpoint such as LM Studio/GPUStack via provider "openai"
#    + base_url; an optional embedding reranker can be configured the same way)
./vibervn-context-engine/target/release/context-engine-rs --port 6699

# 2. Start this server
pip install fastapi uvicorn httpx
python -m uvicorn server.app:app --host 127.0.0.1 --port 8787
```

Config via env: `AUGMENT_CE_URL` (default `http://127.0.0.1:6699`), `HOST`
(`127.0.0.1`), `PORT` (`8787`), `DATA_DIR` (`~/.augment-ce`),
`CE_MAX_CONNECTIONS` (`200`, HTTP connection pool to the engine),
`RETRIEVE_CONCURRENCY` (`8`, retrievals in flight against the engine — excess
requests queue), `INDEX_DEBOUNCE_SECS` (`2.0`, coalescing window for engine
index triggers).

## Use from the Augment SDK

```bash
AUGMENT_API_TOKEN=anything AUGMENT_API_URL=http://127.0.0.1:8787 node your-script.mjs
```

```js
import { DirectContext } from "./client-dist/context/direct-context.js";
const ctx = await DirectContext.create({});
await ctx.addToIndex([{ path: "math.py", contents: "def add(a, b):\n    return a + b\n" }]);
console.log(await ctx.search("function that adds two numbers"));
```

End-to-end test: `node e2e/e2e.mjs` (with both servers running).

## Notes

- Single-process server (asyncio locks + per-workspace `state.json`); don't run
  multiple uvicorn workers.
- The engine indexes the materialized workspace copy, so retrieval results show
  workspace-absolute paths (`.../workspaces/<id>/files/<relpath>`); the relative
  part matches the paths the client uploaded.
- The client retries 499/503/5xx; this server returns 503 when the engine is
  unreachable, 502 for engine errors, 400/401 for terminal client errors, and
  499 when the client disconnects before the request completes.
