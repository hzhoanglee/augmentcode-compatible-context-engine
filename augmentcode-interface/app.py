"""Augment Code SDK-compatible context API, backed by vibervn-context-engine.

Single-process server (asyncio locks + per-workspace JSON state). Run:
    python -m uvicorn server.app:app --port 8787
or  python -m server.app
"""

import asyncio
import logging
import re
import time
import uuid
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

import ce_client
import state
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("augment-ce")

MAX_FILE_SIZE_BYTES = 1024 * 1024
MAX_OUTPUT_LENGTH_CAP = 80_000


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await ce_client.startup()
    yield
    await ce_client.shutdown()


app = FastAPI(title="augment-ce-server", lifespan=lifespan)


# ─── Error mapping ──────────────────────────────────────────────────────────

@app.exception_handler(ce_client.CEUnavailable)
async def _ce_unavailable(_req: Request, exc: ce_client.CEUnavailable):
    return JSONResponse(
        status_code=503, content={"error": str(exc)}, headers={"Retry-After": "2"}
    )


@app.exception_handler(ce_client.CEError)
async def _ce_error(_req: Request, exc: ce_client.CEError):
    return JSONResponse(status_code=502, content={"error": str(exc)})


@app.exception_handler(state.PathViolation)
async def _path_violation(_req: Request, exc: state.PathViolation):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(state.UnknownCheckpoint)
async def _unknown_checkpoint(_req: Request, exc: state.UnknownCheckpoint):
    return JSONResponse(status_code=400, content={"error": f"unknown checkpoint_id: {exc}"})


@app.exception_handler(ClientDisconnect)
async def _client_disconnect(req: Request, _exc: ClientDisconnect):
    # The client gave up (timeout/retry/cancel) before we read the body.
    # Nobody is listening, so any response works — return 499 quietly instead
    # of letting the exception bubble up as a logged 500.
    logger.info("client disconnected before request completed: %s", req.url.path)
    return Response(status_code=499)


# ─── Request logging ────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    logger.info(
        "%s %s -> %d (%.0fms) req_id=%s session=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.monotonic() - started) * 1000,
        request.headers.get("X-Request-Id", "-"),
        request.headers.get("X-Request-Session-Id", "-"),
    )
    return response


# ─── Auth ───────────────────────────────────────────────────────────────────

class Workspace:
    def __init__(self, ws_state: state.WorkspaceState, lock, token: str):
        self.state = ws_state
        self.lock = lock
        self.token = token


async def workspace(request: Request) -> Workspace:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not token:
        raise MissingToken()
    ws_state, ws_id = state.get_state(token)
    return Workspace(ws_state, await state.get_lock(ws_id), token)


class MissingToken(Exception):
    pass


@app.exception_handler(MissingToken)
async def _missing_token(_req: Request, _exc: MissingToken):
    return JSONResponse(status_code=401, content={"error": "missing bearer token"})


# ─── Schemas ────────────────────────────────────────────────────────────────

class FindMissingRequest(BaseModel):
    mem_object_names: list[str] = Field(default_factory=list)


class UploadBlob(BaseModel):
    blob_name: str
    path: str
    content: str


class BatchUploadRequest(BaseModel):
    blobs: list[UploadBlob]


class BlobsDelta(BaseModel):
    checkpoint_id: str | None = None
    added_blobs: list[str] = Field(default_factory=list)
    deleted_blobs: list[str] = Field(default_factory=list)


class CheckpointRequest(BaseModel):
    blobs: BlobsDelta


class RetrievalRequest(BaseModel):
    information_request: str
    blobs: BlobsDelta = Field(default_factory=BlobsDelta)
    dialog: list = Field(default_factory=list)
    max_output_length: int | None = None


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_rfc3339(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


async def _ensure_registered(ws: Workspace) -> None:
    if not ws.state.ce_registered:
        async with state.ce_registration_lock:
            await ce_client.ensure_registered(ws.state.files_dir)
        ws.state.ce_registered = True


async def _register_and_trigger(ws: Workspace) -> None:
    await _ensure_registered(ws)
    await ce_client.trigger_index(ws.state.files_dir)
    # Hold the workspace lock for the threaded save: json.dumps in a worker
    # thread must not race concurrent dict mutations on the event loop.
    async with ws.lock:
        ws.state.mark_index_triggered()
        await asyncio.to_thread(ws.state.save)


# ─── Debounced index triggers ───────────────────────────────────────────────
# A burst of batch-uploads schedules ONE CE trigger per workspace instead of
# one per request: files are already on disk when the pending trigger fires,
# so the coalesced run covers them all. The dirty set catches writes that land
# while a trigger is mid-flight — the worker loops until the workspace is clean.

_pending_triggers: dict[str, asyncio.Task] = {}
_trigger_dirty: set[str] = set()


def schedule_index_trigger(ws: Workspace) -> None:
    ws_id = ws.state.ws_id
    _trigger_dirty.add(ws_id)
    existing = _pending_triggers.get(ws_id)
    if existing is not None and not existing.done():
        return
    _pending_triggers[ws_id] = asyncio.create_task(_fire_trigger(ws))


async def _fire_trigger(ws: Workspace) -> None:
    ws_id = ws.state.ws_id
    try:
        while ws_id in _trigger_dirty:
            await asyncio.sleep(CONFIG.index_debounce_secs)
            # Clear AFTER the sleep: writes landing during the debounce window
            # are already on disk and covered by this trigger. Writes landing
            # during the trigger await re-mark dirty and loop once more.
            _trigger_dirty.discard(ws_id)
            await _register_and_trigger(ws)
    except Exception:
        _trigger_dirty.discard(ws_id)
        logger.exception("background index trigger failed for workspace %s", ws_id)
    finally:
        _pending_triggers.pop(ws_id, None)


def _unlink_files(ws: Workspace, rel_paths: list[str]) -> None:
    for rel in rel_paths:
        target = ws.state.files_dir / rel
        target.unlink(missing_ok=True)
        # prune now-empty parent dirs up to files_dir
        parent = target.parent
        while parent != ws.state.files_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


def _format_for_client(result: str, files_dir) -> str:
    """Rewrite CE block headers ("<abs path>#L10-20 [tags]") into the Augment
    wire format ("Path: <rel path>"), which the auggie CLI parses with
    /^Path:\\s+(.+)$/gm to render its "Found N files" line."""
    prefix = str(files_dir).replace("\\", "/").rstrip("/") + "/"
    header_re = re.compile(
        rf"^{re.escape(prefix)}(?P<file>[^\n#]+)#L(?P<start>\d+)-(?P<end>\d+)(?P<tags>[^\n]*)$",
        re.MULTILINE,
    )

    def _repl(m: re.Match) -> str:
        return (
            f"Path: {m.group('file')}\n"
            f"Lines: {m.group('start')}-{m.group('end')}{m.group('tags')}"
        )

    result = header_re.sub(_repl, result)
    # Scrub any remaining absolute workspace paths (e.g. in caller-file tags).
    return result.replace(prefix, "")


def _log_response(endpoint: str, token: str, data: dict) -> None:
    # logs_dir = Path(__file__).parent / "logs"
    # logs_dir.mkdir(exist_ok=True)
    # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # log_file = logs_dir / f"{ts}_{endpoint}.txt"
    # log_file.write_text(f"Token: {token[:16]}...\n{json.dumps(data, indent=2)}\n")
    pass


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/find-missing")
async def find_missing(req: FindMissingRequest, ws: Workspace = Depends(workspace)):
    async with ws.lock:
        known, unknown = ws.state.classify(req.mem_object_names)
        last_trigger = ws.state.last_index_trigger_at
        uploaded_at = {name: ws.state.blobs[name]["uploaded_at"] for name in known}

    if not known:
        result = {"unknown_memory_names": unknown, "nonindexed_blob_names": []}
        _log_response("find-missing", ws.token, result)
        return result

    status = await ce_client.get_status(ws.state.files_dir)
    if status is None:
        # Repo not registered with CE yet — self-heal.
        await _register_and_trigger(ws)
        result = {"unknown_memory_names": unknown, "nonindexed_blob_names": known}
        _log_response("find-missing", ws.token, result)
        return result

    ce_state = (status.get("state") or "").lower()
    if ce_state == "error":
        raise ce_client.CEError(f"indexing failed: {status.get('error')}")

    indexed_at = _parse_rfc3339(status.get("last_indexed_at"))
    if ce_state != "idle" or indexed_at is None:
        result = {"unknown_memory_names": unknown, "nonindexed_blob_names": known}
        _log_response("find-missing", ws.token, result)
        return result

    # A blob is indexed only if an index run completed after both its upload
    # and our latest trigger — a run that started before the write can finish
    # after it without containing the file.
    nonindexed = [
        name for name in known
        if indexed_at < uploaded_at[name] or indexed_at < last_trigger
    ]
    result = {"unknown_memory_names": unknown, "nonindexed_blob_names": nonindexed}
    _log_response("find-missing", ws.token, result)
    return result


@app.post("/batch-upload")
async def batch_upload(req: BatchUploadRequest, ws: Workspace = Depends(workspace)):
    now = time.time()
    async with ws.lock:
        # Validate everything before writing anything.
        targets = []
        for blob in req.blobs:
            if len(blob.content.encode()) > MAX_FILE_SIZE_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"file too large: {blob.path}"},
                )
            targets.append((blob, state.safe_relpath(blob.path, ws.state.files_dir)))

        def _write_all() -> None:
            for blob, target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(blob.content, encoding="utf-8")

        # Disk writes run in a worker thread so a large batch doesn't stall
        # the event loop (and every other in-flight request) for its duration.
        await asyncio.to_thread(_write_all)
        for blob, _target in targets:
            rel = blob.path.replace("\\", "/")
            ws.state.record_upload(blob.blob_name, rel, now)
        await asyncio.to_thread(ws.state.save)

    await _ensure_registered(ws)
    schedule_index_trigger(ws)
    return {"blob_names": [b.blob_name for b in req.blobs]}


@app.post("/checkpoint-blobs")
async def checkpoint_blobs(req: CheckpointRequest, ws: Workspace = Depends(workspace)):
    async with ws.lock:
        to_unlink = ws.state.apply_deletions(req.blobs.deleted_blobs)
        if to_unlink:
            await asyncio.to_thread(_unlink_files, ws, to_unlink)
        new_id = ws.state.create_checkpoint(
            req.blobs.checkpoint_id, req.blobs.added_blobs, req.blobs.deleted_blobs
        )
        ws.state.save()

    if to_unlink:
        schedule_index_trigger(ws)
    return {"new_checkpoint_id": new_id}


@app.post("/agents/codebase-retrieval")
async def codebase_retrieval(req: RetrievalRequest, ws: Workspace = Depends(workspace)):
    if not req.information_request.strip():
        return JSONResponse(status_code=400, content={"error": "empty information_request"})

    logger.info("[codebase-retrieval] request: %s", req.information_request[:500])
    _log_response("codebase-retrieval-request", ws.token, {
        "information_request": req.information_request,
        "added_blobs": len(req.blobs.added_blobs),
        "deleted_blobs": len(req.blobs.deleted_blobs),
        "checkpoint_id": req.blobs.checkpoint_id,
        "max_output_length": req.max_output_length,
    })

    # The client batches deletions (CHECKPOINT_THRESHOLD=1000), so removals
    # usually arrive here rather than via /checkpoint-blobs — and a blob that
    # was never checkpointed simply disappears from added_blobs without ever
    # being listed in deleted_blobs. The request's resolved blob set is the
    # authoritative active set: sync disk to it.
    deleted_any = False
    async with ws.lock:
        ws.state.apply_deletions(req.blobs.deleted_blobs)
        active = ws.state.resolve_blob_set(
            req.blobs.checkpoint_id, req.blobs.added_blobs, req.blobs.deleted_blobs
        )
        to_unlink = ws.state.sync_to_blob_set(active)
        if to_unlink:
            await asyncio.to_thread(_unlink_files, ws, to_unlink)
            deleted_any = True
        await asyncio.to_thread(ws.state.save)

    if deleted_any:
        schedule_index_trigger(ws)

    result = await ce_client.retrieve(req.information_request, ws.state.files_dir)
    result = _format_for_client(result, ws.state.files_dir)

    logger.info("[codebase-retrieval] CE response length: %d", len(result))
    _log_response("codebase-retrieval-response", ws.token, {
        "information_request": req.information_request,
        "ce_response": result,
    })

    if req.max_output_length:
        limit = max(1, min(req.max_output_length, MAX_OUTPUT_LENGTH_CAP))
        result = result[:limit]
    return {"formatted_retrieval": result}


@app.post("/chat-stream")
async def chat_stream():
    return JSONResponse(status_code=501, content={"error": "chat-stream not implemented"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)
