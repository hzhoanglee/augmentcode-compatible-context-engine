"""Per-workspace state: uploaded blobs, path ownership, checkpoints.

One JSON file per workspace (state.json), guarded by a per-workspace
asyncio.Lock and written atomically. Single-process server assumed.
"""

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from config import CONFIG

MAX_CHECKPOINTS = 5

_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()
# CE settings registration is a read-modify-write of the full settings
# object, so it serializes globally across workspaces.
ce_registration_lock = asyncio.Lock()

_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:")


class PathViolation(ValueError):
    pass


class UnknownCheckpoint(ValueError):
    pass


def workspace_id(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


async def get_lock(ws_id: str) -> asyncio.Lock:
    async with _locks_guard:
        if ws_id not in _locks:
            _locks[ws_id] = asyncio.Lock()
        return _locks[ws_id]


def safe_relpath(path: str, files_dir: Path) -> Path:
    """Validate a client-supplied relative path and return the absolute target."""
    if not path or "\x00" in path:
        raise PathViolation(f"invalid path: {path!r}")
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_DRIVE.match(normalized):
        raise PathViolation(f"absolute path not allowed: {path!r}")
    if any(seg in ("..", "") for seg in normalized.split("/")):
        raise PathViolation(f"path traversal not allowed: {path!r}")
    target = (files_dir / normalized).resolve()
    if not target.is_relative_to(files_dir.resolve()):
        raise PathViolation(f"path escapes workspace: {path!r}")
    return target


class WorkspaceState:
    def __init__(self, ws_id: str):
        self.ws_id = ws_id
        self.root = CONFIG.workspaces_dir / ws_id
        self.files_dir = self.root / "files"
        self.state_path = self.root / "state.json"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
        else:
            data = {}
        self.blobs: dict[str, dict] = data.get("blobs", {})
        self.paths: dict[str, str] = data.get("paths", {})
        self.checkpoints: dict[str, list[str]] = data.get("checkpoints", {})
        self.checkpoint_order: list[str] = data.get("checkpoint_order", list(self.checkpoints))
        self.last_index_trigger_at: float = data.get("last_index_trigger_at", 0.0)
        self.ce_registered: bool = data.get("ce_registered", False)

    def save(self) -> None:
        data = {
            "version": 1,
            "blobs": self.blobs,
            "paths": self.paths,
            "checkpoints": self.checkpoints,
            "checkpoint_order": self.checkpoint_order,
            "last_index_trigger_at": self.last_index_trigger_at,
            "ce_registered": self.ce_registered,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.state_path)

    def record_upload(self, blob_name: str, rel_path: str, now: float) -> None:
        # Ownership transfer: the previous blob for this path stays in
        # history (the client deletes it explicitly later), but the path
        # now points at the new blob.
        self.blobs[blob_name] = {"path": rel_path, "uploaded_at": now, "deleted": False}
        self.paths[rel_path] = blob_name

    def classify(self, blob_names: list[str]) -> tuple[list[str], list[str]]:
        """Return (known, unknown). Deleted blobs count as unknown (forces re-upload)."""
        known, unknown = [], []
        for name in blob_names:
            entry = self.blobs.get(name)
            if entry is None or entry.get("deleted"):
                unknown.append(name)
            else:
                known.append(name)
        return known, unknown

    def apply_deletions(self, deleted_blobs: list[str]) -> list[str]:
        """Mark blobs deleted; return rel paths whose files must be unlinked
        (only when the blob is still the current owner of its path)."""
        to_unlink = []
        for name in deleted_blobs:
            entry = self.blobs.get(name)
            if entry is None or entry.get("deleted"):
                continue
            entry["deleted"] = True
            rel_path = entry["path"]
            if self.paths.get(rel_path) == name:
                del self.paths[rel_path]
                to_unlink.append(rel_path)
        return to_unlink

    def create_checkpoint(self, base_id: str | None, added: list[str], deleted: list[str]) -> str:
        new_set = self.resolve_blob_set(base_id, added, deleted)
        new_id = str(uuid.uuid4())
        self.checkpoints[new_id] = sorted(new_set)
        self.checkpoint_order.append(new_id)
        while len(self.checkpoint_order) > MAX_CHECKPOINTS:
            old = self.checkpoint_order.pop(0)
            self.checkpoints.pop(old, None)
        return new_id

    def resolve_blob_set(self, base_id: str | None, added: list[str], deleted: list[str]) -> set[str]:
        if base_id is not None:
            if base_id not in self.checkpoints:
                raise UnknownCheckpoint(base_id)
            base = set(self.checkpoints[base_id])
        else:
            base = set()
        return (base | set(added)) - set(deleted)

    def sync_to_blob_set(self, active: set[str]) -> list[str]:
        """Drop path ownership for blobs outside the client's active set; return
        rel paths to unlink. Needed because a never-checkpointed blob removed via
        removeFromIndex simply vanishes from added_blobs without ever being sent
        in deleted_blobs."""
        to_unlink = []
        for rel_path, blob_name in list(self.paths.items()):
            if blob_name not in active:
                entry = self.blobs.get(blob_name)
                if entry is not None:
                    entry["deleted"] = True
                del self.paths[rel_path]
                to_unlink.append(rel_path)
        return to_unlink

    def mark_index_triggered(self) -> None:
        self.last_index_trigger_at = time.time()


_states: dict[str, WorkspaceState] = {}


def get_state(token: str) -> tuple[WorkspaceState, str]:
    ws_id = workspace_id(token)
    if ws_id not in _states:
        _states[ws_id] = WorkspaceState(ws_id)
    return _states[ws_id], ws_id
