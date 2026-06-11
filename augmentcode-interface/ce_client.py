"""Async client for the vibervn-context-engine HTTP API."""

import base64
import logging
from pathlib import Path

import httpx

from config import CONFIG

logger = logging.getLogger("augment-ce.ce")

# CE's /api/mcp-tool may internally wait mcp_index_wait_secs (default 50s)
# plus rerank/LLM time, so retrieval needs a generous read timeout.
RETRIEVE_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

client: httpx.AsyncClient | None = None


class CEUnavailable(Exception):
    """CE is unreachable or transiently failing — map to HTTP 503."""


class CEError(Exception):
    """CE rejected the request or reported an error — map to HTTP 502."""


def normalize_repo_path(path: str) -> str:
    # Mirror of store::normalize_repo_path (unix branch).
    return path.replace("\\", "/").rstrip("/")


def repo_id(files_dir: Path) -> str:
    normalized = normalize_repo_path(str(files_dir))
    return base64.urlsafe_b64encode(normalized.encode()).decode().rstrip("=")


async def startup() -> None:
    global client
    client = httpx.AsyncClient(base_url=CONFIG.ce_url, timeout=DEFAULT_TIMEOUT)


async def shutdown() -> None:
    if client is not None:
        await client.aclose()


async def _request(method: str, url: str, **kwargs) -> httpx.Response:
    assert client is not None, "ce_client.startup() not called"
    try:
        resp = await client.request(method, url, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
        raise CEUnavailable(f"context engine unreachable: {e}") from e
    if resp.status_code >= 500:
        raise CEUnavailable(f"context engine error {resp.status_code}: {resp.text[:500]}")
    return resp


async def get_status(files_dir: Path) -> dict | None:
    resp = await _request("GET", f"/api/repos/{repo_id(files_dir)}/status")
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise CEError(f"status check failed ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


async def ensure_registered(files_dir: Path) -> None:
    """Add the workspace files dir to CE's repo list if absent.

    Caller must hold state.ce_registration_lock — PUT /api/config replaces
    the full settings object, so concurrent registrations would lose updates.
    """
    repo = normalize_repo_path(str(files_dir))
    resp = await _request("GET", "/api/config")
    if resp.status_code >= 400:
        raise CEError(f"failed to read CE config ({resp.status_code}): {resp.text[:500]}")
    settings = resp.json()
    if repo in (settings.get("repos") or []):
        return
    settings.setdefault("repos", []).append(repo)
    resp = await _request("PUT", "/api/config", json=settings)
    if resp.status_code >= 400:
        raise CEError(f"failed to register repo with CE ({resp.status_code}): {resp.text[:500]}")
    logger.info("registered workspace with CE: %s", repo)


async def trigger_index(files_dir: Path) -> None:
    resp = await _request("POST", f"/api/repos/{repo_id(files_dir)}/index")
    if resp.status_code >= 400:
        logger.warning("trigger_index failed (%s): %s", resp.status_code, resp.text[:200])


async def retrieve(information_request: str, files_dir: Path) -> str:
    resp = await _request(
        "POST",
        "/api/mcp-tool",
        json={
            "information_request": information_request,
            "workspace_full_path": normalize_repo_path(str(files_dir)),
        },
        timeout=RETRIEVE_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise CEError(f"retrieval failed ({resp.status_code}): {resp.text[:500]}")
    return resp.json().get("result", "")
