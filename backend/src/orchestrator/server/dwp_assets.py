"""Content-addressed workload artifacts and signed worker release distribution."""

import hashlib
import html
import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .auth import require_admin

router = APIRouter()
ROOT = Path(__file__).resolve().parents[4]
HASH = re.compile(r"[a-f0-9]{64}")


def directory(setting: str, default: str) -> Path:
    return Path(os.environ.get(setting, str(ROOT / default))).resolve()


def read_index(binary: bool = False) -> dict | None:
    root = directory("DWP_RELEASES_DIR", "releases")
    path = root / "binaries" / "index.json" if binary else root / "latest.json"
    try:
        if path.stat().st_size > 1024 * 1024:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("manifest"), dict):
            return None
        if not all(isinstance(data.get(k), str) for k in ("signature", "publicKey")):
            return None
        return data
    except (OSError, ValueError):
        return None


def release_info() -> dict:
    release = read_index() or read_index(True)
    return {
        "releaseKey": release["publicKey"] if release else None,
        "releaseVersion": release["manifest"].get("version") if release else None,
    }


async def require_device(request: Request) -> str:
    # Import lazily: enrollment also reads the release metadata above.
    from .dwp import authenticate_device

    try:
        return await authenticate_device(
            request.app.state.store, request.headers.get("authorization")
        )
    except ValueError:
        raise HTTPException(401, "Invalid device credentials") from None


def content_file(root: Path, name: str) -> Path:
    if not HASH.fullmatch(name):
        raise HTTPException(404, "Artifact not found")
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != name:
            raise HTTPException(503, "Artifact is corrupt")
    return path


@router.get("/release/latest")
async def latest_release():
    release = read_index()
    if not release:
        raise HTTPException(404, "No source release published")
    return JSONResponse(release, headers={"Cache-Control": "no-store"})


@router.get("/release/binaries")
async def binary_release():
    release = read_index(True)
    if not release:
        raise HTTPException(404, "No binary release published")
    return JSONResponse(release, headers={"Cache-Control": "no-store"})


@router.get("/release/{digest}", dependencies=[Depends(require_device)])
async def source_bundle(digest: str):
    return FileResponse(
        content_file(directory("DWP_RELEASES_DIR", "releases"), digest),
        media_type="application/gzip",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.get("/download/{name}")
async def binary_download(name: str):
    index = read_index(True)
    if not index:
        raise HTTPException(404, "No binary release published")
    # The two lists live in two directories, and the entry decides which. Serving both
    # out of "binaries" made every app zip a dead link on the /join page -- rendered from
    # the index, which lists them, then 404ed by the route, which looked in the one place
    # they are not. The raw executables worked throughout, so the failure looked like a
    # broken download rather than a server that could not find its own file.
    releases = directory("DWP_RELEASES_DIR", "releases")
    root = None
    for key, folder in (("binaries", "binaries"), ("apps", "apps")):
        if any(
            isinstance(item, dict) and item.get("file") == name for item in index.get(key, [])
        ):
            root = (releases / folder).resolve()
            break
    if root is None:
        raise HTTPException(404, "Download not found")
    path = (root / name).resolve()
    # Still anchored to one directory per entry, so a crafted name cannot walk out of it.
    if path.parent != root or not path.is_file():
        raise HTTPException(404, "Download not found")
    return FileResponse(path, filename=name, media_type="application/octet-stream")


@router.get("/artifacts/{digest}", dependencies=[Depends(require_device)])
async def artifact(digest: str):
    root = directory("DWP_FIXTURES_DIR", "fixtures")
    path = content_file(root, digest)
    return FileResponse(path, media_type="application/octet-stream")


@router.get("/v1/workloads", dependencies=[Depends(require_admin)])
async def workloads():
    root = directory("DWP_FIXTURES_DIR", "fixtures")
    inference = None
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        model, inputs = manifest["model"], manifest["inputs"]
        for record in (model, inputs):
            path = content_file(root, record["hash"])
            if path.stat().st_size != record["bytes"]:
                raise ValueError("Fixture size mismatch")
        inference = {
            "modelHash": model["hash"],
            "inputsHash": inputs["hash"],
            "inputName": model["inputName"],
            "outputName": model["outputName"],
            "from": 0,
            "count": min(100, inputs["count"]),
            "preprocessing": "v1",
        }
    except (OSError, ValueError, KeyError, TypeError, HTTPException):
        pass
    return {"inference": inference}


@router.get("/join", response_class=HTMLResponse)
async def join(request: Request):
    origin = request.app.state.config.public_origin or str(request.base_url).rstrip("/")
    # No invite codes are embedded in HTML or third-party URLs. The app reads its own link.
    safe_origin = html.escape(origin, quote=True)
    index = read_index(True)
    links = []
    if index:
        for entry in index.get("apps", []) + index.get("binaries", []):
            name = entry.get("file", "")
            if name and Path(name).name == name and re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
                links.append(
                    f'<li><a href="/download/{name}">{html.escape(entry.get("target", name))}</a></li>'
                )
    downloads = (
        "<ul>" + "".join(links) + "</ul>"
        if links
        else "<p>No desktop binaries have been published yet. Run the worker from the repository with Node 24 and pnpm.</p>"
    )
    return HTMLResponse(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Connect a device · Dispatch</title><style>body{font:17px/1.6 system-ui;max-width:680px;margin:64px auto;padding:24px;color:#242621;background:#f7f7f2}code{word-break:break-all}a{color:inherit}</style>"
        "<h1>Connect your device</h1><p>Open the desktop worker or iOS app and paste the invite link from your fleet dashboard. Invitations expire after ten minutes and work once.</p>"
        + downloads
        + "<h2>From the repository</h2><p>Install dependencies with <code>pnpm install --frozen-lockfile</code>, then open <code>pnpm agent gui</code> and paste your invite link.</p>"
        + f"<p>Server: <code>{safe_origin}</code></p><p>The iOS app must stay in the foreground to accept new work. Desktop downloads are unsigned at the operating-system level; application updates are verified against the release signing key.</p>"
        + '<p><a href="/">Back to your fleet</a></p></html>',
        headers={
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
