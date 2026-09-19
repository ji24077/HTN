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


def agent_image() -> str:
    """Which published image a new machine should run.

    Configurable because a fork publishes to its own registry, and a default that names
    somebody else's would send every new machine to an image its operator does not
    control.
    """
    return os.environ.get("DWP_AGENT_IMAGE", "").strip() or "ghcr.io/ji24077/dwp-agent:latest"


@router.get("/join", response_class=HTMLResponse)
async def join(request: Request):
    origin = request.app.state.config.public_origin or str(request.base_url).rstrip("/")
    # No invite codes are embedded in HTML or third-party URLs. The app reads its own link.
    safe_origin = html.escape(origin, quote=True)
    safe_image = html.escape(agent_image(), quote=True)
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
    # Whether to offer the button that mints a code, for a reader who arrived at /join
    # with nothing in the address bar. Off, the page keeps its old behaviour exactly:
    # a placeholder, and a reader who has to go and ask someone for a real invite.
    self_serve = "true" if request.app.state.config.self_serve_join else "false"
    return HTMLResponse(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Connect a device · Dispatch</title><style>body{font:17px/1.6 system-ui;max-width:680px;margin:64px auto;padding:24px;color:#242621;background:#f7f7f2}code{word-break:break-all}a{color:inherit}</style>"
        "<h1>Connect your device</h1>"
        # Docker first, and with the invite already in the command. Every other option on
        # this page asks the reader to fetch something and then find where to paste a
        # code; this one is a line they can paste into a terminal on any operating
        # system. The code is filled in by the script below rather than rendered here, so
        # it never enters this HTML, a proxy cache, or a server log.
        + "<h2>Run it in Docker (recommended)</h2>"
        + "<p>One command, the same on macOS, Windows and Linux. Nothing is compiled and no port needs opening \u2014 this machine dials out.</p>"
        # Minting is a step 1 that only some readers need, so it is a block that removes
        # itself rather than a section they have to know to skip. Someone who followed an
        # invite link already has a code and never sees this; someone who opened /join
        # cold sees it and nothing else until they have one.
        + '<div id="mintbox" hidden>'
        + '<p><label for="label">Name this machine</label><br>'
        + '<input id="label" type="text" maxlength="40" placeholder="ethans-laptop" autocomplete="off" spellcheck="false" style="font:inherit;padding:8px 10px;border-radius:8px;border:1px solid #c9c9bf;width:16em"></p>'
        + '<p><button id="mint" style="font:inherit;padding:10px 18px;border-radius:8px;border:1px solid #242621;background:#242621;color:#fff;cursor:pointer">Get my invite</button> <span id="mintnote"></span></p>'
        + "</div>"
        + '<div id="cmdbox" hidden>'
        + '<pre id="dockercmd" style="background:#ecece4;padding:16px;border-radius:8px;overflow-x:auto;white-space:pre-wrap"></pre>'
        + '<p><button id="copy" style="font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #242621;background:#fff;cursor:pointer">Copy</button> <span id="copied" hidden></span></p>'
        + "</div>"
        + "<p>The window is then at <code>http://127.0.0.1:43117/</code> on that machine. Keep the <code>-v</code> volume: it holds the machine\u2019s identity and its record of what it has run.</p>"
        + "<h2>Desktop or iOS app</h2><p>Open the desktop worker or iOS app and paste the invite link from your fleet dashboard. Invitations expire after ten minutes and work once.</p>"
        + downloads
        + "<h2>From the repository</h2><p>Install dependencies with <code>pnpm install --frozen-lockfile</code>, then open <code>pnpm agent gui</code> and paste your invite link.</p>"
        + f"<p>Server: <code>{safe_origin}</code></p><p>The iOS app must stay in the foreground to accept new work. Desktop downloads are unsigned at the operating-system level; application updates are verified against the release signing key.</p>"
        + '<p><a href="/">Back to your fleet</a></p>'
        # Read the code out of this page's own address. A pairing code is [0-9a-f]{32};
        # anything else is shown as a placeholder rather than pasted into a command line.
        #
        # Deliberately one long line rather than a backslash-continued block. Backslash
        # continuation is POSIX shell syntax: pasted into PowerShell, which is where a
        # Windows contributor will paste it, every line after the first is a separate
        # broken command. A single line wraps visually and runs everywhere.
        + "<script>(function(){"
        + "var pre=document.getElementById('dockercmd');"
        + "var cmdbox=document.getElementById('cmdbox');"
        + "var mintbox=document.getElementById('mintbox');"
        + "var cmd='';"
        # Build the command from a code rather than rendering one into the HTML. A minted
        # code arrives over fetch and a followed invite arrives in the address bar; both
        # end up here, so there is one place that decides what a command looks like.
        #
        # The label is the operator's free text and lands inside a double-quoted shell
        # argument, so it is restricted rather than escaped: an escape that is right for
        # sh is wrong for PowerShell, and this one string has to survive both. Anything
        # outside [A-Za-z0-9 ._-] is dropped, which cannot close the quote in either
        # shell. Empty after that means the flag is left out entirely.
        + "function build(c,label){"
        + "var safe=(label||'').replace(/[^A-Za-z0-9 ._-]/g,'').trim().slice(0,40);"
        + "var name=safe?' -e DWP_LABEL=\"'+safe+'\"':'';"
        + "cmd='docker run -d --restart unless-stopped"
        + " -v dwp-agent-data:/data -p 127.0.0.1:43117:43117'+name+'"
        + f" -e DWP_INVITE=\"{safe_origin}/join?code='+c+'\" {safe_image}';"
        + "pre.textContent=cmd;cmdbox.hidden=false;}"
        + "var c=new URLSearchParams(location.search).get('code')||'';"
        + "if(/^[0-9a-fA-F]{32}$/.test(c)){build(c,'');}"
        + f"else if({self_serve}){{mintbox.hidden=false;}}"
        + "else{build('YOUR-INVITE-CODE','');}"
        # Ask the server for a code. The button is disabled for the round trip because a
        # second click is a second code, and codes are rationed -- an impatient reader
        # double-clicking could spend the network's hourly allowance on one laptop.
        + "var mint=document.getElementById('mint');"
        + "if(mint){mint.onclick=function(){"
        + "var note=document.getElementById('mintnote');"
        + "mint.disabled=true;note.textContent='Asking the server\\u2026';"
        + "fetch('/v1/join-requests',{method:'POST',headers:{'Accept':'application/json'}})"
        + ".then(function(r){return r.json().then(function(b){return{ok:r.ok,body:b}})})"
        + ".then(function(res){"
        + "if(!res.ok){mint.disabled=false;"
        # The 429 detail is written for this reader; show it rather than a generic line.
        + "note.textContent=res.body.detail||'Could not get an invite. Try again.';return}"
        + "build(res.body.code,document.getElementById('label').value);"
        + "mintbox.hidden=true;"
        + "note.textContent='';"
        + "})"
        # Without this a dropped connection leaves the button disabled under 'Asking the
        # server...' forever, which reads as the server having hung rather than as
        # something to retry.
        + ".catch(function(){mint.disabled=false;"
        + "note.textContent='Could not reach the server. Check the connection and try again.'});"
        + "}}"
        # navigator.clipboard does not exist on an insecure origin, and writeText can be
        # refused even where it does. Both were silent: the button did nothing at all,
        # which is worse than not having one. Select the command instead so Ctrl-C still
        # works, and say which of the two happened.
        + "function done(t){var n=document.getElementById('copied');"
        + "n.textContent=t;n.hidden=false;setTimeout(function(){n.hidden=true},3000)}"
        + "function select(){try{var r=document.createRange();r.selectNodeContents(pre);"
        + "var s=getSelection();s.removeAllRanges();s.addRange(r);"
        + "done('Selected \u2014 press Ctrl-C (Cmd-C on a Mac) to copy.')}"
        + "catch(e){done('Select the command above and copy it.')}}"
        # Race the write against a timer. writeText does not merely fail when the page
        # is not the visible tab -- it never settles at all, so neither callback runs and
        # the button sits there having done nothing, with nothing in the console either.
        # Measured: pending after 1.5s with visibilityState 'hidden'.
        + "document.getElementById('copy').onclick=function(){"
        + "if(!navigator.clipboard||!navigator.clipboard.writeText){select();return}"
        + "var settled=false;"
        + "var giveUp=setTimeout(function(){if(!settled){settled=true;select()}},600);"
        + "navigator.clipboard.writeText(cmd).then(function(){"
        + "if(settled)return;settled=true;clearTimeout(giveUp);done('Copied.')},"
        + "function(){if(settled)return;settled=true;clearTimeout(giveUp);select()})};"
        + "})();</script></html>",
        headers={
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            # This page needs nothing from anywhere, and now carries one inline script
            # that reads the invite code out of the address bar. Say so explicitly rather
            # than leaving the page open to whatever a proxy or extension injects.
            #
            # connect-src 'self' is the one addition the mint button needs: it POSTs to
            # this same origin. default-src 'none' covers connect-src, so without this
            # the fetch is blocked by the policy and the button fails with nothing but a
            # console entry to say why.
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'unsafe-inline'; connect-src 'self'; "
                "script-src 'unsafe-inline'; form-action 'none'; base-uri 'none'"
            ),
        },
    )
