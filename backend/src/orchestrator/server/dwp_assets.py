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
        "<title>Connect a device · Dispatch</title><style>"
        "body{font:17px/1.6 system-ui;max-width:680px;margin:64px auto;padding:24px;color:#242621;background:#f7f7f2}"
        "code{word-break:break-all}a{color:inherit}"
        "details{border-top:1px solid #dcdcd2;padding:10px 0}"
        "details summary{cursor:pointer;font-weight:600;font-size:16px}"
        "details p{margin:8px 0 0}"
        "details pre{background:#ecece4;padding:12px;border-radius:8px;overflow-x:auto;white-space:pre-wrap;font-size:14px}"
        "</style>"
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
        + "<p>If that port is already taken on the machine \u2014 another agent is the usual "
        + "reason \u2014 change the <em>first</em> number only, and make "
        + "<code>DWP_GUI_PUBLIC_ORIGIN</code> match: "
        + "<code>-p 127.0.0.1:43118:43117 -e DWP_GUI_PUBLIC_ORIGIN=\"http://127.0.0.1:43118\"</code></p>"
        # Updating is not joining, and the difference is the part people get wrong: no
        # invite, and above all the same volume. A reader who improvises this reaches for
        # the join command they already have, which mints nothing and works fine -- until
        # they leave the volume out, at which point the machine silently becomes a new
        # one with no history and no saved limits, and the old identity is orphaned.
        + "<h2>Already connected? Update it</h2>"
        + "<p>A new version is a new image. Nothing is rebuilt on the machine: it pulls the "
        + "same bytes every other machine runs. Keep the <code>-v</code> volume and it stays "
        + "the same machine \u2014 same identity, same history, same limits \u2014 so no new "
        + "invite is needed.</p>"
        # Clear the port by *port*, not by container name.
        #
        # `docker rm -f dwp-agent` only matches a container someone happened to name that.
        # Whatever is actually holding 43117 may be an older container under a different
        # name, and then the run below fails with "Ports are not available: address
        # already in use" -- which is where this stops being an update and becomes a
        # debugging session. Measured: that is exactly how the first machine to join
        # failed. `--filter publish` asks Docker what owns the port and removes that.
        #
        # -r matters and is not decoration: GNU xargs with empty input runs the command
        # anyway, so a machine with nothing on the port would see a bare `docker rm -f`
        # and its "requires at least 1 argument" error, which reads as the reset having
        # gone wrong. macOS accepts -r and already behaves this way.
        + f'<pre id="updatecmd" style="background:#ecece4;padding:16px;border-radius:8px;overflow-x:auto;white-space:pre-wrap">docker pull {safe_image}\ndocker ps -aq --filter publish=43117 | xargs -r docker rm -f\ndocker run -d --name dwp-agent --restart unless-stopped -v dwp-agent-data:/data -p 127.0.0.1:43117:43117 -e DWP_GUI_PUBLIC_ORIGIN="http://127.0.0.1:43117" {safe_image}</pre>'
        + '<p><button id="copyupdate" style="font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #242621;background:#fff;cursor:pointer">Copy</button> <span id="copiedupdate" hidden></span></p>'
        + "<p>The middle line stops whatever is holding that port, whatever the container "
        + "is called \u2014 an older agent under a different name is the usual reason a "
        + "straight update fails. Your data is in the volume, not the container, so "
        + "removing it loses nothing.</p>"
        # PowerShell has no xargs, and the POSIX line above fails there with a message
        # about a command it has never heard of. Two shells, two lines, said plainly.
        + "<p><strong>On Windows, in PowerShell</strong>, the middle line is instead:</p>"
        + '<pre id="updatecmdwin" style="background:#ecece4;padding:16px;border-radius:8px;overflow-x:auto;white-space:pre-wrap">docker ps -aq --filter publish=43117 | ForEach-Object { docker rm -f $_ }</pre>'
        + '<p><button id="copyupdatewin" style="font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #242621;background:#fff;cursor:pointer">Copy</button> <span id="copiedupdatewin" hidden></span></p>'
        + "<p>Separate commands, so they paste into any shell and run in order. Work in "
        + "flight is handed back to the network rather than lost, so this is safe to run on "
        + "a machine that is busy.</p>"
        # Every entry here is something that actually happened while bringing machines
        # onto this network, in the words the reader will see. A troubleshooting list
        # written from imagination covers the failures that are easy to think of rather
        # than the ones people hit, and the two sets barely overlap.
        + "<h2>Didn\u2019t work?</h2>"
        + "<p>The error you got is the heading.</p>"

        + "<details><summary>Ports are not available: address already in use</summary>"
        + "<p>Something already holds 43117 on that machine \u2014 usually an agent you "
        + "installed earlier, under a different name. Either free the port with the middle "
        + "line of the update command above, or leave it alone and put this machine on "
        + "another port by changing the <em>first</em> number and making "
        + "<code>DWP_GUI_PUBLIC_ORIGIN</code> match:</p>"
        + "<pre>-p 127.0.0.1:43118:43117 -e DWP_GUI_PUBLIC_ORIGIN=\"http://127.0.0.1:43118\"</pre>"
        + "<p>Two agents on one machine is fine. Each has its own identity, and both do work.</p>"
        + "</details>"

        + "<details><summary>The container name &quot;/dwp-agent&quot; is already in use</summary>"
        + "<p>An earlier attempt left a container behind, stopped. Remove it and run the "
        + "command again \u2014 this keeps the volume, so nothing is lost:</p>"
        + "<pre>docker rm -f dwp-agent</pre></details>"

        + "<details><summary>That invite was not accepted</summary>"
        + "<p>Invites last ten minutes and work exactly once, so the usual cause is that it "
        + "sat in a chat window too long, or the container was started twice with it. Get "
        + "another from the top of this page \u2014 nothing needs undoing first.</p>"
        + "<p>If a machine has already joined, it does not need an invite at all: its "
        + "identity is in the volume. Use the update command instead.</p></details>"

        + "<details><summary>The window shows &quot;not found&quot;, or nothing at all</summary>"
        + "<p>Each agent has its own address and its own token, so a token from one will not "
        + "open another. Ask the agent you mean for its address:</p>"
        + "<pre>docker logs dwp-agent | grep Window</pre>"
        + "<p>On Windows, <code>docker logs dwp-agent | findstr Window</code>.</p>"
        + "<p>If it prints a port you did not publish, that is the container\u2019s own port, "
        + "not the host\u2019s. Use the first number from your <code>-p</code> flag and keep "
        + "the token unchanged. Setting <code>DWP_GUI_PUBLIC_ORIGIN</code>, as the commands "
        + "above do, makes it print the right one.</p></details>"

        + "<details><summary>How do I know it is actually working?</summary>"
        + "<pre>docker logs dwp-agent</pre>"
        + "<p>You want <code>enrol.joined</code> followed by <code>connect.established</code>. "
        + "That second line means it is on the network and can be given work. "
        + "<code>connect.retry_scheduled</code> over and over means it cannot reach the "
        + "server \u2014 check the machine has internet, and that the address in the command "
        + "is the one this page shows.</p>"
        + "<p>The window\u2019s Status tab says the same thing in words, and Recent work "
        + "lists what this machine has actually run.</p></details>"

        + "<details><summary>It is connected but never does anything</summary>"
        + "<p>Open the window and look at Status. If this machine is turning work down, it "
        + "says so and why \u2014 a limit you set on the Limits tab, or Pause. If it says it "
        + "is waiting for work, the network simply has none to give right now.</p></details>"

        + "<details><summary>docker: denied, or manifest unknown</summary>"
        + "<p>The image name is wrong, or the machine cannot reach the registry. Copy the "
        + "command from this page rather than typing it \u2014 it carries the image this "
        + "network actually publishes.</p></details>"

        + "<details><summary>I want this machine off the network</summary>"
        + "<p>Stopping the container is enough, and it keeps everything:</p>"
        + "<pre>docker rm -f dwp-agent</pre>"
        + "<p>To forget the network as well, remove the volume. That discards the machine\u2019s "
        + "identity and its history, and rejoining will need a new invite:</p>"
        + "<pre>docker volume rm dwp-agent-data</pre></details>"

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
        + "cmd='docker run -d --name dwp-agent --restart unless-stopped"
        + " -v dwp-agent-data:/data -p 127.0.0.1:43117:43117'+name+'"
        # A container cannot discover the host port it was published on, so it is told.
        # Without this it prints its own 43117 and anyone who remapped the port is sent
        # to an address that answers with somebody else's agent, or nothing at all.
        + " -e DWP_GUI_PUBLIC_ORIGIN=\"http://127.0.0.1:43117\""
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
        #
        # Written once and bound to each block, because there are two commands on this
        # page now and a second copy of this logic is a second place for the clipboard
        # quirks below to be got wrong.
        + "function wireCopy(preId,btnId,noteId){"
        + "var block=document.getElementById(preId);"
        + "var btn=document.getElementById(btnId);"
        + "var note=document.getElementById(noteId);"
        + "if(!block||!btn)return;"
        + "function done(t){note.textContent=t;note.hidden=false;"
        + "setTimeout(function(){note.hidden=true},3000)}"
        + "function select(){try{var r=document.createRange();r.selectNodeContents(block);"
        + "var s=getSelection();s.removeAllRanges();s.addRange(r);"
        + "done('Selected \u2014 press Ctrl-C (Cmd-C on a Mac) to copy.')}"
        + "catch(e){done('Select the command above and copy it.')}}"
        # Race the write against a timer. writeText does not merely fail when the page
        # is not the visible tab -- it never settles at all, so neither callback runs and
        # the button sits there having done nothing, with nothing in the console either.
        # Measured: pending after 1.5s with visibilityState 'hidden'.
        + "btn.onclick=function(){"
        + "var text=block.textContent;"
        + "if(!navigator.clipboard||!navigator.clipboard.writeText){select();return}"
        + "var settled=false;"
        + "var giveUp=setTimeout(function(){if(!settled){settled=true;select()}},600);"
        + "navigator.clipboard.writeText(text).then(function(){"
        + "if(settled)return;settled=true;clearTimeout(giveUp);done('Copied.')},"
        + "function(){if(settled)return;settled=true;clearTimeout(giveUp);select()})};"
        + "}"
        + "wireCopy('dockercmd','copy','copied');"
        + "wireCopy('updatecmd','copyupdate','copiedupdate');"
        + "wireCopy('updatecmdwin','copyupdatewin','copiedupdatewin');"
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
