"""Bounded archive ingestion. Uploads are data; never import or execute on the server."""

import base64
import hashlib
import io
import stat
import zipfile
from pathlib import PurePosixPath

from ..shared.protocol import json_text
from .models import MAX_SOURCE, MAX_UPLOAD


def safe_path(name):
    path = PurePosixPath(name)
    if (
        not name
        or len(name) > 240
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or any(part.startswith("__dispatch_") for part in path.parts)
    ):
        raise ValueError("Project paths must be relative and cannot use reserved __dispatch_ names")
    return name


def unpack(files):
    result, names = {}, set()
    size = source_size = 0

    def add(name, content):
        nonlocal size, source_size
        safe_path(name)
        folded = name.casefold()
        if any(
            folded == old or folded.startswith(old + "/") or old.startswith(folded + "/")
            for old in names
        ):
            raise ValueError("Duplicate or conflicting project path")
        names.add(name.casefold())
        size += len(content)
        if len(result) >= 100 or size > MAX_UPLOAD:
            raise ValueError("Project exceeds 100 files or 8 MiB expanded")
        if name.endswith(".py"):
            source_size += len(content)
            if source_size > MAX_SOURCE:
                raise ValueError("Python sources exceed 128 KiB for this version")
            content.decode("utf-8")
        result[name] = base64.b64encode(content).decode()

    for file in files:
        raw = base64.b64decode(file.content, validate=True)
        if len(raw) > MAX_UPLOAD:
            raise ValueError("Upload exceeds 8 MiB")
        if file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if len(archive.infolist()) > 200:
                    raise ValueError("Archive contains too many entries")
                for info in archive.infolist():
                    safe_path(info.filename.rstrip("/"))
                    mode = info.external_attr >> 16
                    if stat.S_ISLNK(mode) or (
                        stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                    ):
                        raise ValueError("Archive links and special files are not supported")
                    if info.flag_bits & 1 or info.file_size > MAX_UPLOAD - size:
                        raise ValueError("Encrypted or oversized archive entry")
                    if info.is_dir():
                        continue
                    with archive.open(info) as stream:
                        content = stream.read(MAX_UPLOAD - size + 1)
                    add(info.filename, content)
        else:
            add(file.name, raw)
    if not any(name.endswith(".py") for name in result):
        raise ValueError("Upload a Python script or a ZIP project containing Python files")
    return result


def bundle(files):
    raw = json_text({"files": files}).encode()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Execution bundle exceeds 16 MiB")
    return hashlib.sha256(raw).hexdigest(), raw


def encoded(text):
    return base64.b64encode(text.encode()).decode()


def inspect_files(files):
    return {
        name: base64.b64decode(content).decode("utf-8")
        if name.endswith((".py", "requirements.txt", "pyproject.toml")) and len(content) < 180000
        else f"[data file: {len(base64.b64decode(content))} bytes]"
        for name, content in files.items()
    }
