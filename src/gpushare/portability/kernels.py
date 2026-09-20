"""Conservative CUDA/HIP source rewriting, not a compiler or universal translator.

Supported: explicit core runtime API/type/enum pairs, runtime headers, ordinary
kernel qualifiers/indexing, CUDA launch syntax (also accepted by HIP-Clang), and
HIP's hipLaunchKernelGGL macro. Unsupported vendor APIs, architecture intrinsics,
assembly, and GPU SDK headers produce blockers. Comments and literals are never
rewritten except a literal that is an actual #include header. Preprocessor
branches are not evaluated; blocked constructs in inactive branches still block.

Reference: https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/how-to/hip_porting_guide.html
Output requires compilation and numerical tests on its target backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RUNTIME_SUFFIXES = (
    "Malloc", "Free", "Memcpy", "MemcpyAsync", "Memset", "MemsetAsync",
    "MemGetInfo", "GetDevice", "SetDevice", "GetDeviceCount", "GetDeviceProperties",
    "DeviceSynchronize", "DeviceReset", "GetLastError", "PeekAtLastError",
    "GetErrorString", "GetErrorName", "StreamCreate", "StreamCreateWithFlags",
    "StreamDestroy", "StreamSynchronize", "StreamQuery", "StreamWaitEvent",
    "EventCreate", "EventCreateWithFlags", "EventRecord", "EventSynchronize",
    "EventDestroy", "EventQuery", "EventElapsedTime", "RuntimeGetVersion",
    "DriverGetVersion", "Error_t", "Stream_t", "Event_t", "MemcpyKind",
    "MemcpyHostToHost", "MemcpyHostToDevice", "MemcpyDeviceToHost",
    "MemcpyDeviceToDevice", "MemcpyDefault", "Success", "ErrorInvalidValue",
    "ErrorInvalidDevice", "ErrorInvalidDeviceFunction", "ErrorNotReady",
    "ErrorUnknown", "StreamDefault", "StreamNonBlocking", "EventDefault",
    "EventBlockingSync", "EventDisableTiming", "EventInterprocess",
)
_CUDA_TO_HIP = {"cuda" + suffix: "hip" + suffix for suffix in _RUNTIME_SUFFIXES}
_CUDA_TO_HIP.update({
    "cudaDeviceProp": "hipDeviceProp_t",
    "cudaErrorMemoryAllocation": "hipErrorOutOfMemory",
})
_HEADERS = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
}
_VENDOR_IDENTIFIER = re.compile(
    r"^(?:cuda[A-Z_]|cu[A-Z]|hip[A-Z_]|cublas|cufft|curand|cusolver|cusparse|"
    r"nvrtc|nvtx|nccl|hipblas|hipfft|hiprand|hipsolver|hipsparse|rocblas|rocfft|"
    r"rocrand|rocsolver|rocsparse|rccl)"
)
_ARCH_IDENTIFIER = re.compile(
    r"^(?:__shfl|__ballot|__activemask|__syncwarp|__match_|__reduce_|__ldg|"
    r"__nv|__half|__bfloat|__builtin_(?:amdgcn|nvptx)|__CUDA|__CUDACC|"
    r"__HIP|__AMDGCN|__launch_bounds__|warpSize$)"
)
_GPU_HEADER = re.compile(
    r"(?:cuda|hip|cublas|cufft|curand|cusolver|cusparse|nvrtc|nccl|rccl|"
    r"rocblas|rocfft|rocrand|rocsolver|rocsparse|(?:^|/)cub(?:/|\.)|"
    r"(?:^|/)thrust/|cooperative_groups)", re.IGNORECASE
)
_RAW_START = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
_QUOTED = re.compile(r'''(?:u8|u|U|L)?(?P<q>["'])(?:\\[\s\S]|(?!\1)[^\\])*?(?P=q)''')
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_HOST_EFFECT_IDENTIFIERS = {
    "system", "popen", "_popen", "exec", "execl", "execle", "execlp", "execv",
    "execve", "execvp", "execvpe", "fork", "vfork", "posix_spawn", "posix_spawnp",
    "dlopen", "dlsym", "LoadLibrary", "LoadLibraryA", "LoadLibraryW", "GetProcAddress",
    "socket", "connect", "bind", "listen", "accept", "send", "sendto", "recv",
    "recvfrom", "curl_easy_perform", "fopen", "freopen", "fwrite", "open", "openat",
    "creat", "write", "pwrite", "unlink", "unlinkat", "remove", "rename", "chmod",
    "chown", "mkdir", "rmdir", "symlink", "getenv", "setenv", "putenv", "unsetenv",
    "filesystem", "fstream", "ifstream", "ofstream", "syscall", "mprotect", "mmap",
    "__attribute__",  # In particular constructors can execute before the C ABI.
}
_HOST_EFFECT_HEADER = re.compile(
    r"^(?:/|[A-Za-z]:|\.\.|sys/|net/|netinet/|arpa/|curl/|"
    r"(?:unistd|dlfcn|spawn|fcntl|windows|process)\.h$|(?:filesystem|fstream)$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int


def _lex(source: str) -> list[_Token]:
    tokens = []
    pos = 0
    while pos < len(source):
        start = pos
        if source[pos].isspace():
            pos += 1
            while pos < len(source) and source[pos].isspace():
                pos += 1
            kind = "space"
        elif source.startswith("//", pos):
            pos += 2
            while pos < len(source):
                if source.startswith("\\\r\n", pos):
                    pos += 3
                elif source.startswith("\\\n", pos):
                    pos += 2
                elif source[pos] == "\n":
                    break
                else:
                    pos += 1
            kind = "comment"
        elif source.startswith("/*", pos):
            end = source.find("*/", pos + 2)
            pos = len(source) if end < 0 else end + 2
            kind = "invalid" if end < 0 else "comment"
        elif raw := _RAW_START.match(source, pos):
            end = source.find(")" + raw.group(1) + '"', raw.end())
            pos = len(source) if end < 0 else end + len(raw.group(1)) + 2
            kind = "invalid" if end < 0 else "literal"
        elif quoted := _QUOTED.match(source, pos):
            pos = quoted.end()
            # Ordinary C++ strings/chars cannot contain a physical newline unless
            # it is escaped. Raw strings are handled by the earlier branch.
            physical = re.sub(r"\\(?:\r\n|\n)", "", quoted.group())
            kind = "invalid" if "\n" in physical or "\r" in physical else "literal"
        elif identifier := _IDENTIFIER.match(source, pos):
            pos = identifier.end()
            kind = "id"
        else:
            pos += 1
            kind = "invalid" if source[start] in "\"'" else "punct"
        tokens.append(_Token(kind, source[start:pos], start, pos))
    return tokens


def _arguments(source: str, tokens: list[_Token], opening: int):
    """Split macro arguments only at top-level commas, retaining their source text."""
    stack = [")"]
    arguments = []
    start = tokens[opening].end
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index in range(opening + 1, len(tokens)):
        token = tokens[index]
        if token.kind != "punct":
            continue
        if token.text in pairs:
            stack.append(pairs[token.text])
        elif token.text in ")]}":
            if not stack or stack.pop() != token.text:
                return None
            if not stack:
                arguments.append(_trim_fragment(source[start:token.start]))
                return arguments, index
        elif token.text == "," and len(stack) == 1:
            arguments.append(_trim_fragment(source[start:token.start]))
            start = token.end
    return None


def _trim_fragment(fragment: str) -> str:
    fragment = fragment.strip()
    tokens = _lex(fragment)
    if tokens and tokens[-1].kind == "comment" and tokens[-1].text.startswith("//"):
        fragment += "\n"  # Do not let a moved comma/launch delimiter become comment text.
    return fragment


def _kernel_name(argument: str) -> str | None:
    """Accept identifiers and template names; reject arbitrary function expressions."""
    significant = [t for t in _lex(argument) if t.kind not in {"space", "comment"}]
    if not significant:
        return None
    if significant[0].text == "HIP_KERNEL_NAME":
        if len(significant) < 4 or significant[1].text != "(" or significant[-1].text != ")":
            return None
        argument = _trim_fragment(
            argument[:significant[0].start]
            + argument[significant[0].end:significant[1].start]
            + argument[significant[1].end:significant[-1].start]
            + argument[significant[-1].end:]
        )
        significant = [t for t in _lex(argument) if t.kind not in {"space", "comment"}]
    while significant and significant[0].text == "(" and significant[-1].text == ")":
        argument = _trim_fragment(
            argument[:significant[0].start]
            + argument[significant[0].end:significant[-1].start]
            + argument[significant[-1].end:]
        )
        significant = [t for t in _lex(argument) if t.kind not in {"space", "comment"}]
    compact = "".join(t.text for t in significant)
    if not re.fullmatch(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:<.+>)?", compact):
        return None
    if any(t.kind != "id" and t.text not in "<>:,*&0123456789" for t in significant):
        return None
    # Template commas are legal here only because HIP_KERNEL_NAME or surrounding
    # parentheses protected them from the outer launch macro's argument splitter.
    depth = 0
    for token in significant:
        if token.text == "<":
            depth += 1
        elif token.text == ">":
            depth -= 1
        if depth < 0:
            return None
    return argument if depth == 0 else None


def convert_kernel(
    source: str, source_backend: str, target_backend: str, filename: str = "kernel.cu"
) -> dict:
    """Rewrite an explicit subset and report blockers; never silently certify output.

    ``mappings`` records from/to/line for actual edits. A blocker means the output
    is a partial draft and must not be built/deployed automatically. Line numbers
    refer to the original source. Same-backend input is audited without edits.
    """
    if source_backend not in {"cuda", "hip"} or target_backend not in {"cuda", "hip"}:
        raise ValueError("source_backend and target_backend must be cuda or hip")
    result = {"source": source, "changed": False, "mappings": [], "findings": []}
    mapping = _CUDA_TO_HIP if source_backend == "cuda" else {v: k for k, v in _CUDA_TO_HIP.items()}
    headers = _HEADERS if source_backend == "cuda" else {v: k for k, v in _HEADERS.items()}
    if source_backend == target_backend:
        mapping = {key: key for key in mapping}
        headers = {key: key for key in headers}
    tokens = _lex(source)
    edits = []
    excluded = set()

    def finding(token, code, message, severity="blocker"):
        result["findings"].append({"severity": severity, "code": code, "path": filename,
                                   "line": source.count("\n", 0, token.start) + 1,
                                   "message": message})

    def edit(start, end, replacement):
        old = source[start:end]
        if old == replacement:
            return
        edits.append((start, end, replacement))
        result["mappings"].append({"from": old, "to": replacement,
                                    "line": source.count("\n", 0, start) + 1})

    significant = [i for i, token in enumerate(tokens) if token.kind not in {"space", "comment"}]
    for position, index in enumerate(significant):
        token = tokens[index]
        if token.kind == "punct" and token.text == "\\" and source[token.end:token.end + 2].startswith(("\n", "\r\n")):
            finding(token, "preprocessor_line_splice", "Code line splicing can hide GPU identifiers and requires preprocessing before conversion.")
        if token.text == "#" and position + 1 < len(significant) and tokens[significant[position + 1]].text == "#":
            finding(token, "preprocessor_token_paste", "Token-pasting macros can synthesize GPU APIs and require preprocessing before conversion.")
    for position, index in enumerate(significant):
        token = tokens[index]
        if token.text != "#" or position + 2 >= len(significant):
            continue
        if tokens[significant[position + 1]].text != "include":
            continue
        begin = significant[position + 2]
        header_token = tokens[begin]
        end = begin
        if header_token.kind == "literal" and header_token.text.startswith('"'):
            header = header_token.text[1:-1]
            start_offset, end_offset = header_token.start + 1, header_token.end - 1
        elif header_token.text == "<":
            while end < len(tokens) and tokens[end].text != ">" and "\n" not in tokens[end].text:
                end += 1
            if end == len(tokens) or tokens[end].text != ">":
                finding(header_token, "malformed_include", "Cannot parse include header safely.")
                continue
            start_offset, end_offset = header_token.end, tokens[end].start
            header = source[start_offset:end_offset]
        else:
            finding(header_token, "dynamic_include", "Macro-based includes require manual backend review.")
            continue
        excluded.update(range(begin, end + 1))
        if header in headers:
            edit(start_offset, end_offset, headers[header])
        elif _GPU_HEADER.search(header):
            finding(header_token, "unsupported_gpu_header", f"GPU header {header!r} is outside the supported runtime-header subset.")
        elif _HOST_EFFECT_HEADER.search(header):
            finding(header_token, "native_host_effect_header", f"Host process/filesystem/network header {header!r} is outside the GPU computation contract.")

    # Launch edits precede token edits. Rewriting a whole launch recursively applies
    # core runtime mappings to its arguments without touching literals/comments.
    if source_backend == "hip":
        for index, token in enumerate(tokens):
            if index in excluded or token.kind != "id" or token.text != "hipLaunchKernelGGL":
                continue
            opening = index + 1
            while opening < len(tokens) and tokens[opening].kind in {"space", "comment"}:
                opening += 1
            parsed = _arguments(source, tokens, opening) if opening < len(tokens) and tokens[opening].text == "(" else None
            if parsed is None or len(parsed[0]) < 5:
                finding(token, "unsupported_kernel_launch", "hipLaunchKernelGGL requires a balanced kernel/grid/block/shared/stream argument list.")
                continue
            args, closing = parsed
            kernel = _kernel_name(args[0])
            if kernel is None or any(not arg for arg in args[:5]):
                finding(token, "unsupported_kernel_launch", "Kernel expression or launch arguments cannot be converted safely; wrap template names in HIP_KERNEL_NAME.")
                continue
            # Vendor constructs inside arguments are checked by the ordinary scan
            # below. Only known token replacements are applied here.
            def mapped_fragment(fragment):
                return "".join(mapping.get(t.text, t.text) if t.kind == "id" else t.text for t in _lex(fragment))

            prefix = source[token.end:tokens[opening].start]
            replacement = prefix + mapped_fragment(kernel) + "<<<" + ", ".join(mapped_fragment(a) for a in args[1:5]) + ">>>(" + ", ".join(mapped_fragment(a) for a in args[5:]) + ")"
            if source_backend != target_backend:
                edit(token.start, tokens[closing].end, replacement)
            excluded.update(range(index, closing + 1))
            # Don't silently accept unsupported intrinsics/APIs in a converted launch.
            for inner in tokens[index + 1:closing]:
                if inner.kind == "invalid":
                    finding(inner, "invalid_lexical_input", "Invalid literal/comment inside the launch cannot be converted safely.")
                if inner.kind == "id" and inner.text in {"asm", "__asm", "__asm__"}:
                    finding(inner, "inline_assembly", "Inline assembly inside a launch argument requires manual implementation.")
                if inner.kind == "id" and inner.text in _HOST_EFFECT_IDENTIFIERS:
                    finding(inner, "native_host_effect", f"Host-effect identifier {inner.text} is outside the GPU computation contract.")
                if inner.kind == "id" and inner.text not in mapping and inner.text != "HIP_KERNEL_NAME":
                    if _VENDOR_IDENTIFIER.match(inner.text) or _ARCH_IDENTIFIER.match(inner.text):
                        finding(inner, "unsupported_launch_argument", f"{inner.text} requires manual conversion inside this launch.")

    for index, token in enumerate(tokens):
        if index in excluded:
            continue
        if token.kind == "invalid":
            finding(token, "invalid_lexical_input", "Unterminated comment or literal prevents safe source conversion.")
        if token.kind != "id":
            continue
        if token.text in mapping:
            edit(token.start, token.end, mapping[token.text])
        elif token.text in _HOST_EFFECT_IDENTIFIERS:
            finding(token, "native_host_effect", f"Host process/filesystem/network identifier {token.text} is outside the GPU computation contract.")
        elif token.text in {"asm", "__asm", "__asm__"}:
            finding(token, "inline_assembly", "Inline assembly/PTX is architecture-specific and requires manual implementation.")
        elif _ARCH_IDENTIFIER.match(token.text):
            finding(token, "architecture_specific", f"{token.text} has backend/warp/architecture semantics outside the supported subset.")
        elif _VENDOR_IDENTIFIER.match(token.text) or token.text in {"cuda", "hip", "cub", "thrust", "cooperative_groups", "HIP_KERNEL_NAME", "HIP_SYMBOL", "CUDA_VERSION", "HIP_VERSION"}:
            finding(token, "unsupported_gpu_identifier", f"{token.text} is not an explicitly supported runtime mapping.")
    converted = source
    for start, end, replacement in sorted(edits, reverse=True):
        converted = converted[:start] + replacement + converted[end:]
    residual = re.compile(r"^(?:cuda[A-Z_]|cu[A-Z])" if target_backend == "hip" else r"^(?:hip[A-Z_]|HIP_KERNEL_NAME$|HIP_SYMBOL$)")
    reported = set()
    for token in _lex(converted):
        if token.kind == "id" and residual.match(token.text) and token.text not in reported:
            original = next((t for t in tokens if t.kind == "id" and t.text == token.text), token)
            finding(original, "unresolved_source_identifier", f"Source-backend identifier {token.text} remains in generated code.")
            reported.add(token.text)
    result["source"] = converted
    result["changed"] = converted != source
    if result["changed"]:
        finding(_Token("", "", 0, 0), "target_validation_required",
                "Source subset rewritten; target compiler and numerical equivalence tests are still required.", "warning")
    return result
