"""Static conversion of one embedded native source string in a Python script.

These helpers never import or execute user code. Native compiler, numerical, and
GPU execution checks remain required; static guards cannot prove equivalence.
"""

from __future__ import annotations

import ast
import copy
import re

from .kernels import _lex, convert_kernel

_NAMES = {"CUDA_SOURCE", "HIP_SOURCE"}
_TARGETS = {"nvidia": "cuda", "amd": "hip"}


def _finding(code: str, message: str, severity: str = "blocker", line: int = 1) -> dict:
    return {"severity": severity, "code": code, "path": "script.py", "line": line,
            "message": message}


def _extract(source: str) -> tuple[ast.Module | None, ast.Constant | None, str | None, list]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return None, None, None, [_finding("invalid_python", str(exc))]
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            matching = _NAMES.intersection(names)
            if matching:
                assignments.append((node, next(iter(matching)), len(node.targets) == 1))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in _NAMES:
                assignments.append((node, node.target.id, True))
    if not assignments:
        hidden = any(isinstance(n, ast.Name) and n.id in _NAMES and isinstance(n.ctx, ast.Store)
                     for n in ast.walk(tree))
        if hidden:
            return tree, None, None, [_finding("dynamic_native_source", "Native source must be one top-level literal assignment.")]
        return tree, None, None, [_finding("no_embedded_native_source", "No top-level CUDA_SOURCE or HIP_SOURCE literal found; no native code conversion was performed.", "note")]
    if len(assignments) != 1:
        return tree, None, None, [_finding("ambiguous_native_source", "Exactly one top-level CUDA_SOURCE or HIP_SOURCE is supported.")]
    node, name, single_target = assignments[0]
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in _NAMES and isinstance(n.ctx, (ast.Store, ast.Del))]
    if not single_target or len(stores) != 1:
        return tree, None, None, [_finding("ambiguous_native_source", "Native source aliases or reassignment require project-level review.", line=node.lineno)]
    value = node.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return tree, None, None, [_finding("dynamic_native_source", "Native source must be a literal string; formatted, computed, and loaded source is unsupported.", line=node.lineno)]
    return tree, value, name, []


def _backend(kernel: str, variable: str) -> tuple[str | None, list]:
    markers = set()
    tokens = [token for token in _lex(kernel) if token.kind not in {"space", "comment"}]
    for index, token in enumerate(tokens):
        if token.kind == "id":
            if re.match(r"^(?:cuda[A-Z_]|cu[A-Z]|cublas|cufft|curand|cusolver|cusparse|nvrtc|__CUDA|__CUDACC)", token.text):
                markers.add("cuda")
            if re.match(r"^(?:hip[A-Z_/]|hipblas|hipfft|hiprand|rocblas|rocfft|__HIP|__AMDGCN)", token.text):
                markers.add("hip")
        if token.kind == "literal" and index >= 2 and tokens[index - 1].text == "include" and tokens[index - 2].text == "#":
            if token.text.strip('"').startswith("cuda"):
                markers.add("cuda")
            if token.text.strip('"').startswith("hip/"):
                markers.add("hip")
    if len(markers) > 1:
        return None, [_finding("mixed_native_backends", "Both CUDA and HIP identifiers appear in native code; choose an explicit source project before conversion.")]
    if markers:
        return markers.pop(), []
    backend = "cuda" if variable == "CUDA_SOURCE" else "hip"
    return backend, [_finding("backend_from_variable", "No backend-specific runtime identifiers found; source backend inferred from the literal's variable name.", "note")]


def _offset(source: str, line: int, byte_column: int) -> int:
    # Python AST columns are UTF-8 byte offsets, even when the caller supplies str.
    lines = source.splitlines(keepends=True)
    prefix = "".join(lines[:line - 1])
    column = len(lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8"))
    return len(prefix) + column


def _literal(value: str) -> str:
    # Keep a full demo readable without risking raw-string quote/ending escapes.
    if value.endswith("\n") and '"""' not in value:
        candidate = 'r"""' + value + '"""'
        try:
            if ast.literal_eval(candidate) == value:
                return candidate
        except (SyntaxError, ValueError):
            pass
    return repr(value)


def get_native_source(source: str) -> str:
    """Read the sole native literal without executing or importing the script."""
    _, node, _, findings = _extract(source)
    if node is None:
        raise ValueError("; ".join(item["message"] for item in findings))
    return node.value


def replace_native_source(source: str, native: str) -> str:
    """Replace exactly the native literal span, including correct UTF-8 offsets."""
    if not isinstance(native, str):
        raise TypeError("native must be a string")
    _, node, _, findings = _extract(source)
    if node is None:
        raise ValueError("; ".join(item["message"] for item in findings))
    start = _offset(source, node.lineno, node.col_offset)
    end = _offset(source, node.end_lineno, node.end_col_offset)
    return source[:start] + _literal(native) + source[end:]


def translate_script(source: str, target: str) -> dict:
    """Convert one native literal, retaining every byte outside its literal span.

    ``target`` is ``nvidia`` or ``amd``. PyTorch's ``torch.cuda`` interface stays
    unchanged on ROCm. Blockers describe unsupported input or partial drafts;
    this function does not certify compiler support or GPU execution.
    """
    if target not in _TARGETS:
        raise ValueError("target must be nvidia or amd")
    result = {"source": source, "changed": False, "findings": [], "mappings": [],
              "source_backend": None, "target_backend": _TARGETS[target]}
    _, node, variable, findings = _extract(source)
    result["findings"].extend(findings)
    if node is None:
        return result
    backend, findings = _backend(node.value, variable)
    result["source_backend"] = backend
    result["findings"].extend(findings)
    if backend is None:
        return result
    if backend == _TARGETS[target]:
        checked = convert_kernel(node.value, backend, backend, "script.py")
        result["findings"].extend({**item, "line": node.lineno + item["line"] - 1}
                                  for item in checked["findings"])
        result["findings"].append(_finding("already_target_backend", "Native source already uses the requested backend; no conversion performed.", "note"))
        return result
    converted = convert_kernel(node.value, backend, _TARGETS[target], "script.py")
    for collection in ("findings", "mappings"):
        result[collection].extend({**entry, "line": node.lineno + entry["line"] - 1}
                                  for entry in converted[collection])
    if converted["changed"]:
        result["source"] = replace_native_source(source, converted["source"])
        result["changed"] = True
    else:
        result["findings"].append(_finding("no_native_edits", "No supported native mappings were applied; this is not evidence of a code translation.", "note"))
    return result


def _protected_ast(tree: ast.Module, node: ast.Constant) -> str:
    cloned = copy.deepcopy(tree)
    for candidate in ast.walk(cloned):
        if isinstance(candidate, ast.Constant) and candidate.lineno == node.lineno and candidate.col_offset == node.col_offset:
            candidate.value = "<editable native source>"
            candidate.kind = None
            break
    return ast.dump(cloned, include_attributes=False)


def _persistent_declarations(native: str) -> list[tuple[str, int]]:
    """Find obvious storage declarations for this small, stateless demo subset.

    This is a brace/statement scanner, not C++ ownership or control-flow analysis.
    A new global object or local static requires review even if it could be safe;
    the fixed host-buffer ABI has no lifecycle hook for cached device allocations.
    """
    tokens = []
    directive = False
    for token in _lex(native):
        if token.kind == "punct" and token.text == "#":
            directive = True
        if not directive and token.kind not in {"space", "comment"}:
            tokens.append(token)
        if "\n" in token.text:
            directive = False
    declarations = []
    scopes = ["global"]
    statement = 0
    parentheses = 0

    def record(header):
        if not header:
            return
        words = {token.text for token in header if token.kind == "id"}
        persistent = bool(words & {"static", "thread_local", "__thread"})
        global_data = scopes[-1] in {"global", "namespace"}
        # Namespace aliases/type aliases have no allocation ownership themselves.
        if words & {"using", "typedef", "namespace"}:
            return
        if persistent or global_data:
            declarations.append((" ".join(token.text for token in header), header[0].start))

    for index, token in enumerate(tokens):
        if token.text == "(":
            parentheses += 1
        elif token.text == ")":
            parentheses = max(0, parentheses - 1)
        elif token.text == "{" and parentheses == 0:
            header = tokens[statement:index]
            words = {item.text for item in header if item.kind == "id"}
            values = [item.text for item in header]
            if "namespace" in words:
                scope = "namespace"
            elif words & {"struct", "class", "union", "enum"} and "=" not in values:
                if words & {"static", "thread_local", "__thread"}:
                    record(header)
                scope = "type"
            elif "(" in values and "=" not in values and scopes[-1] in {"global", "namespace", "type"}:
                scope = "function"
            else:
                # Includes aggregate initialization of a persistent object.
                record(header)
                scope = "block"
            scopes.append(scope)
            statement = index + 1
        elif token.text == "}" and parentheses == 0:
            if len(scopes) > 1:
                scopes.pop()
            statement = index + 1
        elif token.text == ";" and parentheses == 0:
            record(tokens[statement:index])
            statement = index + 1
    return declarations


def validate_demo_edit(original: str, candidate: str, phase: str) -> list[dict]:
    """Protect the demo harness and reject obvious native GPU/precision removal.

    Only its embedded source may change in optimization/translation/repair.
    These are static guardrails, not a C++ verifier: device traces and fixed
    numerical tests must still judge the compiled candidate on each real GPU.
    """
    original_tree, original_node, original_name, original_findings = _extract(original)
    candidate_tree, candidate_node, candidate_name, candidate_findings = _extract(candidate)
    findings = original_findings + candidate_findings
    if original_node is None or candidate_node is None:
        findings.append(_finding("missing_demo_native_source", "Both demo versions require exactly one embedded native source literal."))
        return findings
    if original_name != candidate_name or _protected_ast(original_tree, original_node) != _protected_ast(candidate_tree, candidate_node):
        findings.append(_finding("protected_python_changed", f"{phase}: edits changed the protected Python harness, inputs, precision, oracle, training loop, or build contract."))
    original_tokens = [t for t in _lex(original_node.value) if t.kind not in {"space", "comment", "literal"}]
    candidate_tokens = [t for t in _lex(candidate_node.value) if t.kind not in {"space", "comment", "literal"}]
    original_ids = {t.text for t in original_tokens if t.kind == "id"}
    candidate_ids = {t.text for t in candidate_tokens if t.kind == "id"}
    compact = "".join(t.text for t in candidate_tokens)
    for name in ("portable_transform", "portable_last_error", "__global__"):
        if name in original_ids and name not in candidate_ids:
            findings.append(_finding("native_gpu_contract_removed", f"{phase}: required native entry point or device qualifier {name} was removed."))
    if "__global__" in original_ids and not ("<<<" in compact or "hipLaunchKernelGGL" in candidate_ids):
        findings.append(_finding("native_gpu_launch_removed", "The custom GPU launch was removed; CPU substitution is not accepted."))
    if "__global__" in original_ids and not ({"cudaMemcpy", "hipMemcpy", "cudaMemcpyAsync", "hipMemcpyAsync"} & candidate_ids):
        findings.append(_finding("native_device_transfer_removed", "The demo's host-buffer ABI requires explicit device transfers; changed memory contracts need a separate reviewed fixture."))
    precision = {"double", "half", "__half", "half2", "__half2", "bfloat16", "__nv_bfloat16", "__hip_bfloat16", "_Float16"}
    if introduced := (candidate_ids - original_ids) & precision:
        findings.append(_finding("native_precision_changed", f"Preserve the demo's float32 arithmetic; new precision types: {', '.join(sorted(introduced))}."))
    if {"portable_transform", "portable_last_error", "__global__"} <= original_ids:
        original_storage = {declaration for declaration, _ in _persistent_declarations(original_node.value)}
        for declaration, offset in _persistent_declarations(candidate_node.value):
            if declaration not in original_storage:
                findings.append(_finding(
                    "persistent_native_storage",
                    f"{phase}: added static/thread-local/global storage requires a separate lifetime contract. This stateless C ABI has no destroy hook: release owned GPU buffers within every invocation; do not cache device allocations in persistent pointers or objects. Declaration: {declaration[:180]}",
                    line=candidate_node.lineno + candidate_node.value.count("\n", 0, offset),
                ))
    backend, backend_findings = _backend(candidate_node.value, candidate_name)
    findings.extend(backend_findings)
    if backend is not None:
        scan = convert_kernel(candidate_node.value, backend, backend)
        findings.extend(item for item in scan["findings"] if item["severity"] == "blocker")
    findings.append(_finding("runtime_validation_required", "Static guards cannot prove native equivalence or actual device execution; fixed compiler, GPU-use, numerical, and timing checks remain mandatory.", "warning"))
    return findings
