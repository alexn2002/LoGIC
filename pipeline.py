from __future__ import annotations

import ast
import cmath
import json
import math
import operator
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import interferometer as itf
from scipy.io import mmread, mmwrite

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None

from devices import (
    GaussianDevice,
    embedded_reck_mode_count,
    transform_instructions,
    validate_covariance,
)


def _extract_decomposition(decomp_obj):
    """Handle both tuple and object returns from the interferometer package."""
    if isinstance(decomp_obj, tuple) and len(decomp_obj) == 2:
        return decomp_obj

    bs_local = (
        getattr(decomp_obj, "bs_list", None)
        or getattr(decomp_obj, "BS_list", None)
        or getattr(decomp_obj, "bs", None)
    )
    phases_local = (
        getattr(decomp_obj, "phases", None)
        or getattr(decomp_obj, "output_phases", None)
        or getattr(decomp_obj, "output_phase", None)
    )
    if bs_local is None or phases_local is None:
        raise TypeError("Unexpected return type from interferometer decomposition.")
    return bs_local, phases_local


def _normalize_instructions(bs_params, n_modes: int):
    """Convert interferometer beamsplitters to internal 0-based instructions."""
    instructions = []
    for entry in bs_params:
        # Tuple-like returns may already be 0-based or 1-based depending on the
        # package path, so infer the convention from the values.
        if isinstance(entry, tuple):
            if len(entry) == 4:
                k, l, theta, phi = entry
            elif len(entry) == 3:
                k, l, theta = entry
                phi = 0.0
            else:
                raise ValueError(f"Unexpected beamsplitter tuple: {entry}")

            k_int = int(k)
            l_int = int(l)
            if k_int < 0 or l_int < 0:
                raise ValueError(f"Negative mode index in beamsplitter tuple: {entry}")
            if k_int >= n_modes or l_int >= n_modes:
                # 1-based tuples use the range 1..n.
                k_int -= 1
                l_int -= 1
        else:
            # Beamsplitter objects from interferometer use 1-based mode labels.
            k = getattr(entry, "k", getattr(entry, "i", getattr(entry, "mode1", None)))
            l = getattr(entry, "l", getattr(entry, "j", getattr(entry, "mode2", None)))
            theta = getattr(entry, "theta", getattr(entry, "angle", None))
            phi = getattr(entry, "phi", getattr(entry, "phase", 0.0))
            if k is None or l is None or theta is None:
                raise TypeError(f"Unexpected beamsplitter object: {entry!r}")

            k_int = int(k) - 1
            l_int = int(l) - 1

        if not (0 <= k_int < n_modes and 0 <= l_int < n_modes):
            raise ValueError(
                f"Beam splitter indices {(k_int, l_int)} are out of range for {n_modes} modes."
            )

        instructions.append((k_int, l_int, float(theta), float(phi or 0.0)))
    return instructions


def instructions_from_U(
    U: np.ndarray,
    topology: str,
    embedded_total_modes: int | None = None,
) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    """Decompose a unitary into beamsplitter-network instructions and output phases."""
    U = np.asarray(U, dtype=complex)
    n_modes = U.shape[0]

    topo = topology.lower()
    if topo == "clements":
        decomp_fn = itf.square_decomposition
    elif topo == "reck":
        decomp_fn = itf.triangle_decomposition
    elif topo in {"embedded_reck", "embedded_reck_in_clements"}:
        bs_params_raw, phases = _extract_decomposition(itf.triangle_decomposition(U))
        reck_instructions = _normalize_instructions(bs_params_raw, n_modes)
        instructions = transform_instructions(reck_instructions, n_modes, embedded_total_modes)

        embedded_n = embedded_reck_mode_count(n_modes, embedded_total_modes)
        n_ancilla = embedded_n - n_modes
        embedded_phases = np.zeros(embedded_n, dtype=float)
        phases = np.asarray(phases, dtype=float)
        embedded_phases[n_ancilla : n_ancilla + phases.size] = phases
        return instructions, embedded_phases
    else:
        raise ValueError("topology must be 'Clements', 'Reck', or 'embedded_reck'.")

    bs_params_raw, phases = _extract_decomposition(decomp_fn(U))
    instructions = _normalize_instructions(bs_params_raw, n_modes)
    phases = np.asarray(phases, dtype=float)
    return instructions, phases


def get_Vout(
    U: np.ndarray,
    V0: np.ndarray,
    d0: np.ndarray | None = None,
    eta: float = 0.9,
    topology: str = "Clements",
    embedded_total_modes: int | None = None,
    get_device: bool = False,
):
    """Propagate an input Gaussian state through a target unitary and loss channel."""
    U = np.asarray(U, dtype=complex)
    n_modes = U.shape[0]
    topo = topology.lower()

    instructions, phases = instructions_from_U(U, topology, embedded_total_modes=embedded_total_modes)
    if d0 is None:
        d0 = np.zeros(V0.shape[0], dtype=float)

    validate_covariance(V0.copy(), d0.copy(), hbar=1, tol=1e-9)

    dev = GaussianDevice.from_logical_state(
        d=d0.copy(),
        V=V0.copy(),
        instructions=instructions,
        topology=topology,
        embedded_total_modes=embedded_total_modes,
    )
    d_out, V_out = dev.run(eta=eta, output_phases=phases, logical_output=True)

    if get_device:
        return d_out, V_out, dev
    return d_out, V_out


def _format_real(value: float) -> str:
    if value == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10 ** exp)
    return f"{mant:.16f}*^{exp}"


def _format_number(val: complex, tol: float = 1e-12) -> str:
    real = float(np.real(val))
    imag = float(np.imag(val))
    if abs(imag) <= tol:
        return _format_real(real)
    if abs(real) <= tol:
        return f"{_format_real(imag)} I"
    sign = "+" if imag >= 0 else "-"
    imag_str = f"{_format_real(abs(imag))} I"
    return f"{_format_real(real)} {sign} {imag_str}"


def _matrix_to_wl(matrix: np.ndarray) -> str:
    rows = []
    for row in matrix:
        entries = ",".join(_format_number(v) for v in row)
        rows.append("{" + entries + "}")
    return "{" + ",".join(rows) + "}"


def _extract_three_digit_suffix(stem: str, prefix: str) -> str:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d{{3}})", stem)
    if match is None:
        raise ValueError(
            f"Filename '{stem}' does not follow the expected naming scheme '{prefix}###'."
        )
    return match.group(1)


def _normalize_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return ".pickle"
    if suffix in {".h5", ".hdf5"}:
        return ".h5py"
    return suffix


_SINGLE_FILE_SUFFIXES = {".json", ".pickle", ".npz", ".h5py", ".txt", ".wl"}
_ALL_OUTPUT_SUFFIXES = {".mtx", *_SINGLE_FILE_SUFFIXES}


def _normalize_output_format_option(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"auto", "same"}:
        return "same"
    if not normalized:
        return None
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized == ".pkl":
        normalized = ".pickle"
    if normalized in {".h5", ".hdf5"}:
        normalized = ".h5py"
    return normalized


def _json_to_matrix(value):
    if isinstance(value, dict) and {"real", "imag"} <= set(value):
        real = np.asarray(value["real"], dtype=float)
        imag = np.asarray(value["imag"], dtype=float)
        return real + 1j * imag
    return np.asarray(value)


def _matrix_to_json_compatible(matrix: np.ndarray):
    arr = np.asarray(matrix)
    if np.iscomplexobj(arr):
        return {
            "real": np.asarray(arr.real, dtype=float).tolist(),
            "imag": np.asarray(arr.imag, dtype=float).tolist(),
        }
    return np.asarray(arr, dtype=float).tolist()


def _normalize_time_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _time_values_match(left, right) -> bool:
    left_norm = _normalize_time_value(left)
    right_norm = _normalize_time_value(right)
    if isinstance(left_norm, (int, float, np.integer, np.floating)) and isinstance(
        right_norm, (int, float, np.integer, np.floating)
    ):
        return bool(np.isclose(float(left_norm), float(right_norm)))
    return left_norm == right_norm


def _parse_time_matrix_pairs(payload, *, context: str) -> list[tuple[object, np.ndarray]]:
    if isinstance(payload, dict):
        if "series" in payload:
            payload = payload["series"]
        elif "times" in payload and "matrices" in payload:
            times = payload["times"]
            matrices = payload["matrices"]
            if len(times) != len(matrices):
                raise ValueError(f"{context} has mismatched 'times' and 'matrices' lengths.")
            return [
                (_normalize_time_value(time_val), np.asarray(_json_to_matrix(matrix)))
                for time_val, matrix in zip(times, matrices)
            ]
        else:
            raise ValueError(
                f"{context} must contain either a list of (time, matrix) pairs or keys 'times' and 'matrices'."
            )

    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"{context} must be a list of (time, matrix) pairs.")

    pairs: list[tuple[object, np.ndarray]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"{context} entry {idx} must be a two-item pair of the form (time, matrix)."
            )
        time_val, matrix_val = item
        pairs.append((_normalize_time_value(time_val), np.asarray(_json_to_matrix(matrix_val))))
    return pairs


def _is_scalar_like_json_value(value) -> bool:
    return isinstance(value, (str, int, float, np.integer, np.floating))


def _looks_like_time_matrix_pair_sequence(payload) -> bool:
    if not isinstance(payload, (list, tuple)) or not payload:
        return False
    if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in payload):
        return False

    first_time, first_matrix = payload[0]
    if not _is_scalar_like_json_value(_normalize_time_value(first_time)):
        return False

    if isinstance(first_matrix, dict):
        return {"real", "imag"} <= set(first_matrix) or "matrix" in first_matrix
    return isinstance(first_matrix, (list, tuple, np.ndarray))


def _stack_or_object(values: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    if not arrays:
        return np.asarray([], dtype=float)
    try:
        return np.stack(arrays)
    except ValueError:
        return np.asarray(arrays, dtype=object)


def _time_array_for_storage(times: list[object]) -> np.ndarray:
    normalized = [_normalize_time_value(time_val) for time_val in times]
    try:
        return np.asarray(normalized, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(normalized, dtype=object)


def _looks_like_wolfram_in_json(text: str) -> bool:
    snippet = text.lstrip()
    if not snippet or snippet[0] != "{":
        return False

    second = next((char for char in snippet[1:] if not char.isspace()), "")
    return second != '"'


def _load_wolfram_payload(path: Path):
    text = path.read_text(encoding="utf-8")
    return _parse_wolfram_text(text)


def _load_json_payload(path: Path):
    text = path.read_text(encoding="utf-8")
    if _looks_like_wolfram_in_json(text[:4096]):
        warnings.warn(
            f"File '{path.name}' has a .json suffix but starts with Wolfram-style braces. "
            "It will be parsed as Wolfram Language text instead. "
            "Prefer using the .wl or .txt suffix for this kind of file.",
            stacklevel=3,
        )
        return _load_wolfram_payload(path)
    return json.loads(text)


def _identify_txt_style(text: str) -> str:
    snippet = text.lstrip()
    if not snippet:
        return "unknown_style"

    if snippet.startswith("[") or snippet.startswith('{"') or '"times"' in snippet or '"matrices"' in snippet:
        return "json_style"

    if snippet.startswith("{"):
        if ";" in snippet:
            return "matlab_style"
        if "[" in snippet and not _has_known_wolfram_function_syntax(snippet):
            return "matlab_style"
        return "wolfram_style"

    if snippet.startswith("["):
        return "matlab_style"

    return "unknown_style"


def _warn_txt_style(path: Path, style: str) -> None:
    warnings.warn(
        f"Text file '{path.name}' was identified as {style.replace('_', ' ')}. "
        "Plain-text matrix files can be ambiguous and may introduce parsing or numerical interpretation issues; "
        "prefer native formats such as .json, .pickle, .npz, or .h5py when possible.",
        stacklevel=3,
    )


def _pythonize_wolfram_rational(match: re.Match[str]) -> str:
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        raise ValueError("Invalid Wolfram rational with zero denominator.")
    return f"{numerator / denominator:.17g}"


def _strip_wolfram_comments(text: str) -> str:
    return re.sub(r"\(\*.*?\*\)", "", text, flags=re.DOTALL)


def _has_known_wolfram_function_syntax(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:Abs|ArcCos|ArcSin|ArcTan|Complex|Cos|Cosh|DirectedInfinity|Divide|Exp|Plus|Power|Rational|Sin|Sinh|Sqrt|Subtract|Tan|Tanh|Times)\s*\[",
            text,
        )
    )


def _pythonize_wolfram_text(text: str) -> str:
    converted = _strip_wolfram_comments(text).strip()
    converted = (
        converted.replace(r"\[ImaginaryI]", "I")
        .replace(r"\[ExponentialE]", "E")
        .replace(r"\[Pi]", "Pi")
    )
    converted = re.sub(r"(?<=[0-9.])`(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+))?", "", converted)
    converted = re.sub(
        r"(?<![A-Za-z0-9_.])([+-]?\d+)\s*/\s*([+-]?\d+)(?![A-Za-z0-9_.])",
        _pythonize_wolfram_rational,
        converted,
    )
    converted = converted.replace("*^", "e")
    converted = converted.replace("^", "**")
    converted = converted.replace("[", "(").replace("]", ")")
    converted = re.sub(r"\bI\b", "1j", converted)
    converted = re.sub(r"(?<=[0-9j)])\s+(?=(?:Pi|E|1j|\d|\.|\())", "*", converted)
    converted = re.sub(r"\b(Pi|E)\s+(?=(?:Pi|E|1j|\d|\.|\())", r"\1*", converted)
    converted = converted.replace("{", "[").replace("}", "]")
    return converted


def _as_wolfram_number(value, *, context: str):
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float, complex)):
        raise ValueError(f"Unsupported non-numeric Wolfram {context}: {value!r}")
    return value


def _normalize_wolfram_number(value):
    if isinstance(value, np.generic):
        value = value.item()
    value = np.real_if_close(value)
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, complex) and abs(value.imag) <= 1e-15:
        return float(value.real)
    return value


def _wolfram_rational_value(numerator, denominator):
    numerator = _as_wolfram_number(numerator, context="rational numerator")
    denominator = _as_wolfram_number(denominator, context="rational denominator")
    if denominator == 0:
        raise ValueError("Invalid Wolfram rational with zero denominator.")
    return _normalize_wolfram_number(numerator / denominator)


def _wolfram_complex_value(real, imag):
    return complex(
        _as_wolfram_number(real, context="complex real part"),
        _as_wolfram_number(imag, context="complex imaginary part"),
    )


def _wolfram_unary_math(fn, value):
    value = _as_wolfram_number(value, context="function argument")
    result = fn(value)
    return _normalize_wolfram_number(result)


def _wolfram_directed_infinity(*args):
    if not args:
        return complex(math.inf, math.inf)
    if len(args) != 1:
        raise ValueError("DirectedInfinity expects zero or one argument.")
    direction = _as_wolfram_number(args[0], context="DirectedInfinity direction")
    if direction == 1:
        return math.inf
    if direction == -1:
        return -math.inf
    return complex(math.inf, math.inf)


def _wolfram_product(*values):
    result = 1
    for value in values:
        result *= _as_wolfram_number(value, context="Times argument")
    return _normalize_wolfram_number(result)


def _wolfram_sum(*values):
    result = 0
    for value in values:
        result += _as_wolfram_number(value, context="Plus argument")
    return _normalize_wolfram_number(result)


_WOLFRAM_ALLOWED_NAMES = {
    "Pi": math.pi,
    "E": math.e,
    "Infinity": math.inf,
    "Indeterminate": math.nan,
}
_WOLFRAM_ALLOWED_FUNCTIONS = {
    "Abs": lambda value: _wolfram_unary_math(abs, value),
    "ArcCos": lambda value: _wolfram_unary_math(cmath.acos, value),
    "ArcSin": lambda value: _wolfram_unary_math(cmath.asin, value),
    "ArcTan": lambda value: _wolfram_unary_math(cmath.atan, value),
    "Complex": _wolfram_complex_value,
    "Cos": lambda value: _wolfram_unary_math(cmath.cos, value),
    "Cosh": lambda value: _wolfram_unary_math(cmath.cosh, value),
    "DirectedInfinity": _wolfram_directed_infinity,
    "Divide": _wolfram_rational_value,
    "Exp": lambda value: _wolfram_unary_math(cmath.exp, value),
    "Plus": _wolfram_sum,
    "Power": lambda base, exponent: _normalize_wolfram_number(
        _as_wolfram_number(base, context="Power base") ** _as_wolfram_number(exponent, context="Power exponent")
    ),
    "Rational": _wolfram_rational_value,
    "Sin": lambda value: _wolfram_unary_math(cmath.sin, value),
    "Sinh": lambda value: _wolfram_unary_math(cmath.sinh, value),
    "Sqrt": lambda value: _wolfram_unary_math(cmath.sqrt, value),
    "Subtract": lambda left, right: _normalize_wolfram_number(
        _as_wolfram_number(left, context="Subtract left argument")
        - _as_wolfram_number(right, context="Subtract right argument")
    ),
    "Tan": lambda value: _wolfram_unary_math(cmath.tan, value),
    "Tanh": lambda value: _wolfram_unary_math(cmath.tanh, value),
    "Times": _wolfram_product,
}
_WOLFRAM_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _eval_wolfram_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_wolfram_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, str)):
            return node.value
        raise ValueError(f"Unsupported Wolfram literal: {node.value!r}")
    if isinstance(node, ast.List):
        return [_eval_wolfram_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_wolfram_ast(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        value = _as_wolfram_number(_eval_wolfram_ast(node.operand), context="unary operand")
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("Unsupported unary operator in Wolfram expression.")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _WOLFRAM_BINARY_OPERATORS:
            raise ValueError("Unsupported binary operator in Wolfram expression.")
        left = _as_wolfram_number(_eval_wolfram_ast(node.left), context="binary left operand")
        right = _as_wolfram_number(_eval_wolfram_ast(node.right), context="binary right operand")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Invalid Wolfram rational with zero denominator.")
        return _normalize_wolfram_number(_WOLFRAM_BINARY_OPERATORS[op_type](left, right))
    if isinstance(node, ast.Name):
        if node.id in _WOLFRAM_ALLOWED_NAMES:
            return _WOLFRAM_ALLOWED_NAMES[node.id]
        raise ValueError(f"Unsupported Wolfram symbol '{node.id}'.")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _WOLFRAM_ALLOWED_FUNCTIONS:
            raise ValueError("Unsupported Wolfram function.")
        if node.keywords:
            raise ValueError("Wolfram function keywords are not supported.")
        args = [_eval_wolfram_ast(arg) for arg in node.args]
        return _WOLFRAM_ALLOWED_FUNCTIONS[node.func.id](*args)
    raise ValueError(f"Unsupported Wolfram expression node: {type(node).__name__}.")


def _parse_wolfram_text(text: str):
    converted = _pythonize_wolfram_text(text)
    try:
        parsed = ast.parse(converted, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse Wolfram-style text: {exc.msg}") from exc
    return _eval_wolfram_ast(parsed)


def _split_top_level(text: str, delimiters: set[str]) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        if char in delimiters and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _matlab_scalar_to_python(token: str) -> str:
    value = token.strip()
    value = value.replace("*^", "e")
    value = re.sub(
        r"(?<![A-Za-z0-9_])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*[ij]\b",
        lambda match: match.group(1) + "j",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?<![A-Za-z0-9_])([+-]?)i\b", lambda m: (m.group(1) or "") + "1j", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<![A-Za-z0-9_])([+-]?)j\b", lambda m: (m.group(1) or "") + "1j", value, flags=re.IGNORECASE)
    return value


def _matlab_matrix_body_to_python(body: str) -> str:
    rows = _split_top_level(body, {";"})
    row_literals: list[str] = []
    for row in rows:
        stripped_row = row.strip()
        if not stripped_row:
            continue
        if "," in stripped_row:
            entries = [entry.strip() for entry in stripped_row.split(",") if entry.strip()]
        else:
            entries = [entry.strip() for entry in re.split(r"\s+", stripped_row) if entry.strip()]
        row_literals.append("[" + ", ".join(_matlab_scalar_to_python(entry) for entry in entries) + "]")
    return "[" + ", ".join(row_literals) + "]"


def _pythonize_matlab_text(text: str) -> str:
    source = text.strip()
    result: list[str] = []
    idx = 0
    while idx < len(source):
        char = source[idx]
        if char == "[":
            depth = 1
            end = idx + 1
            while end < len(source) and depth > 0:
                if source[end] == "[":
                    depth += 1
                elif source[end] == "]":
                    depth -= 1
                end += 1
            if depth != 0:
                raise ValueError("Unbalanced brackets in MATLAB-style text file.")
            body = source[idx + 1 : end - 1]
            result.append(_matlab_matrix_body_to_python(body))
            idx = end
            continue
        result.append(char)
        idx += 1

    converted = "".join(result)
    converted = converted.replace("{", "[").replace("}", "]")
    converted = converted.replace(";", ",")
    return converted


def _load_txt_payload(path: Path):
    text = path.read_text(encoding="utf-8")
    style = _identify_txt_style(text[:4096])
    if style == "unknown_style":
        raise ValueError(
            f"Could not identify the style of text file '{path.name}'. "
            "Supported styles are json_style, wolfram_style, and matlab_style."
        )

    _warn_txt_style(path, style)

    if style == "json_style":
        return json.loads(text), style
    if style == "wolfram_style":
        return _parse_wolfram_text(text), style
    if style == "matlab_style":
        return ast.literal_eval(_pythonize_matlab_text(text)), style

    raise ValueError(f"Unsupported text style '{style}' in '{path.name}'.")


def _load_pickle_payload(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_npz_payload(path: Path):
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        if "matrix" in keys:
            return np.asarray(data["matrix"])
        if "times" in keys and "matrices" in keys:
            times = data["times"]
            matrices = data["matrices"]
            return {
                "times": [_normalize_time_value(item) for item in times.tolist()],
                "matrices": matrices.tolist() if matrices.dtype == object else [matrix for matrix in matrices],
            }
        raise ValueError(
            f"Unsupported .npz layout in '{path.name}'. Expected 'matrix' or both 'times' and 'matrices'."
        )


def _load_h5_payload(path: Path):
    if h5py is None:
        raise ImportError(
            "h5py is required to read '.h5py' files. Install it in PowerShell with: "
            "'python -m pip install h5py'."
        )

    with h5py.File(path, "r") as handle:
        if "matrix" in handle:
            return np.asarray(handle["matrix"])
        if "times" in handle and "matrices" in handle:
            times_raw = np.asarray(handle["times"])
            times = [_normalize_time_value(item) for item in times_raw.tolist()]
            matrices = np.asarray(handle["matrices"])
            return {"times": times, "matrices": [matrix for matrix in matrices]}
        raise ValueError(
            f"Unsupported .h5py layout in '{path.name}'. Expected datasets 'matrix' or both 'times' and 'matrices'."
        )


def _load_single_file_payload(path: Path):
    suffix = _normalize_suffix(path)
    if suffix == ".txt":
        return _load_txt_payload(path)
    if suffix == ".wl":
        return _load_wolfram_payload(path), "wolfram_style"
    if suffix == ".json":
        return _load_json_payload(path), None
    if suffix == ".pickle":
        return _load_pickle_payload(path), None
    if suffix == ".npz":
        return _load_npz_payload(path), None
    if suffix == ".h5py":
        return _load_h5_payload(path), None
    raise ValueError(
        f"Unsupported file format '{path.suffix}'. Supported formats are .mtx, .json, .pickle, .npz, .h5py, .txt, and .wl."
    )


def _load_input_series_from_file(path: Path) -> tuple[list[tuple[object, np.ndarray]], str | None]:
    payload, txt_style = _load_single_file_payload(path)
    series = _parse_time_matrix_pairs(payload, context=f"Input file '{path.name}'")
    if not series:
        raise ValueError(f"Input file '{path.name}' does not contain any covariance matrices.")
    return series, txt_style


def _load_unitary_spec_from_file(path: Path, *, time_dependent_unitary: bool):
    payload, txt_style = _load_single_file_payload(path)
    if time_dependent_unitary:
        series = _parse_time_matrix_pairs(payload, context=f"Unitary file '{path.name}'")
        if not series:
            raise ValueError(f"Unitary file '{path.name}' does not contain any matrices.")
        return series, txt_style

    if isinstance(payload, dict) and "matrix" in payload:
        return np.asarray(_json_to_matrix(payload["matrix"])), txt_style

    if isinstance(payload, (list, tuple)):
        if _looks_like_time_matrix_pair_sequence(payload):
            raise ValueError(
                f"Unitary file '{path.name}' contains a time series, but time_dependent_unitary=False."
            )

    return np.asarray(_json_to_matrix(payload)), txt_style


def _write_json_time_series(path: Path, series: list[tuple[object, np.ndarray]]) -> None:
    payload = [
        [_normalize_time_value(time_val), _matrix_to_json_compatible(matrix)]
        for time_val, matrix in series
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_pickle_time_series(path: Path, series: list[tuple[object, np.ndarray]]) -> None:
    payload = [(_normalize_time_value(time_val), np.asarray(matrix)) for time_val, matrix in series]
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_npz_time_series(path: Path, series: list[tuple[object, np.ndarray]]) -> None:
    times = _time_array_for_storage([time_val for time_val, _ in series])
    matrices = _stack_or_object([np.asarray(matrix) for _, matrix in series])
    np.savez(path, times=times, matrices=matrices)


def _write_h5_time_series(path: Path, series: list[tuple[object, np.ndarray]]) -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required to write '.h5py' files. Install it in PowerShell with: "
            "'python -m pip install h5py'."
        )

    times = _time_array_for_storage([time_val for time_val, _ in series])
    matrices = _stack_or_object([np.asarray(matrix) for _, matrix in series])
    with h5py.File(path, "w") as handle:
        if times.dtype == object:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("times", data=np.asarray([str(v) for v in times.tolist()], dtype=str_dtype))
        else:
            handle.create_dataset("times", data=times)
        handle.create_dataset("matrices", data=matrices)


def _txt_number(value) -> str:
    normalized = np.real_if_close(value)
    if np.iscomplexobj(normalized) and abs(np.imag(normalized)) > 1e-12:
        real = float(np.real(normalized))
        imag = float(np.imag(normalized))
        return f"{real:.16g}{imag:+.16g}j"
    return f"{float(np.real(normalized)):.16g}"


def _matrix_to_matlab(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix):
        rows.append(" ".join(_txt_number(entry) for entry in row))
    return "[" + "; ".join(rows) + "]"


def _matrix_to_wolfram_text(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix):
        rows.append("{" + ", ".join(_txt_number(entry) for entry in row) + "}")
    return "{" + ", ".join(rows) + "}"


def _time_to_txt(time_val) -> str:
    normalized = _normalize_time_value(time_val)
    if isinstance(normalized, str):
        return json.dumps(normalized)
    return _txt_number(normalized)


def _write_txt_time_series(path: Path, series: list[tuple[object, np.ndarray]], *, style: str | None) -> None:
    txt_style = style or "json_style"
    if txt_style == "json_style":
        payload = [
            [_normalize_time_value(time_val), _matrix_to_json_compatible(matrix)]
            for time_val, matrix in series
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if txt_style == "wolfram_style":
        entries = [
            "{" + _time_to_txt(time_val) + ", " + _matrix_to_wolfram_text(matrix) + "}"
            for time_val, matrix in series
        ]
        path.write_text("{" + ", ".join(entries) + "}", encoding="utf-8")
        return

    if txt_style == "matlab_style":
        entries = [
            "{" + _time_to_txt(time_val) + ", " + _matrix_to_matlab(matrix) + "}"
            for time_val, matrix in series
        ]
        path.write_text("{" + "; ".join(entries) + "}", encoding="utf-8")
        return

    raise ValueError(f"Unsupported text style '{txt_style}' for output.")


def _write_native_time_series(path: Path, series: list[tuple[object, np.ndarray]], *, txt_style: str | None = None) -> None:
    suffix = _normalize_suffix(path)
    if suffix == ".wl":
        path.write_text(_time_series_to_wl(series), encoding="utf-8")
        return
    if suffix == ".txt":
        _write_txt_time_series(path, series, style=txt_style)
        return
    if suffix == ".json":
        _write_json_time_series(path, series)
        return
    if suffix == ".pickle":
        _write_pickle_time_series(path, series)
        return
    if suffix == ".npz":
        _write_npz_time_series(path, series)
        return
    if suffix == ".h5py":
        _write_h5_time_series(path, series)
        return
    raise ValueError(f"Unsupported output file format '{path.suffix}'.")


def _wl_atom(value) -> str:
    normalized = _normalize_time_value(value)
    if isinstance(normalized, str):
        escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(normalized, (int, float, np.integer, np.floating)):
        return _format_real(float(normalized))
    return f'"{str(normalized)}"'


def _time_series_to_wl(series: list[tuple[object, np.ndarray]]) -> str:
    rows = []
    for time_val, matrix in series:
        rows.append("{" + _wl_atom(time_val) + "," + _matrix_to_wl(np.asarray(matrix, dtype=complex)) + "}")
    return "{\n" + ",\n".join(rows) + "\n}"


def _single_file_output_path(
    output_root: Path,
    topology_label: str,
    eta_tag: str,
    *,
    requested_format: str | None,
    input_suffix: str,
) -> tuple[Path, str]:
    suffix = input_suffix if requested_format in {None, "same"} else requested_format
    if suffix not in _SINGLE_FILE_SUFFIXES:
        raise ValueError(
            "Single-file mode supports output_format='same' or one of "
            ".json, .pickle, .npz, .h5py, .txt, .wl."
        )
    return output_root / f"{topology_label}_{eta_tag}{suffix}", suffix


def _normalize_topology_label(topology: str) -> str:
    topo = topology.lower()
    if topo == "clements":
        return "Clements"
    if topo == "reck":
        return "Reck"
    if topo in {"embedded_reck", "embedded_reck_in_clements"}:
        return "embedded_reck"
    raise ValueError("topology must be 'Clements', 'Reck', or 'embedded_reck'.")


def _load_unitary_from_mtx(matrix_path: Path, n_modes: int) -> np.ndarray:
    data = mmread(str(matrix_path))
    arr = np.asarray(data.todense() if hasattr(data, "todense") else data)

    stem = matrix_path.stem
    if stem == "symplectic" or stem.startswith("symplectic"):
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (2 * n_modes, 2 * n_modes):
            raise ValueError(
                f"Symplectic matrix '{matrix_path.name}' has shape {arr.shape}, expected {(2 * n_modes, 2 * n_modes)}."
            )
        X = np.asarray(arr[:n_modes, :n_modes], dtype=float)
        Y = np.asarray(arr[n_modes:, :n_modes], dtype=float)
        return X + 1j * Y if np.linalg.norm(Y) >= 1e-12 else X.astype(float)

    if stem == "unitary" or stem.startswith("unitary"):
        arr = np.asarray(arr, dtype=complex)
        if arr.shape != (n_modes, n_modes):
            raise ValueError(
                f"Unitary matrix '{matrix_path.name}' has shape {arr.shape}, expected {(n_modes, n_modes)}."
            )
        return arr

    raise ValueError(
        f"Matrix filename '{matrix_path.name}' must start with 'symplectic' or 'unitary'."
    )


def _discover_static_matrix_file(matrix_dir: Path) -> Path:
    unitary_path = matrix_dir / "unitary.mtx"
    symplectic_path = matrix_dir / "symplectic.mtx"
    found = [path for path in (unitary_path, symplectic_path) if path.exists()]
    if not found:
        raise FileNotFoundError(
            f"No static matrix file found in '{matrix_dir}'. Expected 'unitary.mtx' or 'symplectic.mtx'."
        )
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous matrix specification in '{matrix_dir}': found both 'unitary.mtx' and 'symplectic.mtx'."
        )
    return found[0]


def _discover_time_dependent_matrix_files(matrix_dir: Path) -> dict[str, Path]:
    candidates = sorted(matrix_dir.glob("*.mtx"))
    by_index: dict[str, Path] = {}
    total = 0
    for path in candidates:
        stem = path.stem
        is_symplectic = stem.startswith("symplectic")
        is_unitary = stem.startswith("unitary")
        if not (is_symplectic or is_unitary):
            continue
        prefix = "symplectic" if is_symplectic else "unitary"
        idx = _extract_three_digit_suffix(stem, prefix)
        if idx in by_index:
            raise ValueError(
                f"Ambiguous matrix specification for index {idx} in '{matrix_dir}': "
                f"found both '{by_index[idx].name}' and '{path.name}'."
            )
        by_index[idx] = path
        total += 1
    if total == 0:
        raise FileNotFoundError(
            f"No time-dependent matrix files found in '{matrix_dir}'. Expected files named 'symplectic###.mtx' or 'unitary###.mtx'."
        )
    return by_index


def run_on_files(
    input_dir: Path | str | None = None,
    matrix_dir: Path | str | None = None,
    eta: float = 0.9,
    topology: str = "Clements",
    output_dir: Path | str | None = None,
    output_format: str | None = None,
    time_dependent_unitary: bool = False,
    embedded_total_modes: int | None = None,
    input_file: Path | str | None = None,
    unitary_file: Path | str | None = None,
) -> dict[str, Path]:
    """
    High-level wrapper for propagating covariance inputs through a shared or
    time-dependent unitary/symplectic process.

    `.mtx` inputs use the existing directory-based workflow.
    Other supported formats use a single input file and a single unitary file.
    """
    topology_label = _normalize_topology_label(topology)
    output_format_normalized = _normalize_output_format_option(output_format)

    input_file_path = Path(input_file) if input_file is not None else None
    unitary_file_path = Path(unitary_file) if unitary_file is not None else None
    input_dir_path = Path(input_dir) if input_dir is not None else None
    matrix_dir_path = Path(matrix_dir) if matrix_dir is not None else None

    use_single_file_mode = input_file_path is not None or unitary_file_path is not None
    if use_single_file_mode:
        if input_file_path is None or unitary_file_path is None:
            raise ValueError("Single-file mode requires both 'input_file' and 'unitary_file'.")
        if input_dir_path is not None or matrix_dir_path is not None:
            raise ValueError("Use either directory mode or single-file mode, not both at once.")
        if _normalize_suffix(input_file_path) == ".mtx" or _normalize_suffix(unitary_file_path) == ".mtx":
            raise ValueError("Single-file mode is only for non-.mtx formats.")
        if not input_file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        if not unitary_file_path.is_file():
            raise FileNotFoundError(f"Unitary file not found: {unitary_file_path}")
    else:
        if input_dir_path is None or matrix_dir_path is None:
            raise ValueError("Directory mode requires both 'input_dir' and 'matrix_dir'.")
        if not input_dir_path.is_dir():
            raise FileNotFoundError(f"Input covariance directory not found: {input_dir_path}")
        if not matrix_dir_path.is_dir():
            raise FileNotFoundError(f"Matrix directory not found: {matrix_dir_path}")

    if use_single_file_mode:
        if output_format_normalized not in {None, "same", *_SINGLE_FILE_SUFFIXES}:
            raise ValueError(
                "Single-file mode supports output_format='same' or one of "
                ".json, .pickle, .npz, .h5py, .txt, .wl."
            )

        native_suffix = _normalize_suffix(input_file_path)
        if output_dir is None:
            output_root = Path(__file__).resolve().parent / "demos" / "output_covariance_files"
        else:
            output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        eta_tag = f"ETA{int(round(eta * 100)):03d}"
        output_path, output_suffix = _single_file_output_path(
            output_root,
            topology_label,
            eta_tag,
            requested_format=output_format_normalized,
            input_suffix=native_suffix,
        )

        input_series, input_txt_style = _load_input_series_from_file(input_file_path)
        unitary_spec, _ = _load_unitary_spec_from_file(
            unitary_file_path, time_dependent_unitary=time_dependent_unitary
        )

        if time_dependent_unitary:
            assert isinstance(unitary_spec, list)
            if len(unitary_spec) != len(input_series):
                raise ValueError(
                    f"time_dependent_unitary=True requires the same number of covariance and unitary entries, "
                    f"got {len(input_series)} and {len(unitary_spec)}."
                )
            for idx, ((time_cov, _), (time_u, _)) in enumerate(zip(input_series, unitary_spec)):
                if not _time_values_match(time_cov, time_u):
                    raise ValueError(
                        f"Time mismatch at entry {idx}: covariance time {time_cov!r} does not match unitary time {time_u!r}."
                    )
            work_items = [
                (time_cov, np.asarray(V0), np.asarray(U))
                for (time_cov, V0), (_, U) in zip(input_series, unitary_spec)
            ]
        else:
            assert not isinstance(unitary_spec, list)
            work_items = [
                (time_cov, np.asarray(V0), np.asarray(unitary_spec))
                for time_cov, V0 in input_series
            ]

        output_series: list[tuple[object, np.ndarray]] = []
        for time_val, V0, U in work_items:
            if V0.ndim != 2 or V0.shape[0] != V0.shape[1] or V0.shape[0] % 2 != 0:
                raise ValueError("Each covariance matrix in single-file mode must be an even square matrix.")
            n_modes = V0.shape[0] // 2
            if U.shape == (2 * n_modes, 2 * n_modes):
                X = np.asarray(U[:n_modes, :n_modes], dtype=float)
                Y = np.asarray(U[n_modes:, :n_modes], dtype=float)
                U_eff = X + 1j * Y if np.linalg.norm(Y) >= 1e-12 else X.astype(float)
            elif U.shape == (n_modes, n_modes):
                U_eff = np.asarray(U, dtype=complex)
            else:
                raise ValueError(
                    f"Unitary/symplectic matrix at time {time_val!r} has incompatible shape {U.shape} for {n_modes} modes."
                )

            d0 = np.zeros(V0.shape[0], dtype=float)
            _, V_out = get_Vout(
                U_eff,
                np.asarray(V0, dtype=float),
                d0=d0,
                eta=eta,
                topology=topology_label,
                embedded_total_modes=embedded_total_modes,
            )
            V_out_real = np.asarray(np.real_if_close(V_out), dtype=float)
            output_series.append((time_val, V_out_real))
        output_txt_style = input_txt_style if output_suffix == native_suffix else None
        _write_native_time_series(output_path, output_series, txt_style=output_txt_style)
        return {"output_path": output_path}

    if output_format_normalized is None:
        output_format_normalized = ".mtx"
    if output_format_normalized == "same":
        output_format_normalized = ".mtx"
    if output_format_normalized not in _ALL_OUTPUT_SUFFIXES:
        raise ValueError(
            "Directory mode supports output_format='.mtx' or one of "
            ".json, .pickle, .npz, .h5py, .txt, .wl."
        )

    input_files = sorted(input_dir_path.glob("input_cov*.mtx"))
    if not input_files:
        raise FileNotFoundError(
            f"No input covariance files found in '{input_dir_path}'. Expected files named 'input_cov###.mtx'."
        )

    input_by_index: dict[str, Path] = {}
    for path in input_files:
        idx = _extract_three_digit_suffix(path.stem, "input_cov")
        input_by_index[idx] = path

    if output_dir is None:
        default_name = (
            "output_covariance_mtx" if output_format_normalized == ".mtx" else "output_covariance_files"
        )
        output_root = Path(__file__).resolve().parent / "demos" / default_name
    else:
        output_root = Path(output_dir)

    eta_tag = f"ETA{int(round(eta * 100)):03d}"
    written_files: dict[str, Path] = {}
    output_series: list[tuple[object, np.ndarray]] = []

    if output_format_normalized == ".mtx":
        output_target = output_root / topology_label
        output_target.mkdir(parents=True, exist_ok=True)
    else:
        output_target = output_root
        output_target.mkdir(parents=True, exist_ok=True)

    if time_dependent_unitary:
        matrix_by_index = _discover_time_dependent_matrix_files(matrix_dir_path)
        if len(matrix_by_index) != len(input_by_index):
            raise ValueError(
                f"time_dependent_unitary=True requires the same number of input and matrix files, "
                f"got {len(input_by_index)} inputs and {len(matrix_by_index)} matrices."
            )
        missing = sorted(set(input_by_index) - set(matrix_by_index))
        if missing:
            raise ValueError(
                f"Missing matching unitary/symplectic files for input indices: {', '.join(missing)}."
            )
        work_items = [(idx, input_by_index[idx], matrix_by_index[idx]) for idx in sorted(input_by_index)]
    else:
        matrix_path = _discover_static_matrix_file(matrix_dir_path)
        work_items = [(idx, path, matrix_path) for idx, path in sorted(input_by_index.items())]

    for idx, cov_path, matrix_path in work_items:
        data = mmread(str(cov_path))
        V0 = np.asarray(data.todense() if hasattr(data, "todense") else data, dtype=float)
        if V0.ndim != 2 or V0.shape[0] != V0.shape[1] or V0.shape[0] % 2 != 0:
            raise ValueError(f"Input covariance '{cov_path.name}' must be an even square matrix.")

        n_modes = V0.shape[0] // 2
        U = _load_unitary_from_mtx(matrix_path, n_modes)
        d0 = np.zeros(V0.shape[0], dtype=float)
        _, V_out = get_Vout(
            U,
            V0,
            d0=d0,
            eta=eta,
            topology=topology_label,
            embedded_total_modes=embedded_total_modes,
        )

        if output_format_normalized == ".mtx":
            out_name = f"{topology_label}_{idx}_{eta_tag}.mtx"
            out_path = output_target / out_name
            mmwrite(out_path, np.asarray(V_out))
            written_files[idx] = out_path
        else:
            V_out_real = np.asarray(np.real_if_close(V_out), dtype=float)
            output_series.append((idx, V_out_real))

    if output_format_normalized == ".mtx":
        result = {"output_dir": output_target}
    else:
        output_path = output_target / f"{topology_label}_{eta_tag}{output_format_normalized}"
        txt_style = "json_style" if output_format_normalized == ".txt" else None
        _write_native_time_series(output_path, output_series, txt_style=txt_style)
        result = {"output_path": output_path}

    return result
