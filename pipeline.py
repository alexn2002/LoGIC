from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import interferometer as itf
from scipy.io import mmread, mmwrite

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

    validate_covariance(V0.copy(), d0.copy(), hbar=1, tol=1e-12)

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
    input_dir: Path | str,
    matrix_dir: Path | str,
    eta: float = 0.9,
    topology: str = "Clements",
    output_dir: Path | str | None = None,
    output_format: str = "mtx",
    time_dependent_unitary: bool = False,
    embedded_total_modes: int | None = None,
) -> dict[str, Path]:
    """
    High-level wrapper that propagates input covariance .mtx files through a
    shared or time-dependent unitary/symplectic process and writes the output
    covariance matrices in the same style as demo_literature.py.
    """
    input_dir = Path(input_dir)
    matrix_dir = Path(matrix_dir)
    topology_label = _normalize_topology_label(topology)
    output_format_normalized = output_format.lower()
    if output_format_normalized not in {"mtx", "wl"}:
        raise ValueError("output_format must be 'mtx' or 'wl'.")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input covariance directory not found: {input_dir}")
    if not matrix_dir.is_dir():
        raise FileNotFoundError(f"Matrix directory not found: {matrix_dir}")

    input_files = sorted(input_dir.glob("input_cov*.mtx"))
    if not input_files:
        raise FileNotFoundError(
            f"No input covariance files found in '{input_dir}'. Expected files named 'input_cov###.mtx'."
        )

    input_by_index: dict[str, Path] = {}
    for path in input_files:
        idx = _extract_three_digit_suffix(path.stem, "input_cov")
        input_by_index[idx] = path

    if output_dir is None:
        default_name = "output_covariance_mtx" if output_format_normalized == "mtx" else "output_covariance_wl"
        output_root = Path(__file__).resolve().parent / "demos" / default_name
    else:
        output_root = Path(output_dir)

    eta_tag = f"ETA{int(round(eta * 100)):03d}"
    written_files: dict[str, Path] = {}
    wl_matrices: list[str] = [] if output_format_normalized == "wl" else []

    if output_format_normalized == "mtx":
        output_target = output_root / topology_label
        output_target.mkdir(parents=True, exist_ok=True)
    else:
        output_target = output_root
        output_target.mkdir(parents=True, exist_ok=True)

    if time_dependent_unitary:
        matrix_by_index = _discover_time_dependent_matrix_files(matrix_dir)
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
        matrix_path = _discover_static_matrix_file(matrix_dir)
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

        if output_format_normalized == "mtx":
            out_name = f"{topology_label}_{idx}_{eta_tag}.mtx"
            out_path = output_target / out_name
            mmwrite(out_path, np.asarray(V_out))
            written_files[idx] = out_path
        else:
            wl_matrices.append(_matrix_to_wl(np.asarray(V_out, dtype=complex)))

    if output_format_normalized == "mtx":
        result: dict[str, Path] = {"output_dir": output_target}
    else:
        wl_path = output_target / f"{topology_label}_{eta_tag}.wl"
        content = "{\n" + ",\n".join(wl_matrices) + "\n}"
        wl_path.write_text(content, encoding="utf-8")
        result = {"output_path": wl_path}

    return result
