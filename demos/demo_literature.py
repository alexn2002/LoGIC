from __future__ import annotations

import math
import re
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as sla
from scipy.io import mmread, mmwrite
import argparse

# compatibility for external interferometer package with numpy>=2
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
RESULT_ROOT = PROJECT_ROOT / "demos" / "output_covariance_mtx"
LOG_ROOT = PROJECT_ROOT / "demos" / "logs" / "demo_literature"
WL_RESULT_ROOT = PROJECT_ROOT / "demos" / "output_covariance_wl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from devices import GaussianDevice
import interferometer as itf

RESULT_LABELS = ("Reck", "Clements")


def _reset_result_dirs(result_root: Path | None = None):
    root = result_root if result_root is not None else RESULT_ROOT
    log_root = LOG_ROOT
    for label in RESULT_LABELS:
        target_dir = root / label
        if target_dir.exists():
            for file in target_dir.glob("*.mtx"):
                file.unlink()
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
    if log_root.exists():
        for file in log_root.glob("**/*.txt"):
            file.unlink()
    else:
        log_root.mkdir(parents=True, exist_ok=True)


def _convert_beamsplitters(bs_list):
    """Normalize BS tuples from interferometer to (k, l, theta, phi) with 0-based indices."""
    instructions = []
    for entry in bs_list:
        if len(entry) == 4:
            k, l, theta, phi = entry
        elif len(entry) == 3:
            k, l, theta = entry
            phi = 0.0
        else:
            raise ValueError(f"Unexpected beamsplitter tuple: {entry}")
        instructions.append((int(k), int(l), float(theta), float(phi or 0.0)))
    return instructions


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


def _write_wl_from_dir(mtx_dir: Path, output_path: Path) -> int:
    if not mtx_dir.is_dir():
        return 0
    wl_matrices = []
    for path in sorted(mtx_dir.glob("*.mtx")):
        data = mmread(str(path))
        arr = np.asarray(data.todense() if hasattr(data, "todense") else data, dtype=complex)
        wl_matrices.append(_matrix_to_wl(arr))
    if not wl_matrices:
        return 0
    content = "{\n" + ",\n".join(wl_matrices) + "\n}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return len(wl_matrices)


def _extract_input_index(stem: str) -> str:
    match = re.search(r"(\\d+)$", stem)
    return match.group(1) if match else stem


def _unitary_from_instructions(instructions, n):
    """Reconstruct the mesh unitary using the SAME convention
    as GaussianDevice._bs_unitary_block().

        B(theta, phi) = [[ e^{i phi} cos theta,   -sin theta ],
                         [ e^{i phi} sin theta,    cos theta ]]
    """
    M = np.eye(n, dtype=complex)
    for inst in instructions:
        if len(inst) == 3:
            k, l, theta = inst
            phi = 0.0
        else:
            k, l, theta, phi = inst
            if phi is None:
                phi = 0.0

        c = math.cos(theta)
        s = math.sin(theta)
        phase = np.exp(1j * phi)

        block = np.eye(n, dtype=complex)
        block[k, k] = c * phase
        block[k, l] = -s
        block[l, k] = s * phase
        block[l, l] = c

        M = block @ M

    return M


def _unitary_to_symplectic(U: np.ndarray) -> np.ndarray:
    """Map an n x n complex unitary to its 2n x 2n real symplectic representation."""
    U = np.asarray(U, dtype=complex)
    return np.block([[U.real, -U.imag], [U.imag, U.real]])


def compute_lossy_V(
    cov_filename: Path,
    eta_loss: float = 0.9,
    result_root: Path | None = None,
    symplectic_path: Path | None = None,
    symplectic: bool = True,
    write_files: bool = True,
    return_results: bool = False,
    instructions_override: list[tuple[int, int, float, float]] | None = None,
    phases_override: np.ndarray | None = None,
    label_override: str | None = None,
) -> dict | None:
    cov_path = cov_filename
    if not cov_path.exists():
        raise FileNotFoundError(f"Covariance matrix file not found: {cov_path}")
    V = mmread(str(cov_path))
    V_eff = np.asarray(V)
    d = np.zeros(shape=(V.shape[0]))

    if symplectic_path is None:
        symplectic_path = PROJECT_ROOT / "demos" / "interferometer_symplectic" / "symplectic.mtx"
    if not symplectic_path.exists():
        raise FileNotFoundError(f"Symplectic matrix file not found: {symplectic_path}")
    M = mmread(str(symplectic_path))
    if symplectic:
        unitary_full = np.asarray(M, dtype=float)
        ortho_error = np.linalg.norm(unitary_full @ unitary_full.T - np.eye(unitary_full.shape[0]))
        det_val = np.linalg.det(unitary_full)
        print(f"[unitary] ||MM^T - I||_F={ortho_error:.3e} det(M)={det_val:.3e}")

        n_modes = V.shape[0] // 2
        if unitary_full.shape[0] != 2 * n_modes:
            raise ValueError("Symplectic unitary size does not match covariance size.")
        X = np.asarray(unitary_full[:n_modes, :n_modes], dtype=float)
        Y = np.asarray(unitary_full[n_modes:, :n_modes], dtype=float)
        Y_top = np.asarray(unitary_full[:n_modes, n_modes:], dtype=float)
        y_norm = float(np.linalg.norm(Y))
        y_mismatch = float(np.linalg.norm(Y + Y_top))
        if y_norm < 1e-12:
            unitary_block = X.astype(float)
            print(f"[unitary] Y block norm ~0 (||Y||={y_norm:.3e}); using real X as unitary block.")
        else:
            unitary_block = X + 1j * Y
            print(f"[unitary] Using X + iY with ||Y||={y_norm:.3e}, ||Y+Y_top||={y_mismatch:.3e}")
    else:
        unitary_block = np.asarray(M, dtype=complex)
        err_unitary = np.linalg.norm(unitary_block @ unitary_block.conj().T - np.eye(unitary_block.shape[0]))
        det_val = np.linalg.det(unitary_block)
        print(f"[unitary] ||UU^H - I||_F={err_unitary:.3e} det(U)={det_val:.3e}")
        n_modes = unitary_block.shape[0]

    def _run_topology(label: str, bs_decomp):
        # ------------------------------------------------------------------
        # Decomposition from PyPI interferometer
        # ------------------------------------------------------------------
        if instructions_override is not None:
            if label_override is not None and label != label_override:
                return None
            instructions = instructions_override
            phases = np.asarray(phases_override if phases_override is not None else [], dtype=float)
            print("Warning: instructions are overridden; using provided instructions/phases instead of unitary decomposition.")
        else:
            decomp = bs_decomp(unitary_block)
            if hasattr(decomp, "BS_list"):
                # interferometer-style object
                bs = [(bs.mode1 - 1, bs.mode2 - 1, bs.theta, bs.phi) for bs in decomp.BS_list]
                phases = np.asarray(getattr(decomp, "output_phases", []), dtype=float)
            else:
                # (bs_list, phases) tuple
                bs, phases = decomp

            instructions = _convert_beamsplitters(bs)

        print(f"[{label}] Generated {len(instructions)} instructions for {n_modes} modes.")
        # Save instructions (once per topology) and phases (if any) for inspection
        if write_files:
            instr_path = (result_root or RESULT_ROOT) / label / "instructions.txt"
            instr_path.parent.mkdir(parents=True, exist_ok=True)
            if not instr_path.exists():
                with instr_path.open("w", encoding="utf-8") as fh:
                    for k, l, theta, phi in instructions:
                        fh.write(f"{k},{l},{theta:.12f},{phi:.12f}\n")
            if phases is not None and phases.size > 0:
                phases_path = (result_root or RESULT_ROOT) / label / "phases.txt"
                if not phases_path.exists():
                    with phases_path.open("w", encoding="utf-8") as fh:
                        fh.write(",".join(f"{float(p):.17g}" for p in phases))

        # ------------------------------------------------------------------
        # Unitary reconstruction
        # ------------------------------------------------------------------
        M_mesh = _unitary_from_instructions(instructions, n_modes)
        phase_mat = np.diag(np.exp(1j * phases)) if phases is not None and phases.size > 0 else None
        M_total = phase_mat @ M_mesh if phase_mat is not None else M_mesh

        diff_instr = np.linalg.norm(M_total - unitary_block)
        print(f"[{label}] ||unitary_from_instructions - target||_F={diff_instr:.3e}")

        # ------------------------------------------------------------------
        # GaussianDevice simulation: lossless and lossy
        # ------------------------------------------------------------------
        device = GaussianDevice(d.copy(), V_eff.copy(), instructions=instructions)
        lossy = GaussianDevice(d.copy(), V_eff.copy(), instructions=instructions)

        # Input statistics (before network)
        n_in = float(device.exp_photon_number())
        first_in = device.first_moments()

        # Apply mesh and phases
        device.apply_network(eta=1.0)
        lossy.apply_network(eta=eta_loss)
        if phases is not None and phases.size > 0:
            device.apply_output_phases(phases)
            lossy.apply_output_phases(phases)

        V_lossless = device.V
        V_lossy = lossy.V

        # Output statistics
        n_out_lossless = float(np.real_if_close(device.exp_photon_number()))
        first_out_lossless = device.first_moments()
        n_out_lossy = float(np.real_if_close(lossy.exp_photon_number()))
        first_out_lossy = lossy.first_moments()

        S_total = _unitary_to_symplectic(M_total)
        target_local = S_total @ V_eff @ S_total.T
        err = np.linalg.norm(V_lossless - target_local)
        print(f'[{label}] Lossless network ||GVG^T - MVM^T||_F =', err)


        if write_files:
            target_root = result_root if result_root is not None else RESULT_ROOT
            res_dir = target_root / label
            res_dir.mkdir(parents=True, exist_ok=True)
            stem = cov_path.stem
            file_idx = _extract_input_index(stem)
            eta_tag = f"eta{int(round(eta_loss * 100)):03d}"
            out_name = f"{label}_{file_idx}_ETA{eta_tag}.mtx"
            mmwrite(res_dir / out_name, V_lossy)
            print(f"[{label}] Lossy network photon number (eta={eta_loss:.2f}):", float(np.real_if_close(lossy.exp_photon_number())))
            # Save photon numbers and first moments with full precision
            def _fmt(arr):
                return "[" + ",".join(f"{float(x):.17g}" for x in np.ravel(arr)) + "]"

            log_dir = LOG_ROOT / label
            log_dir.mkdir(parents=True, exist_ok=True)
            stats_path = log_dir / f"moments_{label}_{stem}.txt"
            with stats_path.open("w", encoding="utf-8") as fh:
                fh.write(f"n_in={n_in:.17g}\n")
                fh.write(f"first_in={_fmt(first_in)}\n")
                fh.write(f"n_out_lossless={n_out_lossless:.17g}\n")
                fh.write(f"first_out_lossless={_fmt(first_out_lossless)}\n")
                fh.write(f"n_out_lossy={n_out_lossy:.17g}\n")
                fh.write(f"first_out_lossy={_fmt(first_out_lossy)}\n")
            # Append total photon number summary
            totals_path = log_dir / "N_total.txt"
            with totals_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{stem}, {n_out_lossy:.17g}\n")
        return {
            "label": label,
            "M_unitary": M_total,
            "V_lossy": V_lossy,
            "V_lossless": V_lossless,
        }

    reck_result = _run_topology("Reck", itf.triangle_decomposition)
    print("\n--- Clements topology ---")
    clements_result = _run_topology("Clements", itf.square_decomposition)

    if reck_result is not None and clements_result is not None:
        diff_unitary = np.linalg.norm(reck_result["M_unitary"] - clements_result["M_unitary"])
        diff_lossless = np.linalg.norm(reck_result["V_lossless"] - clements_result["V_lossless"])
        diff_lossy = np.linalg.norm(reck_result["V_lossy"] - clements_result["V_lossy"])
        print(
            f"[Reck vs Clements] ||M_reck - M_clements||_F={diff_unitary:.3e} "
            f"||V_lossless_reck - V_lossless_clements||_F={diff_lossless:.3e} "
            f"||V_lossy_reck - V_lossy_clements||_F={diff_lossy:.3e}"
        )

    if return_results:
        return {"Reck": reck_result, "Clements": clements_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lossy covariance matrices for Reck/Clements decompositions.")
    parser.add_argument(
        "--in-dir",
        "--input-dir",
        dest="in_dir",
        type=Path,
        default=None,
        help="Path to covariance .mtx file or directory. If omitted, uses ./demos/input_covariance_mtx.",
    )
    parser.add_argument(
        "--eta",
        "--loss",
        type=float,
        default=0.9,
        dest="eta_loss",
        help="Loss transmissivity eta applied when simulating the network (default: 0.9).",
    )
    parser.add_argument(
        "--symplectic-file",
        "--process-mtx-file",
        "--symplectic-mtx-file",
        type=Path,
        default=None,
        dest="symplectic_file",
        help="Explicit symplectic .mtx file to use (default: ./demos/interferometer_symplectic/symplectic.mtx).",
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Directory for output .mtx files (default: ./demos/output_covariance_mtx).",
    )
    parser.add_argument(
        "--no-symplectic",
        action="store_true",
        default=False,
        help="Treat provided matrix as n x n unitary directly (default: interpret as 2n x 2n symplectic).",
    )
    args = parser.parse_args()

    _reset_result_dirs(args.out_dir)

    if args.in_dir is not None:
        cov_path = args.in_dir
        if cov_path.is_dir():
            cov_files = sorted(cov_path.glob("*.mtx"))
            if not cov_files:
                raise FileNotFoundError(f"No .mtx files found in directory: {cov_path}")
            for path in cov_files:
                print("\n=== Processing", path.stem, "===")
                compute_lossy_V(
                    path,
                    eta_loss=args.eta_loss,
                    symplectic_path=args.symplectic_file,
                    symplectic=not args.no_symplectic,
                    result_root=args.out_dir,
                )
        else:
            print("\n=== Processing", cov_path.stem, "===")
            compute_lossy_V(
                cov_path,
                eta_loss=args.eta_loss,
                symplectic_path=args.symplectic_file,
                symplectic=not args.no_symplectic,
                result_root=args.out_dir,
            )
    else:
        cov_dir = PROJECT_ROOT / "demos" / "input_covariance_mtx"
        for cov_path in sorted(cov_dir.glob("*.mtx")):
            print("\n=== Processing", cov_path.stem, "===")
            compute_lossy_V(
                cov_path,
                eta_loss=args.eta_loss,
                symplectic_path=args.symplectic_file,
                symplectic=not args.no_symplectic,
                result_root=args.out_dir,
            )

    eta_tag = f"eta{int(round(args.eta_loss * 100)):03d}"
    for label in RESULT_LABELS:
        mtx_dir = (args.out_dir or RESULT_ROOT) / label
        output_path = WL_RESULT_ROOT / f"{label}_ETA{eta_tag}.wl"
        count = _write_wl_from_dir(mtx_dir, output_path)
        if count:
            print(f"[{label}] Wrote {count} matrices to {output_path}")
