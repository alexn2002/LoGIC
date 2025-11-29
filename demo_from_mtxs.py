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
RESULT_ROOT = TESTS_ROOT / "mtxs_res"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from devices import GaussianDevice
import interferometer as itf

RESULT_LABELS = ("Reck", "Clements")


def _cov_sort_key(path: Path) -> tuple[int, str]:
    """Sort covariance files numerically by trailing digits if present."""
    m = re.search(r"(\d+)\.mtx$", path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (0, path.name)


def _reset_result_dirs():
    for label in RESULT_LABELS:
        target_dir = RESULT_ROOT / label
        if target_dir.exists():
            for file in target_dir.glob("*.mtx"):
                file.unlink()
            for file in target_dir.glob("N_total*.txt"):
                file.unlink()
        else:
            target_dir.mkdir(parents=True, exist_ok=True)


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
    unitary_path: Path | None = None,
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

    if unitary_path is None:
        unitary_path = TESTS_ROOT / "mtxs_untry" / "matDFT25.mtx"
    if not unitary_path.exists():
        raise FileNotFoundError(f"Unitary matrix file not found: {unitary_path}")
    M = mmread(str(unitary_path))
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
            #V_save = np.real_if_close(V_lossy)
            mmwrite(res_dir / f"Lossy{label}{stem}.mtx", V_lossy)
            print(f"[{label}] Lossy network photon number (eta={eta_loss:.2f}):", float(np.real_if_close(lossy.exp_photon_number())))
            # Save photon numbers and first moments with full precision
            def _fmt(arr):
                return "[" + ",".join(f"{float(x):.17g}" for x in np.ravel(arr)) + "]"

            stats_path = res_dir / f"moments_{label}_{stem}.txt"
            with stats_path.open("w", encoding="utf-8") as fh:
                fh.write(f"n_in={n_in:.17g}\n")
                fh.write(f"first_in={_fmt(first_in)}\n")
                fh.write(f"n_out_lossless={n_out_lossless:.17g}\n")
                fh.write(f"first_out_lossless={_fmt(first_out_lossless)}\n")
                fh.write(f"n_out_lossy={n_out_lossy:.17g}\n")
                fh.write(f"first_out_lossy={_fmt(first_out_lossy)}\n")
            # Append total photon number summary
            totals_path = res_dir / "N_total.txt"
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
        "--cov",
        type=Path,
        default=None,
        help="Path to a specific covariance .mtx file. If omitted, process all files in ./mtxs_cov.",
    )
    parser.add_argument(
        "--eta-loss",
        type=float,
        default=0.9,
        dest="eta_loss",
        help="Loss transmissivity eta applied when simulating the network (default: 0.9).",
    )
    parser.add_argument(
        "--untry-file",
        type=Path,
        default=None,
        help="Explicit unitary .mtx file to use (ignored if --cov points to covariances).",
    )
    parser.add_argument(
        "--no-symplectic",
        action="store_true",
        default=False,
        help="Treat provided matrix as n x n unitary directly (default: interpret as 2n x 2n symplectic).",
    )
    args = parser.parse_args()

    _reset_result_dirs()

    if args.cov is not None:
        cov_path = args.cov
        if cov_path.is_dir():
            cov_files = sorted(cov_path.glob("*.mtx"), key=_cov_sort_key)
            if not cov_files:
                raise FileNotFoundError(f"No .mtx files found in directory: {cov_path}")
            for path in cov_files:
                print("\n=== Processing", path.stem, "===")
                compute_lossy_V(path, eta_loss=args.eta_loss, unitary_path=args.untry_file, symplectic=not args.no_symplectic)
        else:
            print("\n=== Processing", args.cov.stem, "===")
            compute_lossy_V(args.cov, eta_loss=args.eta_loss, unitary_path=args.untry_file, symplectic=not args.no_symplectic)
    else:
        cov_dir = TESTS_ROOT / "mtxs_cov"
        for cov_path in sorted(cov_dir.glob("*.mtx"), key=_cov_sort_key):
            print("\n=== Processing", cov_path.stem, "===")
            compute_lossy_V(cov_path, eta_loss=args.eta_loss, unitary_path=args.untry_file, symplectic=not args.no_symplectic)
