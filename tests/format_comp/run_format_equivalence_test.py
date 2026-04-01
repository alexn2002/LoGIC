'''
This script is for developers to generate and compare outputs across all supported input and output
formats, using the same underlying data. It is not intended as a regular test, but rather as a tool
to check for format-related discrepancies during development.

No cli flags are supported, parameters have to be changed by hand in the script. The main parameters to adjust are:
- N_MODES: the number of modes in the test fixture (default 3)
- ETA: the loss parameter to use in the run_on_files() calls (default 1.0)
- TOPOLOGIES: the list of topologies to test (default ("Clements", "Reck", "embedded_reck"))
- TOL: the per-element absolute tolerance for comparing outputs (default 1e-14). Note that due to the nature of the computations
    and the use of different formats, some small discrepancies may be expected, but they should ideally be well below typical numerical precision limits.
Note that for 0.0 < eta < 1.0 the outputs will differ topology wise
'''

from __future__ import annotations

import warnings
from pathlib import Path
import sys

import numpy as np
from scipy.io import mmread, mmwrite

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import h5py  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

from devices import random_squeezed_vacuum
from pipeline import _load_input_series_from_file, _write_native_time_series, run_on_files


# Change this by hand to test a different number of modes.
N_MODES = 3

ETA = 1.0
TIME_VALUE = 0.0
TOPOLOGIES = ("Clements", "Reck", "embedded_reck")
TOL = 1e-14

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "generated_outputs"


def random_unitary(n: int, *, seed: int = 1234) -> np.ndarray:
    """Generate a Haar-random unitary via QR decomposition."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 1e-15, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conjugate(phases))


def unitary_to_symplectic(U: np.ndarray) -> np.ndarray:
    """Return the real 2n x 2n symplectic matrix induced by an n x n unitary."""
    U = np.asarray(U, dtype=complex)
    X = U.real
    Y = U.imag
    return np.block([[X, -Y], [Y, X]])


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_mtx_series(series: list[tuple[object, np.ndarray]], target_dir: Path, prefix: str) -> None:
    ensure_clean_dir(target_dir)
    for old in target_dir.glob(f"{prefix}*.mtx"):
        old.unlink()
    for idx, (_, matrix) in enumerate(series, start=1):
        mmwrite(target_dir / f"{prefix}{idx:03d}.mtx", np.asarray(matrix))


def write_supported_single_file_formats(
    series: list[tuple[object, np.ndarray]],
    stem: str,
    target_dir: Path,
) -> dict[str, Path]:
    ensure_clean_dir(target_dir)
    outputs: dict[str, Path] = {}
    for suffix in (".json", ".pickle", ".npz", ".wl", ".txt"):
        path = target_dir / f"{stem}{suffix}"
        txt_style = "json_style" if suffix == ".txt" else None
        _write_native_time_series(path, series, txt_style=txt_style)
        outputs[suffix] = path
    if h5py is not None:
        path = target_dir / f"{stem}.h5py"
        _write_native_time_series(path, series)
        outputs[".h5py"] = path
    return outputs


def load_series_from_output(path: Path) -> list[tuple[object, np.ndarray]]:
    if path.is_dir():
        files = sorted(path.glob("*.mtx"))
        series: list[tuple[object, np.ndarray]] = []
        for idx, file_path in enumerate(files):
            data = mmread(str(file_path))
            arr = np.asarray(
                np.real_if_close(data.todense() if hasattr(data, "todense") else data),
                dtype=float,
            )
            series.append((idx, arr))
        return series

    series, _ = _load_input_series_from_file(path)
    parsed: list[tuple[object, np.ndarray]] = []
    for time_value, matrix in series:
        parsed.append((time_value, np.asarray(np.real_if_close(matrix), dtype=float)))
    return parsed


def compare_series(
    truth: list[tuple[object, np.ndarray]],
    candidate: list[tuple[object, np.ndarray]],
) -> float:
    if len(truth) != len(candidate):
        raise ValueError(f"Series length mismatch: {len(truth)} vs {len(candidate)}")

    max_abs_diff = 0.0
    for (_, truth_matrix), (_, candidate_matrix) in zip(truth, candidate):
        if truth_matrix.shape != candidate_matrix.shape:
            raise ValueError(
                f"Matrix shape mismatch: {truth_matrix.shape} vs {candidate_matrix.shape}"
            )
        if truth_matrix.size:
            max_abs_diff = max(
                max_abs_diff,
                float(np.max(np.abs(truth_matrix - candidate_matrix))),
            )
    return max_abs_diff


def main() -> None:
    print(f"Generating fresh format-equivalence fixtures in {ROOT}")
    ensure_clean_dir(INPUTS)
    ensure_clean_dir(OUTPUTS)

    U = random_unitary(N_MODES)
    S = unitary_to_symplectic(U)
    _, V = random_squeezed_vacuum(N_MODES, rng=np.random.default_rng(2026))

    input_series = [(TIME_VALUE, np.asarray(V, dtype=float))]
    unitary_series = [(TIME_VALUE, np.asarray(U, dtype=complex))]
    symplectic_series = [(TIME_VALUE, np.asarray(S, dtype=float))]

    # Overwrite the shared fixtures in tests/format_comp/inputs.
    input_paths = write_supported_single_file_formats(input_series, "series_input", INPUTS)
    unitary_paths = write_supported_single_file_formats(unitary_series, "series_unitary", INPUTS)
    symplectic_paths = write_supported_single_file_formats(
        symplectic_series, "series_symplectic", INPUTS
    )

    write_mtx_series(input_series, INPUTS / "mtx" / "input_covariance_mtx", "input_cov")
    write_mtx_series(unitary_series, INPUTS / "mtx" / "unitary_mtx", "unitary")
    write_mtx_series(symplectic_series, INPUTS / "mtx" / "symplectic_mtx", "symplectic")

    format_pairs: dict[str, dict[str, Path]] = {
        "mtx_unitary": {
            "input_dir": INPUTS / "mtx" / "input_covariance_mtx",
            "matrix_dir": INPUTS / "mtx" / "unitary_mtx",
        },
        "mtx_symplectic": {
            "input_dir": INPUTS / "mtx" / "input_covariance_mtx",
            "matrix_dir": INPUTS / "mtx" / "symplectic_mtx",
        },
    }

    for suffix, input_path in input_paths.items():
        label = suffix.lstrip(".")
        format_pairs[f"{label}_unitary"] = {
            "input_file": input_path,
            "unitary_file": unitary_paths[suffix],
        }
        format_pairs[f"{label}_symplectic"] = {
            "input_file": input_path,
            "unitary_file": symplectic_paths[suffix],
        }

    print("\nRunning run_on_files() for all formats and topologies...")
    discrepancies: list[tuple[str, str, float]] = []
    for topology in TOPOLOGIES:
        print(f"\n=== {topology} | eta={ETA} ===")
        truth_result = run_on_files(
            input_dir=INPUTS / "mtx" / "input_covariance_mtx",
            matrix_dir=INPUTS / "mtx" / "unitary_mtx",
            output_dir=OUTPUTS / topology / "mtx_unitary",
            output_format=".mtx",
            eta=ETA,
            topology=topology,
            time_dependent_unitary=True,
        )
        truth_series = load_series_from_output(truth_result["output_dir"])
        print(f"{'mtx_unitary':>16}: baseline -> {truth_result['output_dir']}")

        for label, kwargs in format_pairs.items():
            if label == "mtx_unitary":
                continue

            output_root = OUTPUTS / topology / label
            call_kwargs = dict(kwargs)
            call_kwargs.update(
                {
                    "eta": ETA,
                    "topology": topology,
                    "output_dir": output_root,
                    "time_dependent_unitary": True,
                    "output_format": ".mtx" if label.startswith("mtx_") else "same",
                }
            )
            result = run_on_files(**call_kwargs)
            output_path = result.get("output_dir", result.get("output_path"))
            candidate_series = load_series_from_output(output_path)
            max_abs_diff = compare_series(truth_series, candidate_series)
            print(f"{label:>16}: max_abs_diff = {max_abs_diff:.3e}")
            if max_abs_diff > TOL:
                warnings.warn(
                    f"{topology} / {label} differs from the mtx_unitary baseline by "
                    f"{max_abs_diff:.3e} (tolerance {TOL:.1e}).",
                    stacklevel=1,
                )
                discrepancies.append((topology, label, max_abs_diff))

    print("\nFinished.")
    if discrepancies:
        print("Discrepancies above tolerance were detected:")
        for topology, label, max_abs_diff in discrepancies:
            print(f"  {topology:>14} | {label:>16} | {max_abs_diff:.3e}")
    else:
        print(f"All outputs agree with the mtx_unitary baseline within {TOL:.1e} per-element tolerance.")


if __name__ == "__main__":
    main()
