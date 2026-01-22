from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as sla

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from devices import GaussianDevice, random_squeezed_vacuum
from pipeline import get_Vout


def _haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random n x n unitary via QR decomposition."""
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = sla.qr(z)
    diag = np.diag(r)
    phases = np.ones_like(diag)
    non_zero = np.abs(diag) > 0
    phases[non_zero] = diag[non_zero] / np.abs(diag[non_zero])
    return q * phases


def main():
    parser = argparse.ArgumentParser(description="Simple demo: propagate a squeezed vacuum through a random mesh.")
    parser.add_argument("--modes", type=int, default=4, help="Number of spatial modes.")
    parser.add_argument("--eta", type=float, default=0.9, help="Amplitude transmission for each beamsplitter.")
    parser.add_argument(
        "--topology", type=str, default="Clements", choices=("Clements", "Reck", "clements", "reck"), help="Mesh topology."
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    U = _haar_unitary(args.modes, rng)
    d0, V0 = random_squeezed_vacuum(args.modes, rng=rng)
    input_dev = GaussianDevice(d0.copy(), V0.copy(), instructions=()) # store the input state for reference

    d_out, V_out, output_dev = get_Vout(U, V0, d0=d0, eta=args.eta, topology=args.topology, get_device=True)

    log_dir = PROJECT_ROOT / "demos" / "logs" / "demo_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"demo_pipeline_{args.topology}_eta{int(round(args.eta * 100)):03d}.txt"
    log_path.write_text(
        "\n".join(
            [
                f"topology={args.topology}",
                f"modes={args.modes}",
                f"eta={args.eta:.6f}",
                f"n_in={float(np.real_if_close(input_dev.exp_photon_number())):.17g}",
                f"n_out={float(np.real_if_close(output_dev.exp_photon_number())):.17g}",
                f"first_moments={output_dev.first_moments()}",
                f"covariance_shape={V_out.shape}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Mesh topology: {args.topology}, modes: {args.modes}, eta={args.eta}")
    print("Input photon number:", float(np.real_if_close(input_dev.exp_photon_number())))
    print("Output photon number:", float(np.real_if_close(output_dev.exp_photon_number())))
    print("First moments:", output_dev.first_moments())
    print("Wrote log:", log_path)


if __name__ == "__main__":
    main()
