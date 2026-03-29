from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from devices import (
    GaussianDevice,
    build_instructions,
    effective_loss_curve,
    random_squeezed_vacuum,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _format_array(arr: np.ndarray) -> str:
    flat = np.ravel(arr)
    return "[" + ", ".join(f"{float(x):.17g}" for x in flat) + "]"


def _normalize_topology(label: str) -> str:
    topo = label.lower()
    if topo == "clements":
        return "Clements"
    if topo == "reck":
        return "Reck"
    if topo in {"reck down", "reck_down", "reck-down"}:
        return "Reck down"
    if topo in {"embedded_reck", "embedded_reck_in_clements"}:
        return "embedded_reck"
    raise ValueError(f"Unsupported topology: {label}")


def _plot_loss_curve(etas: np.ndarray, curves: dict[str, np.ndarray], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best effort plotting
        raise RuntimeError("matplotlib is required to write plots; install it and retry.") from exc

    plt.figure(figsize=(6, 4))
    for label, vals in curves.items():
        plt.plot(etas, vals, label=label)
    plt.xlabel("eta (loss transmissivity)")
    plt.ylabel("expected photon number")
    plt.title("Effective loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Intro demo using devices.GaussianDevice directly (no pipeline).")
    parser.add_argument("--modes", type=int, default=4, help="Number of spatial modes.")
    parser.add_argument("--eta", type=float, default=0.9, help="Loss transmissivity for the beam splitter network.")
    parser.add_argument(
        "--topology",
        type=str,
        default="Clements",
        choices=(
            "Clements",
            "Reck",
            "Reck down",
            "reck_down",
            "clements",
            "reck",
            "embedded_reck",
            "embedded_reck_in_clements",
        ),
        help="Beam splitter network topology.",
    )
    parser.add_argument(
        "--embedded-total-modes",
        type=int,
        default=None,
        help="Total Clements mesh size for embedded_reck. Must be at least 2*modes - 2.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    topology = _normalize_topology(args.topology)

    # ------------------------------------------------------------------
    # 1) Build an input Gaussian state (random squeezed vacuum)
    # ------------------------------------------------------------------
    d_in, V_in = random_squeezed_vacuum(args.modes, rng=rng)

    # ------------------------------------------------------------------
    # 2) Build a random instruction list for the interferometer
    # ------------------------------------------------------------------
    instructions = build_instructions(
        args.modes,
        topology,
        rng=rng,
        include_phases=False,
        embedded_total_modes=args.embedded_total_modes,
    )
    output_phases = rng.uniform(0.0, 2.0 * np.pi, size=args.modes)

    # ------------------------------------------------------------------
    # 3) Initialize GaussianDevice and compute "in" moments
    # ------------------------------------------------------------------
    input_device = GaussianDevice(d_in.copy(), V_in.copy(), instructions=())
    n_in = input_device.exp_photon_number()
    first_in = input_device.first_cumulants()
    second_in = input_device.second_cumulants()
    device = GaussianDevice.from_logical_state(
        d_in.copy(),
        V_in.copy(),
        instructions=instructions,
        topology=topology,
        embedded_total_modes=args.embedded_total_modes,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # 4) Apply the network and output phases, compute "out" moments
    # ------------------------------------------------------------------
    d_out, V_out = device.run(eta=args.eta, output_phases=output_phases, logical_output=True)
    output_device = GaussianDevice(d_out, V_out, instructions=())

    n_out = output_device.exp_photon_number()
    first_out = output_device.first_cumulants()
    second_out = output_device.second_cumulants()

    # ------------------------------------------------------------------
    # 5) Compute effective loss curve
    # ------------------------------------------------------------------
    etas, n_vac, curves = effective_loss_curve(args.modes, rng=rng)

    # ------------------------------------------------------------------
    # 6) Write logs and plot
    # ------------------------------------------------------------------
    logs_dir = PROJECT_ROOT / "demos" / "logs" / "demo_devices"
    plots_dir = PROJECT_ROOT / "demos" / "plots"
    _ensure_dir(logs_dir)
    _ensure_dir(plots_dir)

    log_path = logs_dir / f"demo_devices_{topology}_eta{int(round(args.eta * 100)):03d}.txt"
    _write_text(
        log_path,
        "\n".join(
            [
                f"topology={topology}",
                f"modes={args.modes}",
                f"eta={args.eta:.6f}",
                f"n_in={n_in:.17g}",
                f"first_cumulants_in={_format_array(first_in)}",
                f"second_cumulants_in={_format_array(second_in)}",
                f"n_out={n_out:.17g}",
                f"first_cumulants_out={_format_array(first_out)}",
                f"second_cumulants_out={_format_array(second_out)}",
                f"n_vac={n_vac:.17g}",
                f"etas={_format_array(etas)}",
                f"curve_Reck={_format_array(curves.get('Reck', np.array([])))}",
                f"curve_Clements={_format_array(curves.get('Clements', np.array([])))}",
            ]
        ),
    )

    plot_path = plots_dir / f"effective_loss_{topology}_eta{int(round(args.eta * 100)):03d}.png"
    _plot_loss_curve(etas, curves, plot_path)

    print("Wrote log:", log_path)
    print("Wrote plot:", plot_path)


if __name__ == "__main__":
    main()
