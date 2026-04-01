from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_on_files


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value such as true/false, yes/no, or 1/0."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Top-level CLI for LoGIC helpers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run_on_files",
        help="Propagate covariance inputs from .mtx directories or supported single-file formats through unitary/symplectic data.",
    )
    run_parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input covariance files named input_cov###.mtx.",
    )
    run_parser.add_argument(
        "--unitary-dir",
        type=Path,
        help="Directory containing unitary/symplectic .mtx files.",
    )
    run_parser.add_argument(
        "--input-file",
        type=Path,
        help="Single non-.mtx input file (.json, .pickle, .npz, .h5py, .txt, or .wl) containing a time series of covariance matrices.",
    )
    run_parser.add_argument(
        "--unitary-file",
        type=Path,
        help="Single non-.mtx unitary/symplectic file (.json, .pickle, .npz, .h5py, .txt, or .wl) containing either one matrix or a time series of matrices.",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the requested output format is written.",
    )
    run_parser.add_argument(
        "--output-format",
        default="same",
        help=(
            "Requested output format. Use 'same' (default) to preserve the input file type in single-file mode. "
            "Otherwise choose one of: .mtx, .json, .pickle, .npz, .h5py, .txt, .wl "
            "(dot-prefixed and bare names such as 'json' are both accepted)."
        ),
    )
    run_parser.add_argument(
        "--eta",
        type=float,
        default=0.9,
        help="Loss transmissivity applied at each beam splitter.",
    )
    run_parser.add_argument(
        "--topology",
        type=str,
        default="Clements",
        choices=("Clements", "Reck", "embedded_reck", "clements", "reck", "embedded_reck_in_clements"),
        help="Beam splitter network topology.",
    )
    run_parser.add_argument(
        "--time-dependent",
        type=_parse_bool,
        default=False,
        metavar="BOOL",
        help="Whether to pair each input file with a matching unitary/symplectic file.",
    )
    run_parser.add_argument(
        "--embedded-total-modes",
        type=int,
        default=None,
        help="Optional total Clements mesh size for embedded_reck.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run_on_files":
        result = run_on_files(
            input_dir=args.input_dir,
            matrix_dir=args.unitary_dir,
            eta=args.eta,
            topology=args.topology,
            output_dir=args.output_dir,
            output_format=args.output_format,
            time_dependent_unitary=args.time_dependent,
            embedded_total_modes=args.embedded_total_modes,
            input_file=args.input_file,
            unitary_file=args.unitary_file,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
