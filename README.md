# <ins>Lo</ins>ssy <ins>G</ins>aussian <ins>I</ins>nterferometer <ins>C</ins>omputation - <ins>LoGIC</ins>

[![DOI](https://zenodo.org/badge/1106338282.svg)](https://doi.org/10.5281/zenodo.18379681)

Lightweight tools for propagating Gaussian states through programmable interferometers that allows simulation of internal, balanced photon loss. It wraps the [`interferometer`](https://pypi.org/project/interferometer/) package with a small `GaussianDevice` helper, plus ready-to-run demos for beam splitter networks or matrix files.

**This project was developed to produce the data of Figure 7 in [D'Achille et al. 2026](#references) (unpublished; see Section VI Error Analysis, B Photon Loss).** You can find a preprint version [preprint version][paper] on the arXiv.

[paper]: https://arxiv.org/pdf/2506.23838

For details, see the [demo_literature.py](demos/demo_literature.py) section in the [user manual](demos/user_manual.md) and please read the [Disclaimer](#disclaimer) below.

## Quick start

Install from the local source tree:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The package can also be installed with:

```bash
pip install logic-jena
```

After installation, the Python API is available as:

```python
from logic.pipeline import run_on_files
```

and the CLI is available as:

```bash
logic run_on_files --help
```

Run a basic simulation of a random squeezed input through a random beam splitter network:

```bash
python demos/demo_pipeline.py --modes 4 --eta 0.9 --topology Clements --seed 123
```

In the Python API, `get_Vout(..., topology="embedded_reck")` is also supported. This embeds a Reck decomposition into a larger Clements mesh with explicit identity beam splitters and vacuum ancilla modes, then traces out the ancillas before returning the final state. The embedded mesh size can be chosen explicitly with `embedded_total_modes`, and defaults to the smallest validated size `2*modes - 2`.

Process covariance/symplectic matrices from `demos/input_covariance_mtx` and `demos/interferometer_symplectic`:

```bash
python demos/demo_literature.py --input-dir demos/input_covariance_mtx --eta 0.9
```

Results land in `demos/output_covariance_mtx/` (ignored by git).

The same workflow is also available through a top-level CLI for [`pipeline.run_on_files(...)`](pipeline.py). The wrapper supports the original `.mtx` directory workflow and a newer single-file workflow for `.json`, `.pickle`, `.npz`, `.h5py`, `.txt`, and `.wl` files.

```bash
python main.py run_on_files --input-dir demos/input_covariance_mtx --unitary-dir demos/interferometer_symplectic --output-dir demos/notebook_outputs --output-format wl --eta 0.9 --topology Clements --time-dependent false
```

Example for the single-file JSON workflow:

```bash
python main.py run_on_files --input-file tests/format_comp/inputs/series_input.json --unitary-file tests/format_comp/inputs/series_unitary.json --output-dir demos/notebook_outputs --output-format same --eta 0.9 --topology Clements --time-dependent true
```

In single-file mode, the covariance input is expected to contain a time series of the form `{{t1, cov1}, {t2, cov2}, ...}` and the unitary file may contain either one single matrix or a matching time series `{{t1, U1}, {t2, U2}, ...}`. By default, `output_format="same"` preserves the input file type. You can also request any supported output format explicitly: `.json`, `.pickle`, `.npz`, `.h5py`, `.txt`, or `.wl`. In `.mtx` directory mode, `output_format` can be `.mtx` or any of those aggregated single-file formats.

For `.txt` files, LoGIC automatically detects `json style`, `wolfram style`, or `matlab style` text and issues a warning because plain-text matrix files are more ambiguous than native structured formats. Unknown `.txt` styles raise a clear error. If a file has a `.json` suffix but begins with Wolfram-style braces, LoGIC automatically falls back to the Wolfram parser and warns that `.wl` or `.txt` would be a better suffix.

## API highlight

```python
import numpy as np
from pipeline import get_Vout
from devices import random_squeezed_vacuum

n_modes = 4
U = np.eye(n_modes)  # or any unitary of shape (n_modes, n_modes)
d0, V0 = random_squeezed_vacuum(n_modes)
d_out, V_out = get_Vout(U, V0, d0=d0, eta=0.9, topology="Clements")
```

`get_Vout` validates the input covariance, decomposes the unitary with the chosen beam splitter network (`"Clements"`, `"Reck"`, or `"embedded_reck"`), applies optional loss, and returns the output mean/covariance (and the device if `get_device=True`).

For `topology="embedded_reck"`, the code first computes a Reck decomposition, transforms it into an enlarged Clements-style instruction list with inserted identity beam splitters, prepends vacuum ancillas to the Gaussian input state, propagates the enlarged system, and finally traces out the ancilla modes so `d_out` and `V_out` again describe only the original logical modes. If `embedded_total_modes` is not provided, the code uses the smallest validated Clements mesh size `2*modes - 2`.

## Repository layout

- [`devices.py`](devices.py) — core [`GaussianDevice`](devices.py#L85) class, covariance validation, random squeezed-state helpers, and beamsplitter-network builders.
- [`pipeline.py`](pipeline.py) — thin wrapper that decomposes a target unitary and feeds it through [`GaussianDevice`](devices.py#L85).
- [`main.py`](main.py) — top-level CLI entry point, including `run_on_files`.
- [`demos/`](demos/) directory containing different code demonstrations and CLIs. Read the [user manual](demos/user_manual.md) for more information.
- [`demos/user_manual.md`](demos/user_manual.md) — supplementary sheet for the `demos/` directory
- [`demos/demo_devices.py`](demos/demo_devices.py) — introducionary demo for devices.py
- [`demos/demo_pipeline.py`](demos/demo_pipeline.py) — minimal CLI demo for random beam splitter networks.
- [`demos/demo_literature.py`](demos/demo_literature.py) — CLI utility to process `.mtx` covariance/symplectic files; writes results to [`demos/output_covariance_mtx/`](demos/output_covariance_mtx/).
- [`demos/input_covariance_mtx/`](demos/input_covariance_mtx/), [`demos/interferometer_symplectic/`](demos/interferometer_symplectic/) — sample matrix inputs used by [`demos/demo_literature.py`](demos/demo_literature.py).

## Authors
- [Alexander Naumann](https://mbqd.de/author/alexander-naumann/), Friedrich Schiller University Jena
- [Robin Strahlendorf](https://mbqd.de/author/robin-strahlendorf/), Friedrich Schiller University Jena

## Supervisors
- [Mauro D'Achille](https://mbqd.de/author/mauro-dachille/), Friedrich Schiller University Jena
- [Prof. Dr. Martin Gärttner](https://mbqd.de/author/martin-garttner/), Friedrich Schiller University Jena

## References

- [Reck, Michael, et al.](https://doi.org/10.1103/PhysRevLett.73.58) "Experimental realization of any discrete unitary operator." Physical Review Letters 73.1 (1994): 58.
- [Clements, William R., et al.](http://dx.doi.org/10.1364/OPTICA.3.001460) "Optimal design for universal multiport interferometers." Optica 3.12 (2016): 1460-1465.
- <ins>D'Achille, Mauro, et al.</ins> (unpublished) "Configurable photonic simulator for quantum field dynamics." arXiv preprint [arXiv:2506.23838](https://arxiv.org/pdf/2506.23838) (2025)

## Notes

- Requires Python 3.11+ and the packages listed in `requirements.txt`.
- The Python API and demo CLIs support `Clements`, `Reck`, and `embedded_reck`.
- `run_on_files(...)` supports `.mtx` directory mode plus single-file `.json`, `.pickle`, `.npz`, `.h5py`, and `.txt` workflows.
- If you regenerate results, `demos/output_covariance_mtx/` and `demos/logs/` will be overwritten; commit only the inputs you care about.
- The current PyPI prerelease version is `0.1.0rc3`, corresponding to the GitHub prerelease tag `v0.1.0-rc.3`.

## DISCLAIMER
We observed hardware-dependent numerical differences when simulating large beam splitter networks in finite precision. In particular, using the same code version of [demos/demo_literature.py](demos/demo_literature.py) on different machines produced output covariance matrices whose *difference* had a Frobenius norm on the order of 1e-2 for the 25×25 case. Repeated runs on the same machine did not show such deviations. These differences did not materially affect the information‑theoretic analysis in [D'Achille et al.][paper], for which this code was developed.

The discrepancy appears only in the lossy case. This is expected because the QR-based decomposition used to construct a beam splitter network from a target unitary is not unique. While different decompositions implement the same unitary in the lossless case, they can induce different effective loss channels, leading to different lossy outputs.

Although the PyPI `interferometer` package is deterministic, we observed machine‑dependent variation in its decomposition results, which likely explains the cross‑machine discrepancies. We have not yet identified a reliable fix or the specific routines within `interferometer` that cause this problem. Users should therefore interpret results from unitary‑decomposition‑based simulations of non‑unitary dynamics with care.

If you have suggestions to address this issue, please contact us at `alexander.naumann@uni-jena.de` or `robin.strahlendorf@uni-jena.de`.
