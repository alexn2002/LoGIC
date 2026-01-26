# <ins>Lo</ins>ssy <ins>G</ins>aussian <ins>I</ins>nterferometer <ins>C</ins>omputation - <ins>LoGIC</ins>

Lightweight tools for propagating Gaussian states through programmable interferometers that allows simulation of internal, balanced photon loss. It wraps the [`interferometer`](https://pypi.org/project/interferometer/) package with a small `GaussianDevice` helper, plus ready-to-run demos for beam splitter networks or matrix files.

**This project was developed to produce the data of figure 7 in [D'Archille et al.][paper].**

[paper]: https://arxiv.org/pdf/2506.23838

For details, see the [demo_literature.py](demos/demo_literature.py) section in the [user manual](demos/user_manual.md) and please read the [Disclaimer](#disclaimer) below.

## Quick start

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Run a basic simulation of a random squeezed input through a random beam splitter network:

```bash
python demos/demo_pipeline.py --modes 4 --eta 0.9 --topology Clements --seed 123
```

Process covariance/symplectic matrices from `demos/input_covariance_mtx` and `demos/interferometer_symplectic`:

```bash
python demos/demo_literature.py --input-dir demos/input_covariance_mtx --eta 0.9
```

Results land in `demos/output_covariance_mtx/` (ignored by git).

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

`get_Vout` validates the input covariance, decomposes the unitary with the chosen beam splitter network (Clements or Reck), applies optional loss, and returns the output mean/covariance (and the device if `get_device=True`).

## Repository layout

- [`devices.py`](devices.py) — core [`GaussianDevice`](devices.py#L85) class, covariance validation, random squeezed-state helpers, and beamsplitter-network builders.
- [`pipeline.py`](pipeline.py) — thin wrapper that decomposes a target unitary and feeds it through [`GaussianDevice`](devices.py#L85).
- [`demos/`](demos/) directory containing different code demonstrations and CLIs. Read the [user manual](demos/user_manual.md) for more information.
- [`demos/user_manual.md`](demos/user_manual.md) — supplementary sheet for the `demos/` directory
- [`demos/demo_devices.py`](demos/demo_devices.py) — introducionary demo for devices.py
- [`demos/demo_pipeline.py`](demos/demo_pipeline.py) — minimal CLI demo for random beam splitter networks.
- [`demos/demo_literature.py`](demos/demo_literature.py) — CLI utility to process `.mtx` covariance/symplectic files; writes results to [`demos/output_covariance_mtx/`](demos/output_covariance_mtx/).
- [`demos/input_covariance_mtx/`](demos/input_covariance_mtx/), [`demos/interferometer_symplectic/`](demos/interferometer_symplectic/) — sample matrix inputs used by [`demos/demo_literature.py`](demos/demo_literature.py).

## Authors

- Alexander Naumann, Friedrich Schiller University Jena
- Robin Strahlendorf, Friedrich Schiller University Jena

## References

- Reck, Michael, et al. "Experimental realization of any discrete unitary operator." Physical Review Letters 73.1 (1994): 58.
- Clements, William R., et al. "Optimal design for universal multiport interferometers." Optica 3.12 (2016): 1460-1465.

## Notes

- Requires Python 3.10+ and the `interferometer` package (installed via `requirements.txt`).
- If you regenerate results, `demos/output_covariance_mtx/` and `demos/logs/` will be overwritten; commit only the inputs you care about.

## DISCLAIMER
We observed hardware-dependent numerical differences when simulating large beam splitter networks in finite precision. In particular, using the same code version of [demos/demo_literature.py](demos/demo_literature.py) on different machines produced output covariance matrices whose *difference* had a Frobenius norm on the order of 1e-2 for the 25×25 case. Repeated runs on the same machine did not show such deviations. These differences did not materially affect the information‑theoretic analysis in [D'Archille et al.][paper], for which this code was developed.

The discrepancy appears only in the lossy case. This is expected because the QR-based decomposition used to construct a beam splitter network from a target unitary is not unique. While different decompositions implement the same unitary in the lossless case, they can induce different effective loss channels, leading to different lossy outputs.

Although the PyPI `interferometer` package is deterministic, we observed machine‑dependent variation in its decomposition results, which likely explains the cross‑machine discrepancies. We have not yet identified the root cause or a reliable fix. Users should therefore interpret results from unitary‑decomposition‑based simulations of non‑unitary dynamics with care.

If you have suggestions to address this issue, please contact us at `alexander.naumann@uni-jena.de` or `robin.strahlendorf@uni-jena.de`.
