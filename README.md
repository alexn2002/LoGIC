# <ins>Lo</ins>ssy <ins>G</ins>aussian <ins>I</ins>nterferometer <ins>C</ins>omputation - <ins>LoGIC</ins>

Lightweight tools for propagating Gaussian states through programmable interferometers that allows simulation of internal, balanced photon loss. It wraps the [`interferometer`](https://pypi.org/project/interferometer/) package with a small `GaussianDevice` helper, plus ready-to-run demos for random meshes or matrix files.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run a basic simulation of a random squeezed input through a random unitary mesh:

```bash
python demos/demo_random.py --modes 4 --eta 0.9 --topology Clements --seed 123
```

Process covariance/unitary matrices from `demos/input_covariance_mtx` and `demos/interferometer_unitary_mtx`:

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

`get_Vout` validates the input covariance, decomposes the unitary with the chosen mesh (Clements or Reck), applies optional loss, and returns the output mean/covariance (and the device if `get_device=True`).

## Repository layout

- `devices.py` — core `GaussianDevice` class, covariance validation, random squeezed-state helpers, and mesh builders.
- `pipeline.py` — thin wrapper that decomposes a target unitary and feeds it through `GaussianDevice`.
- `demos/demo_random.py` — minimal CLI demo for random meshes.
- `demos/demo_literature.py` — CLI utility to process `.mtx` covariance/unitary files; writes results to `demos/output_covariance_mtx/`.
- `demos/input_covariance_mtx/`, `demos/interferometer_unitary_mtx/` — sample matrix inputs used by `demos/demo_literature.py`.

## Authors

- Alexander Naumann, Friedrich Schiller University Jena
- Robin Strahlendorf, Friedrich Schiller University Jena

## References

- Reck, Michael, et al. "Experimental realization of any discrete unitary operator." Physical Review Letters 73.1 (1994): 58.
- Clements, William R., et al. "Optimal design for universal multiport interferometers." Optica 3.12 (2016): 1460-1465.

## Notes

- Requires Python 3.10+ and the `interferometer` package (installed via `requirements.txt`).
- If you regenerate results, `demos/output_covariance_mtx/` will be overwritten; commit only the inputs you care about.
