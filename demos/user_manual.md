# LoGIC - Demos User Manual

This manual provides a detailed, user-friendly guide for the demo scripts. It is more complete than the README and aims to answer all common questions.

Contents
- demo_random.py (introductory: random unitary + random squeezed input)
- demo_literature.py (batch processing of input covariance matrices, optional Wolfram Language export)

-----------------------------------------------------------------------
demo_random.py
-----------------------------------------------------------------------

Purpose
Run a minimal Gaussian simulation: build a random Haar unitary, create a random squeezed vacuum input, propagate through a lossy interferometer, and print basic photon statistics.

Dependencies
- Internal: devices.py, pipeline.py
- External: numpy, scipy, interferometer

### How to use
Full example:

```
python ./demos/demo_random.py --eta 0.9 --topology Clements --modes 4 --seed 42
```

Minimal example (uses defaults):

```
python ./demos/demo_random.py --eta 0.9
```

Help:

```
python ./demos/demo_random.py --h
```

Flags and defaults
- --eta: loss channel transmissivity (default 0.9)
- --topology: Clements or Reck (default Clements)
- --modes: number of spatial modes (default 4)
- --seed: RNG seed for reproducibility (default 123)

### Code explanation
1) Build a Haar-random unitary U via QR decomposition:
   - Key lines:
     ```python
     U = _haar_unitary(args.modes, rng)
     ```
2) Build a random squeezed vacuum (d0, V0):
   - Key lines:
     ```python
     d0, V0 = random_squeezed_vacuum(args.modes, rng=rng)
     ```
3) Decompose U into a beamsplitter mesh and propagate the state:
   - The decomposition and propagation are handled inside get_Vout().
   - Key lines:
     ```python
     d_out, V_out, output_dev = get_Vout(U, V0, d0=d0, eta=args.eta, topology=args.topology, get_device=True)
     ```
4) Print photon-number diagnostics:
   - Key lines:
     ```python
     print("Input photon number:", float(np.real_if_close(input_dev.exp_photon_number())))
     print("Output photon number:", float(np.real_if_close(output_dev.exp_photon_number())))
     print("First moments:", output_dev.first_moments())
     print("Covariance shape:", V_out.shape)
     ```

### Output
The script prints:
- Mesh topology, modes, and eta
- Total photon number before and after the interferometer
- First moments (per-mode photon expectations)
- Covariance matrix shape

-----------------------------------------------------------------------
demo_literature.py
-----------------------------------------------------------------------

Purpose
Batch-process one or more input covariance matrices (.mtx files) through a fixed interferometer. This workflow is designed for reliable processing of many time steps.

Input and output folders (defaults)
- Input covariance directory:
  demos/input_covariance_mtx/
  Files are expected to be named input_cov001.mtx, input_cov002.mtx, ...
  Ordering is lexicographic (which matches numeric order with that naming).

- Symplectic directory:
  demos/interferometer_symplectic/
  Default symplectic file: symplectic.mtx

- Output covariance directory:
  demos/output_covariance_mtx/
  Outputs are stored in subfolders:
    demos/output_covariance_mtx/Reck/
    demos/output_covariance_mtx/Clements/

- Wolfram Language output directory:
  demos/output_covariance_wl/
  One file per topology per run.

### How to use
Full example:

```
python ./demos/demo_literature.py --eta 0.9 --input-dir ./demos/input_covariance_mtx --out-dir ./demos/output_covariance_mtx --symplectic-file demos/interferometer_symplectic/symplectic.mtx
```

Minimal example (uses defaults):

```
python ./demos/demo_literature.py --eta 0.9
```

Help:

```
python ./demos/demo_literature.py --h
```

Flags and defaults
- --in-dir / --input-dir: path to a single .mtx file or a directory of .mtx files.
  Default: ./demos/input_covariance_mtx
- --out-dir / --output-dir: output directory for generated .mtx results.
  Default: ./demos/output_covariance_mtx
- --symplectic-file / --process-mtx-file / --symplectic-mtx-file: explicit symplectic .mtx file to use.
  Default: ./demos/interferometer_symplectic/symplectic.mtx
- --eta / --loss: loss channel transmissivity (default 0.9)
- --no-symplectic: treat the unitary file as n x n (not 2n x 2n symplectic).

### Output
For each input covariance matrix:
- A lossy output covariance matrix (.mtx) for each topology:
  demos/output_covariance_mtx/Reck/LossyReck<stem>.mtx
  demos/output_covariance_mtx/Clements/LossyClements<stem>.mtx
- A text file with photon statistics:
  demos/output_covariance_mtx/<Topology>/moments_<Topology>_<stem>.txt
- A running summary file:
  demos/output_covariance_mtx/<Topology>/N_total.txt

Additionally, a Wolfram Language list file is created after the run:
- demos/output_covariance_wl/Reck_ETAetaXYZ.wl
- demos/output_covariance_wl/Clements_ETAetaXYZ.wl
where etaXYZ is computed from the eta value (e.g., eta=0.9 -> eta090).

The .wl file contains a list of all output matrices in the order processed.
Ordering is alphabetical by filename (which matches numeric order for input_cov001, input_cov002, ...).

### Code explanation
Step-by-step: compute_lossy_V()
This function is the core of demo_literature.py. It processes one input covariance at a time.

1) Read the input covariance V:
   - Uses scipy.io.mmread() on the provided .mtx file.
   - Key lines:
     ```python
     V = mmread(str(cov_path))
     V_eff = np.asarray(V)
     ```

2) Read the interferometer symplectic:
   - If --symplectic-file is not given, it loads:
     demos/interferometer_symplectic/symplectic.mtx
   - If --no-symplectic is NOT set:
     It assumes the file contains a 2n x 2n real symplectic matrix.
     It extracts the n x n unitary block U = X + iY.
   - If --no-symplectic IS set:
     It treats the file as an n x n complex unitary directly.
   - Key lines:
     ```python
     M = mmread(str(symplectic_path))
     ```

3) Decompose the unitary into beamsplitter instructions:
   - Uses the interferometer package decomposition for both Reck and Clements.
   - Converts the decomposition into a list of instructions:
     (mode_1, mode_2, theta, phi)
   - Key lines:
     ```python
     decomp = bs_decomp(unitary_block)
     instructions = _convert_beamsplitters(bs)
     ```

4) Build GaussianDevice and apply the network:
   - Computes a lossless output (eta=1.0) and a lossy output (eta=--eta).
   - Applies output phase shifts if provided by the decomposition.
   - Key lines:
     ```python
     device.apply_network(eta=1.0)
     lossy.apply_network(eta=eta_loss)
     ```

5) Save outputs:
   - Writes lossy covariance matrices to .mtx.
   - Writes photon statistics to text files.
   - Key lines:
     ```python
     mmwrite(res_dir / f"Lossy{label}{stem}.mtx", V_lossy)
     ```

Matrix conventions (symplectic vs unitary)
By default, the symplectic file is interpreted as a 2n x 2n real symplectic matrix:

U = X + iY  <->  S = [[ X, -Y ],
                     [ Y,  X ]]

If you pass --no-symplectic, the file is treated as a standard n x n unitary directly.

File format
All matrix inputs and outputs are in .mtx (Matrix Market) format.
