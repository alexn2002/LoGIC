# LoGIC - demos
## demo_random.py

### How to use
Internal code dependencies: ```devices.py, pipeline.py```.

Please note: To change the available parameters of this demo (```eta``` - loss channel transmitivity, ```topology``` - the topology of the multimode interferometer (Clements or Reck), ```modes``` - the number of modes, ```seed``` - the seed for all random generators for reproducability) the user does not have to interfere with the code. Argument parsers are provided such that flags are available. **Run the following command in your terminal**

```
python ./demos/demo_random.py --eta 0.9 --topology Clements --modes 4 --seed 42
```

or minimal

```
python ./demos/demo_random.py --eta 0.9
```

for help run

```python
python ./demos/demo_random.py --h
```

change the parameter values as desired. This command alone is sufficient to run the demo.

### Code explanation
This python script sets up a Haar random unitary using 

```python
def _haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random n x n unitary via QR decomposition."""
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = sla.qr(z)
    diag = np.diag(r)
    phases = np.ones_like(diag)
    non_zero = np.abs(diag) > 0
    phases[non_zero] = diag[non_zero] / np.abs(diag[non_zero])
    return q * phases
        
U = _haar_unitary(args.modes, rng)
```
and a randomized squeezed state via
```python
from devices import GaussianDevice, random_squeezed_vacuum
[...]
d0, V0 = random_squeezed_vacuum(args.modes, rng=rng)
```
please note that ```d0``` will be a 1D zero array of appropiate length for squeezed vacuum.

Then we **set up a Gaussian interferometer** and **propagate the input state through**. The methods to build and simulate a Gaussian device are found in ```devices.py``` as methods of the python object ```GaussianDevice```, which can be used to build your own simulations on top. The GaussianDevice object stores the parameters of the GausianDevice according to some instruction, which is a list of tuples ``instructions=(mode_1, mode_2, transmitivity, phase``that tells the device the parameters of each beamsplitter and which two modes they couple, as well as ```phases``` the phases of single mode phase shifters before the beam splitter network. The instruction list can be generated for a random Clements or Reck beam splitter network via ```devices.build_instructions()```, an intruction list based on a QR decomposition (we refer to the PyPI package interferometer) of a target unitary can be generated with ```pipeline.instructions_from_U()```.

We demonstrate the initialization of GaussianDevice to store the initial state as reference

```python
input_dev = GaussianDevice(d0.copy(), V0.copy(), instructions=())
```

 **For convinient work** we offer a **ready-to-run pipeline** and refer to ```pipeline.py```. Within this demo we use the function ```pipeline.get_Vout()``` to compute the output


```python
from pipeline import get_Vout
[...]
d_out, V_out, output_dev = get_Vout(U, V0, d0=d0, eta=args.eta, topology=args.topology, get_device=True)
```

Finally, we present available methods for a simple photon number analysis of the resulting state. 

```python
    print(f"Mesh topology: {args.topology}, modes: {args.modes}, eta={args.eta}")
    print("Input photon number:", float(np.real_if_close(input_dev.exp_photon_number())))
    print("Output photon number:", float(np.real_if_close(output_dev.exp_photon_number())))
    print("First moments:", output_dev.first_moments())
    print("Covariance shape:", V_out.shape)
```

The GaussianDevice offers methods to investigate the expectation values and covariances of a PNR measure. Note that this computation can be done directly from the covariance matrix and is not hard. The current version **does not** offer a hafnian based PNR simulation for boson sampling.

### Output
This demos output is in the form of print outs produced by the previously referenced lines. They contain information about the implemented device (as check for the user) and the expected total photon number before and after the device (relevant if loss is present), as well as the single mode expectation values.

## demo_literature.py

This demo contains the code to generate the data for figure 7 in ```Mauro D'Archille et al. 2026``` (unpublished as of 22th Jan. 2026, citation will appear here).

### How to use
Internal code dependencies: ```devices.py```.

This demo offers an indepent workflow from pipeline.py which is intended to be convinient to work with but less introductory than the previous demo. Argument parsers are provided to change the input directory with ```--in-dir```, the output directory with ```--out-dir```, the loss channel transmitivity with ```--eta```, the target unitary according to which the multimode interferometer is constructed wtih ```-unitary-file```. **Run the following command in your terminal**

```
python ./demos/demo_literature.py --eta 0.9 --in-dir ./demos/input_covariance_mtx --out-dir ./demos/output_covariance_mtx --unitary-file demos/interferometer_unitary_mtx/matDFT25.mtx
```

or minimal

```
python ./demos/demo_literature.py --eta 0.9
```

for help run

```python
python ./demos/demo_literature.py --h
```

In the related literature multiple covariance matrices (representing different time steps of a physical system) had to be processed by the same multimode interferometer. Therefore, ```demo_literature``` accepts multiple ```.mtx``` files inside the input directory of ```--in-dir``` and will compute the output which is stored as an mtx file ```--out-dir``` for each input accordingly. 

Please Note: A ```.wl``` will be generated containing a list of all output matrices in the order they were processed (file name order) in wolfram language form. This file can be directly imported to wolfram alpha and was the data format used for follow computations within the related literature.

We also added an option ```--no-symplectic``` if the user decides to use all provided matrices in shape $ n\times n $ for bases $\{a_{i}\}_{i=1..n}$ instead of $2n \times 2n$ matrices of the symplectic formalism. Note that the $2n\times 2n$ interferometer symplectic $S$ and the $n\times n$ interferpmeter unitary $U$  is related via

$$ U = X + \mathrm{i}Y \Leftrightarrow S=
\begin{pmatrix}
    X & -Y\\
    Y & X
\end{pmatrix} $$

the code expects the file of ```--unitary-file``` to contain a $2n \times 2n$ symplectic by default.

**All matrix files are expected in ```.mtx``` format.**

## Code explanation

explain very user friendly what "compute_lossy_V" does step by step with code snippets. 