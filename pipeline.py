from __future__ import annotations

import numpy as np
import interferometer as itf

from devices import GaussianDevice, validate_covariance


def _extract_decomposition(decomp_obj):
    """Handle both tuple and object returns from the interferometer package."""
    if isinstance(decomp_obj, tuple) and len(decomp_obj) == 2:
        return decomp_obj

    bs_local = (
        getattr(decomp_obj, "bs_list", None)
        or getattr(decomp_obj, "BS_list", None)
        or getattr(decomp_obj, "bs", None)
    )
    phases_local = (
        getattr(decomp_obj, "phases", None)
        or getattr(decomp_obj, "output_phases", None)
        or getattr(decomp_obj, "output_phase", None)
    )
    if bs_local is None or phases_local is None:
        raise TypeError("Unexpected return type from interferometer decomposition.")
    return bs_local, phases_local


def _normalize_instructions(bs_params, n_modes: int):
    """Convert interferometer tuples to (k, l, theta, phi) with 0-based indices."""
    instructions = []
    for entry in bs_params:
        # Accept both tuples and beamsplitter objects from the interferometer package.
        if hasattr(entry, "__len__"):
            if len(entry) == 4:
                k, l, theta, phi = entry
            elif len(entry) == 3:
                k, l, theta = entry
                phi = 0.0
            else:
                raise ValueError(f"Unexpected beamsplitter tuple: {entry}")
        else:
            k = getattr(entry, "k", getattr(entry, "i", getattr(entry, "mode1", None)))
            l = getattr(entry, "l", getattr(entry, "j", getattr(entry, "mode2", None)))
            theta = getattr(entry, "theta", getattr(entry, "angle", None))
            phi = getattr(entry, "phi", getattr(entry, "phase", 0.0))
            if k is None or l is None or theta is None:
                raise TypeError(f"Unexpected beamsplitter object: {entry!r}")

        # The interferometer package often uses 1-based indexing; shift if needed.
        k_int = int(k)
        l_int = int(l)
        if k_int >= n_modes or l_int >= n_modes:
            k_int -= 1
            l_int -= 1

        instructions.append((k_int, l_int, float(theta), float(phi or 0.0)))
    return instructions


def instructions_from_U(U: np.ndarray, topology: str) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    """Decompose a unitary into mesh instructions and output phases."""
    U = np.asarray(U, dtype=complex)
    n_modes = U.shape[0]

    topo = topology.lower()
    if topo == "clements":
        decomp_fn = itf.square_decomposition
    elif topo == "reck":
        decomp_fn = itf.triangle_decomposition
    else:
        raise ValueError("topology must be 'Clements' or 'Reck'.")

    bs_params_raw, phases = _extract_decomposition(decomp_fn(U))
    instructions = _normalize_instructions(bs_params_raw, n_modes)
    phases = np.asarray(phases, dtype=float)
    return instructions, phases


def get_Vout(
    U: np.ndarray,
    V0: np.ndarray,
    d0: np.ndarray | None = None,
    eta: float = 0.9,
    topology: str = "Clements",
    get_device: bool = False,
):
    """Propagate an input Gaussian state through a target unitary and loss channel."""
    U = np.asarray(U, dtype=complex)
    n_modes = U.shape[0]

    instructions, phases = instructions_from_U(U, topology)
    if d0 is None:
        d0 = np.zeros(V0.shape[0], dtype=float)

    validate_covariance(V0.copy(), d0.copy(), hbar=1, tol=1e-12)

    dev = GaussianDevice(d=d0.copy(), V=V0.copy(), instructions=instructions)
    dev.apply_network(eta=eta)
    if phases.size:
        dev.apply_output_phases(phases)

    d_out, V_out = dev.d, dev.V
    if get_device:
        return d_out, V_out, dev
    return d_out, V_out
