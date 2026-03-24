from __future__ import annotations

import numpy as np
import interferometer as itf

from devices import (
    GaussianDevice,
    add_vacuum_ancillas,
    embedded_reck_mode_count,
    reduce_gaussian_state,
    transform_instructions,
    validate_covariance,
)


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
    """Convert interferometer beamsplitters to internal 0-based instructions."""
    instructions = []
    for entry in bs_params:
        # Tuple-like returns may already be 0-based or 1-based depending on the
        # package path, so infer the convention from the values.
        if isinstance(entry, tuple):
            if len(entry) == 4:
                k, l, theta, phi = entry
            elif len(entry) == 3:
                k, l, theta = entry
                phi = 0.0
            else:
                raise ValueError(f"Unexpected beamsplitter tuple: {entry}")

            k_int = int(k)
            l_int = int(l)
            if k_int < 0 or l_int < 0:
                raise ValueError(f"Negative mode index in beamsplitter tuple: {entry}")
            if k_int >= n_modes or l_int >= n_modes:
                # 1-based tuples use the range 1..n.
                k_int -= 1
                l_int -= 1
        else:
            # Beamsplitter objects from interferometer use 1-based mode labels.
            k = getattr(entry, "k", getattr(entry, "i", getattr(entry, "mode1", None)))
            l = getattr(entry, "l", getattr(entry, "j", getattr(entry, "mode2", None)))
            theta = getattr(entry, "theta", getattr(entry, "angle", None))
            phi = getattr(entry, "phi", getattr(entry, "phase", 0.0))
            if k is None or l is None or theta is None:
                raise TypeError(f"Unexpected beamsplitter object: {entry!r}")

            k_int = int(k) - 1
            l_int = int(l) - 1

        if not (0 <= k_int < n_modes and 0 <= l_int < n_modes):
            raise ValueError(
                f"Beam splitter indices {(k_int, l_int)} are out of range for {n_modes} modes."
            )

        instructions.append((k_int, l_int, float(theta), float(phi or 0.0)))
    return instructions


def instructions_from_U(U: np.ndarray, topology: str) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    """Decompose a unitary into beamsplitter-network instructions and output phases."""
    U = np.asarray(U, dtype=complex)
    n_modes = U.shape[0]

    topo = topology.lower()
    if topo == "clements":
        decomp_fn = itf.square_decomposition
    elif topo == "reck":
        decomp_fn = itf.triangle_decomposition
    elif topo in {"embedded_reck", "embedded_reck_in_clements"}:
        embedded_n = embedded_reck_mode_count(n_modes)
        U_embedded = np.eye(embedded_n, dtype=complex)
        U_embedded[:n_modes, :n_modes] = U
        bs_params_raw, phases = _extract_decomposition(itf.square_decomposition(U_embedded))
        instructions = _normalize_instructions(bs_params_raw, embedded_n)
        phases = np.asarray(phases, dtype=float)
        return instructions, phases
    else:
        raise ValueError("topology must be 'Clements', 'Reck', or 'embedded_reck'.")

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
    topo = topology.lower()

    instructions, phases = instructions_from_U(U, topology)
    if d0 is None:
        d0 = np.zeros(V0.shape[0], dtype=float)

    validate_covariance(V0.copy(), d0.copy(), hbar=1, tol=1e-12)

    if topo in {"embedded_reck", "embedded_reck_in_clements"}:
        n_ancilla = embedded_reck_mode_count(n_modes) - n_modes
        d_in, V_in = add_vacuum_ancillas(d0.copy(), V0.copy(), n_ancilla)
    else:
        d_in, V_in = d0.copy(), V0.copy()

    dev = GaussianDevice(d=d_in, V=V_in, instructions=instructions)
    dev.apply_network(eta=eta)
    if phases.size:
        dev.apply_output_phases(phases)

    if topo in {"embedded_reck", "embedded_reck_in_clements"}:
        d_out, V_out = reduce_gaussian_state(dev.d, dev.V, range(n_modes))
    else:
        d_out, V_out = dev.d, dev.V

    if get_device:
        return d_out, V_out, dev
    return d_out, V_out
