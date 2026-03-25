from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union
import warnings

import numpy as np
import scipy.linalg as sla

# numpy 2.x compatibility
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

TWOPI = 2.0 * math.pi

# Instructions can be (k, l, theta) or (k, l, theta, phi)
Instruction = Union[Tuple[int, int, float], Tuple[int, int, float, float]]


def validate_covariance(V: np.ndarray, d: np.ndarray, hbar: float = 1.0, tol: float = 1e-10) -> None:
    """Validate that (d, V) encodes a physical Gaussian state."""
    V = np.asarray(V, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if V.shape[0] % 2 != 0:
        raise ValueError("Covariance dimension must be even (2n x 2n).")
    if d.size != V.shape[0]:
        raise ValueError("Mean vector must match covariance dimension.")
    if not np.allclose(V, V.T, atol=tol, rtol=0):
        raise ValueError("Covariance matrix must be symmetric.")

    n = V.shape[0] // 2
    I = np.eye(n)
    Omega = np.block([[np.zeros_like(I), I], [-I, np.zeros_like(I)]])
    eigs = np.linalg.eigvalsh(V + 0.5j * hbar * Omega)
    if eigs.min().real < -tol:
        raise ValueError("Robertson-Schroedinger uncertainty bound violated.")


def embedded_reck_mode_count(n: int, total_modes: int | None = None) -> int:
    """Return the Clements mode count used to embed an n-mode Reck mesh."""
    if n < 1:
        raise ValueError("Number of modes must be positive.")
    minimum = 2 * n - 2 if n > 1 else 1
    if total_modes is None:
        return minimum
    if total_modes < minimum:
        raise ValueError(
            f"Embedded Reck on {n} logical modes does not fit into a Clements mesh with {total_modes} modes. "
            f"It requires at least {minimum} total modes so that floor(N/2) >= n-1."
        )
    return int(total_modes)


def _quadrature_indices(modes: Sequence[int], n_modes: int) -> np.ndarray:
    """Map mode indices to quadrature indices in [x_0..x_n-1, p_0..p_n-1] ordering."""
    modes_arr = np.asarray(modes, dtype=int).reshape(-1)
    if modes_arr.size == 0:
        return np.array([], dtype=int)
    if np.any(modes_arr < 0) or np.any(modes_arr >= n_modes):
        raise ValueError("Mode index out of range.")
    return np.concatenate([modes_arr, n_modes + modes_arr])


def add_vacuum_ancillas(
    d: np.ndarray,
    V: np.ndarray,
    n_ancilla: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend vacuum ancillas to a Gaussian state while preserving quadrature ordering."""
    if n_ancilla < 0:
        raise ValueError("Number of ancilla modes must be non-negative.")

    d = np.asarray(d, dtype=complex).reshape(-1)
    V = np.asarray(V, dtype=complex)
    n_phys = d.size // 2
    if n_ancilla == 0:
        return d.copy(), V.copy()

    n_total = n_phys + n_ancilla
    phys_quad = _quadrature_indices(range(n_ancilla, n_total), n_total)

    d_out = np.zeros(2 * n_total, dtype=complex)
    V_out = 0.5 * np.eye(2 * n_total, dtype=complex)

    d_out[phys_quad] = d
    V_out[np.ix_(phys_quad, phys_quad)] = V
    return d_out, V_out


def reduce_gaussian_state(
    d: np.ndarray,
    V: np.ndarray,
    modes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reduced Gaussian state obtained by tracing out all other modes."""
    d = np.asarray(d, dtype=complex).reshape(-1)
    V = np.asarray(V, dtype=complex)
    n_total = d.size // 2
    quad_idx = _quadrature_indices(modes, n_total)
    return d[quad_idx].copy(), V[np.ix_(quad_idx, quad_idx)].copy()


def _clements_mode_pairs(n: int) -> list[tuple[int, int]]:
    """Return the adjacent mode pairs in the column order of a square Clements mesh."""
    pairs: list[tuple[int, int]] = []
    for column in range(n):
        start = 0 if column % 2 == 0 else 1
        for k in range(start, n - 1, 2):
            pairs.append((k, k + 1))
    return pairs


# ---------------------------------------------------------------------
# Helpers: unitary and symplectic from a single beamsplitter
# ---------------------------------------------------------------------
def _bs_unitary_block(n: int, k: int, l: int, theta: float, phi: float) -> np.ndarray:
    """
    n-mode unitary with a single 2x2 beamsplitter acting on modes (k, l),
    using the same convention as the PyPI ``interferometer`` package:

        B(theta, phi) = [[ e^{i phi} cos theta,   -sin theta ],
                         [ e^{i phi} sin theta,    cos theta ]]

    embedded into an n x n identity.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    phase = np.exp(1j * phi)

    U = np.eye(n, dtype=complex)
    U[k, k] = c * phase
    U[k, l] = -s
    U[l, k] = s * phase
    U[l, l] = c
    return U


def _bs_symplectic(n: int, k: int, l: int, theta: float, phi: float) -> np.ndarray:
    """Symplectic representation of the beamsplitter matching `_bs_unitary_block`."""
    U = _bs_unitary_block(n, k, l, theta, phi)

    X = U.real
    Y = U.imag

    # Build full symplectic
    S = np.block([[X, -Y], [Y, X]])

    return S


# ---------------------------------------------------------------------
# GaussianDevice
# ---------------------------------------------------------------------
@dataclass
class GaussianDevice:
    """Gaussian state followed by a programmable interferometer with optional loss.

    The beamsplitter convention matches the PyPI ``interferometer`` package:
        B(theta, phi) = [[ e^{i phi} cos theta,   -sin theta ],
                         [ e^{i phi} sin theta,    cos theta ]]
    """

    d: np.ndarray
    V: np.ndarray
    instructions: Sequence[Instruction] = field(default_factory=tuple)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        d_real = np.asarray(self.d, dtype=float).reshape(-1)
        V_real = np.asarray(self.V, dtype=float)
        validate_covariance(V_real, d_real)
        self.n = V_real.shape[0] // 2
        # keep complex dtype to preserve phase information through the network
        self.d = np.asarray(self.d, dtype=complex).reshape(-1)
        self.V = np.asarray(self.V, dtype=complex)
        self.eta = 1.0

    # ------------------------------------------------------------------
    # State diagnostics
    # ------------------------------------------------------------------
    def exp_photon_number(self) -> float:
        val = 0.5 * (np.trace(self.V) + np.dot(self.d, self.d)) - 0.5 * self.n
        return float(np.real_if_close(val))

    def first_cumulants(self, tol: float = 1e-15) -> np.ndarray:
        """Return expected photon numbers per mode; warn if Im part exceeds tol."""
        res = np.zeros(self.n, dtype=float)
        max_imag = 0.0
        for k in range(self.n):
            val = (
                0.5
                * (
                    self.V[k, k]
                    + self.V[self.n + k, self.n + k]
                    + self.d[k] ** 2
                    + self.d[self.n + k] ** 2
                )
                - 0.5
            )
            max_imag = max(max_imag, abs(val.imag))
            res[k] = val.real
        if max_imag > tol:
            warnings.warn(
                f"first_cumulants: imaginary component magnitude {max_imag:.3e} exceeds tol={tol}",
                RuntimeWarning,
            )
        return res

    def second_cumulants(self) -> np.ndarray:
        S = np.zeros((self.n, self.n), dtype=float)
        nbar = self.first_cumulants()
        for k in range(self.n):
            for l in range(self.n):
                V_xx = self.V[k, l]
                V_pp = self.V[self.n + k, self.n + l]
                V_xp = self.V[k, self.n + l]
                V_px = self.V[self.n + k, l]

                a_a = 0.5 * (V_xx - V_pp + 1j * (V_xp + V_px))
                adag_a = 0.5 * (V_xx + V_pp + 1j * (V_xp - V_px))

                if k == l:
                    S[k, k] = nbar[k] ** 2 + nbar[k] + abs(a_a) ** 2
                else:
                    S[k, l] = (abs(adag_a) ** 2 + abs(a_a) ** 2).real
        return S

    # ------------------------------------------------------------------
    # Optical circuit application
    # ------------------------------------------------------------------
    def apply_beamsplitter(
        self, k: int, l: int, theta: float, eta: float = 1.0, phi: float = 0.0
    ) -> None:
        """
        Apply a single beamsplitter with (theta, phi) on modes (k, l),
        plus local loss eta on those modes.

        Beamsplitter convention matches ``interferometer``:
            B(theta, phi) = [[ e^{i phi} cos theta,   -sin theta ],
                             [ e^{i phi} sin theta,    cos theta ]]
        """
        S_bs = _bs_symplectic(self.n, k, l, theta, phi)

        # Apply passive BS symplectic first
        Sd = S_bs @ self.d
        SVS = S_bs @ self.V @ S_bs.T

        # Now apply loss (eta) on those modes (amplitude damping)
        A = np.eye(2 * self.n)
        for idx in (k, l, self.n + k, self.n + l):
            A[idx, idx] = math.sqrt(eta)

        Nmat = np.zeros_like(self.V)
        for idx in (k, l, self.n + k, self.n + l):
            Nmat[idx, idx] = (1.0 - eta) * 0.5

        self.d = A @ Sd
        self.V = A @ SVS @ A.T + Nmat

    def apply_network(
        self,
        eta: float = 1.0,
        output_phases: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """
        Apply the full interferometer network.

        * `instructions` specify only the internal beamsplitters.
        * `output_phases` are intentionally ignored here and should be applied
          via `apply_output_phases`.
        """
        for inst in self.instructions:
            if len(inst) == 3:
                k, l, theta = inst
                phi = 0.0
            elif len(inst) == 4:
                k, l, theta, phi = inst
                if phi is None:
                    phi = 0.0
            else:
                raise ValueError("Instructions must be length 3 or 4.")
            self.apply_beamsplitter(int(k), int(l), float(theta), eta=float(eta), phi=float(phi))

        self.eta = float(eta)
        # output_phases are deliberately ignored here; handled separately.

    def apply_output_phases(self, phases: Sequence[float]) -> None:
        """
        Apply single-mode phase shifters diag(e^{i phi_j}) to the current state.

        This acts in the quadrature basis as a 2x2 rotation on each mode:
            R(phi_j) = [[ cos phi_j, -sin phi_j ],
                        [ sin phi_j,  cos phi_j ]]
        """
        phases = np.asarray(phases, dtype=float)
        n = self.n
        S = np.eye(2 * n)
        for j, phi in enumerate(phases):
            c = math.cos(phi)
            s = math.sin(phi)
            x = j
            p = n + j
            S[x, x] = c
            S[x, p] = -s
            S[p, x] = s
            S[p, p] = c

        # Apply to first and second moments
        self.d = S @ self.d
        self.V = S @ self.V @ S.T

    def rescale(self, eta_override: float | None = None) -> None:
        eta_val = float(self.eta) if eta_override is None else float(eta_override)
        factor = float(eta_val ** self.n)
        self.V = self.V / factor - 0.5 * (1 - factor) / factor * np.eye(2 * self.n)

    def get_unitary(self) -> np.ndarray:
        """
        Return the n x n unitary corresponding to the beamsplitter network only.

        This uses the same beamsplitter convention as ``interferometer``,
        so if `instructions` came from `triangle_decomposition` or
        `square_decomposition`, this matrix should match the internal network
        unitary (up to output phases).
        """
        complex_needed = any(len(inst) == 4 and inst[3] not in (0, None) for inst in self.instructions)
        dtype = complex if complex_needed else float

        M = np.eye(self.n, dtype=dtype)
        for inst in self.instructions:
            if len(inst) == 3:
                k, l, theta = inst
                phi = 0.0
            elif len(inst) == 4:
                k, l, theta, phi = inst
                if phi is None:
                    phi = 0.0
            else:
                raise ValueError("Instruction entries must have length 3 or 4.")

            U_bs = _bs_unitary_block(self.n, int(k), int(l), float(theta), float(phi)).astype(dtype)
            M = U_bs @ M

        return M

    def _infer_topology(self) -> str:
        """Infer topology from instruction count."""
        L = len(self.instructions)
        n = self.n
        reck_len = n * (n - 1) // 2
        clements_len = n * (n - 1)
        return "Clements" if abs(L - clements_len) < abs(L - reck_len) else "Reck"

    def instr_from_M(
        self, M: np.ndarray
    ) -> tuple[float, float | None, list[tuple[int, int, float, float]], np.ndarray]:
        """
        Decompose a target matrix M into beamsplitter instructions and compare against this device.

        Returns (theta_norm_diff, phi_norm_diff, instr, phases).
        """
        import interferometer as itf

        M = np.asarray(M)
        if M.shape == (2 * self.n, 2 * self.n):
            X = M[: self.n, : self.n]
            Y = M[self.n :, : self.n]
            U = X + 1j * Y
        else:
            U = M.astype(complex)

        topo = self._infer_topology().lower()
        if topo == "clements":
            decomp = itf.square_decomposition(U)
        else:
            decomp = itf.triangle_decomposition(U)

        def _extract(decomp_obj):
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

        bs_list, phases = _extract(decomp)

        # Gauge align: remove output phases to match identity diagonal
        D = np.diag(np.exp(-1j * np.asarray(phases, dtype=complex)))
        U_gauged = U @ D
        decomp_gauged = itf.square_decomposition(U_gauged) if topo == "clements" else itf.triangle_decomposition(U_gauged)
        bs_list_g, phases_g = _extract(decomp_gauged)
        bs_list = bs_list_g
        phases = phases_g

        instr = []
        for bs in bs_list:
            if isinstance(bs, tuple):
                if len(bs) == 4:
                    k, l, theta, phi = bs
                elif len(bs) == 3:
                    k, l, theta = bs
                    phi = 0.0
                else:
                    raise TypeError("Unexpected beamsplitter tuple from interferometer decomposition.")

                k_int = int(k)
                l_int = int(l)
                if k_int < 0 or l_int < 0:
                    raise ValueError("Negative beamsplitter index returned by interferometer.")
                if k_int >= self.n or l_int >= self.n:
                    k_int -= 1
                    l_int -= 1
            else:
                k = getattr(bs, "k", getattr(bs, "i", getattr(bs, "mode1", None)))
                l = getattr(bs, "l", getattr(bs, "j", getattr(bs, "mode2", None)))
                theta = getattr(bs, "theta", getattr(bs, "angle", None))
                phi = getattr(bs, "phi", getattr(bs, "phase", 0.0))
                if k is None or l is None or theta is None:
                    raise TypeError("Unexpected beamsplitter entry from interferometer decomposition.")
                k_int = int(k) - 1
                l_int = int(l) - 1

            if not (0 <= k_int < self.n and 0 <= l_int < self.n):
                raise ValueError("Beamsplitter indices out of range after normalization.")
            instr.append((k_int, l_int, float(theta), float(phi)))

        def _angle_diff(a, b):
            return (a - b + math.pi) % (2 * math.pi) - math.pi

        theta_new = np.asarray([t[2] for t in instr], dtype=float)
        theta_old = np.asarray([t[2] for t in self.instructions], dtype=float)
        theta_diff = float(np.linalg.norm(_angle_diff(theta_new, theta_old)))

        phi_new = np.asarray([t[3] if len(t) > 3 and t[3] is not None else 0.0 for t in instr], dtype=float)
        phi_old = np.asarray([t[3] if len(t) > 3 and t[3] is not None else 0.0 for t in self.instructions], dtype=float)
        phi_diff = None
        if np.any(phi_new) or np.any(phi_old):
            phi_diff = float(np.linalg.norm(_angle_diff(phi_new, phi_old)))

        return theta_diff, phi_diff, instr, np.asarray(phases, dtype=float)

    # ------------------------------------------------------------------
    # Moment utilities
    # ------------------------------------------------------------------
    def n_over_eta(self, etas: Sequence[float]) -> np.ndarray:
        results = np.zeros(len(etas), dtype=float)
        d0 = self.d.copy()
        V0 = self.V.copy()
        for idx, eta in enumerate(etas):
            self.d = d0.copy()
            self.V = V0.copy()
            self.apply_network(eta=eta)
            results[idx] = self.exp_photon_number()
        self.d = d0
        self.V = V0
        return results


# ----------------------------------------------------------------------
# Squeezed vacuum + random instructions
# ----------------------------------------------------------------------
def _random_signed_r_values(
    n: int,
    rng: np.random.Generator,
    min_abs: float = 0.3,
    span: float = 0.5,
) -> np.ndarray:
    """Sample squeezing amplitudes with random signs and magnitudes."""
    magnitudes = rng.random(n) * span + min_abs
    signs = rng.choice([-1.0, 1.0], size=n)
    return signs * magnitudes


def squeezed_vacuum(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (d, V) for a squeezed vacuum with squeezing parameters z."""
    z = np.asarray(z, dtype=float).reshape(-1)
    n = z.size
    d = np.zeros(2 * n, dtype=float)
    S = sla.block_diag(np.diag(z), np.diag(1.0 / z))
    V = 0.5 * S @ S.T
    return d, V


def random_squeezed_vacuum(
    n: int,
    rng: np.random.Generator | None = None,
    min_abs: float = 0.3,
    span: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (d, V) for an n-mode squeezed vacuum with random squeezing parameters."""
    rng = rng or np.random.default_rng()
    r = _random_signed_r_values(n, rng, min_abs, span)
    z = np.exp(r)
    return squeezed_vacuum(z)


def build_instructions(
    n: int,
    topology: str,
    rng: np.random.Generator | None = None,
    include_phases: bool = False,
    embedded_total_modes: int | None = None,
) -> Tuple[Instruction, ...]:
    """Generate a random beamsplitter network for a target topology."""
    rng = rng or np.random.default_rng()
    instructions: list[Instruction] = []
    if topology.lower() == "reck":
        for k in range(n, 1, -1):
            for j in range(1, k):
                instructions.append((j - 1, j, rng.uniform(0.0, TWOPI)))
    elif topology.lower() == "clements":
        # simple random beamsplitter network: alternating layers
        for _ in range(n // 2):
            for k in range(0, n - 1, 2):
                if include_phases:
                    instructions.append((k, k + 1, rng.uniform(0.0, TWOPI), rng.uniform(0.0, TWOPI)))
                else:
                    instructions.append((k, k + 1, rng.uniform(0.0, TWOPI)))
            for k in range(1, n - 1, 2):
                if include_phases:
                    instructions.append((k, k + 1, rng.uniform(0.0, TWOPI), rng.uniform(0.0, TWOPI)))
                else:
                    instructions.append((k, k + 1, rng.uniform(0.0, TWOPI)))
        if n % 2:
            for k in range(0, n - 1, 2):
                if include_phases:
                    instructions.append((k, k + 1, rng.uniform(0.0, TWOPI), rng.uniform(0.0, TWOPI)))
                else:
                    instructions.append((k, k + 1, rng.uniform(0.0, TWOPI)))
    elif topology.lower() in {"embedded_reck", "embedded_reck_in_clements"}:
        reck_instructions: list[Instruction] = []
        for k in range(n, 1, -1):
            for j in range(1, k):
                if include_phases:
                    reck_instructions.append((j - 1, j, rng.uniform(0.0, TWOPI), rng.uniform(0.0, TWOPI)))
                else:
                    reck_instructions.append((j - 1, j, rng.uniform(0.0, TWOPI)))
        instructions = transform_instructions(reck_instructions, n, embedded_total_modes)
    else:
        raise ValueError(f"Unknown topology '{topology}'.")
    return tuple(instructions)


#----------------------------------------------------------------------
# Helpers: Transform from reck to clements instructions when embedding smaller Reck into larger Clements
# helper function, marks start index of each diagonal reck layer (recursively implemented)
def a(d, n):
    if d == 0:
        return 1
    else:
        return a(d-1, n) + (n-2) - (d-1) + 1
    
def my_sort(n):
    """
    provides the sorting mask generated from n (from Reck indexing) as a list of indices in the order they should be accessed for coloumn wise indexing
    example: [1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]
    """
    sorted_indices = []

    # For fixed d, the accessed term lies on diagonal j = d-k.
    # That diagonal has length n-1-j, so the largest valid offset is n-2-j.
    # Substituting j = d-k gives the correct bounds:
    #   2k     <= n-2-(d-k)  -> k <= n-2-d
    #   2k + 1 <= n-2-(d-k)  -> k <= n-3-d
    for d in range(n-1):
        kmax_even = min(d, n - 2 - d)
        for k in range(kmax_even, -1, -1):
            sorted_indices.append(a(d-k, n) + 2*k)

        kmax_odd = min(d, n - 3 - d)
        for k in range(kmax_odd, -1, -1):
            sorted_indices.append(a(d-k, n) + 2*k + 1)

    return sorted_indices


def reck_column_starts(n: int) -> list[int]:
    """Return the 1-based start indices of the upward-pointing Reck columns."""
    if n < 2:
        return []

    starts = list(range(1, n))
    current = n - 1
    for step in range(n - 2, 0, -1):
        current += step
        starts.append(current)
    return starts

def sort_and_insert(n, total_modes: int | None = None):
    """
    creates a flat list representing the n(n-1)//2 beam splitters of the reck scheme
    resorts them according to column wise indexing and inserts identity (marked as 0)
    to get a full clements network for the requested total mode count
    """
    instr = list(range(1, n*(n-1)//2 + 1))

    # Reorder according to the diagonal-wise / column-wise access pattern.
    sorted_instr = [instr[i - 1] for i in my_sort(n)]

    enlarged_n = embedded_reck_mode_count(n, total_modes)
    leading_empty_columns = 0 if enlarged_n % 2 == 0 else 1
    available_columns = enlarged_n - leading_empty_columns
    column_starts = reck_column_starts(n)

    # The start markers define the embedded Reck columns. For larger systems the
    # raw marker list can overshoot the target square Clements mesh by a few
    # trailing singleton columns, so merge those back into the last physical
    # column by dropping the extra terminal start markers.
    if len(column_starts) > available_columns:
        column_starts = column_starts[:available_columns]

    start_markers = set(column_starts[1:])
    embedded_columns = []
    current_column = []
    for value in sorted_instr:
        if current_column and value in start_markers:
            embedded_columns.append(current_column)
            current_column = [value]
        else:
            current_column.append(value)
    if current_column:
        embedded_columns.append(current_column)

    # Small even enlarged meshes need one final ghost column of identities.
    if leading_empty_columns:
        embedded_columns = [[] for _ in range(leading_empty_columns)] + embedded_columns

    while len(embedded_columns) < enlarged_n:
        embedded_columns.append([])

    enlarged_instr_list = []
    for i, column in enumerate(embedded_columns):
        if enlarged_n % 2 == 0:
            target_len = enlarged_n//2 if i % 2 == 0 else enlarged_n//2 - 1
        else:
            target_len = enlarged_n//2

        if len(column) > target_len:
            raise ValueError(
                f"embedded column {i} has length {len(column)}, exceeds target length {target_len}"
            )

        enlarged_instr_list.extend(["Id"] * (target_len - len(column)) + column)

    return enlarged_instr_list


def transform_instructions(
    reck_instructions: Sequence[Instruction],
    n: int,
    total_modes: int | None = None,
) -> list[tuple[int, int, float, float]]:
    """
    Embed an n-mode Reck instruction list into a larger Clements mesh.

    The returned instructions act on a Clements network of size
    ``embedded_reck_mode_count(n, total_modes)``. Identity beamsplitters are inserted with
    ``theta = phi = 0`` so later loss or drift models still see the full mesh.
    """
    expected_len = n * (n - 1) // 2
    if len(reck_instructions) != expected_len:
        raise ValueError(
            f"Expected {expected_len} Reck instructions for {n} modes, got {len(reck_instructions)}."
        )

    normalized: list[tuple[int, int, float, float]] = []
    for inst in reck_instructions:
        if len(inst) == 3:
            k, l, theta = inst
            phi = 0.0
        elif len(inst) == 4:
            k, l, theta, phi = inst
            phi = 0.0 if phi is None else phi
        else:
            raise ValueError("Instructions must be length 3 or 4.")
        normalized.append((int(k), int(l), float(theta), float(phi)))

    slot_mask = sort_and_insert(n, total_modes)
    embedded_n = embedded_reck_mode_count(n, total_modes)
    target_pairs = _clements_mode_pairs(embedded_n)
    if len(slot_mask) != len(target_pairs):
        raise ValueError(
            f"Embedding mask has {len(slot_mask)} slots, but Clements layout has {len(target_pairs)}."
        )

    transformed: list[tuple[int, int, float, float]] = []
    for slot, (k_target, l_target) in zip(slot_mask, target_pairs):
        if slot == "Id":
            transformed.append((k_target, l_target, 0.0, 0.0))
            continue

        _, _, theta, phi = normalized[int(slot) - 1]
        transformed.append((k_target, l_target, theta, phi))

    return transformed



def effective_loss_curve(n: int, rng: np.random.Generator | None = None):
    """Compute expected photon number vs loss for random beamsplitter networks."""
    rng = rng or np.random.default_rng(123)
    z = np.exp(-(rng.random(n) * 0.5 + 0.3))
    d, V = squeezed_vacuum(z)
    vacuum = GaussianDevice(d, V, instructions=())
    N_vac = vacuum.exp_photon_number()

    etas = 1 - np.logspace(-5, -0.5, 25)
    curves = {}
    for topo in ("Reck", "Clements"):
        device = GaussianDevice(d.copy(), V.copy(), instructions=build_instructions(n, topo, rng))
        curves[topo] = device.n_over_eta(etas)

    return etas, N_vac, curves
