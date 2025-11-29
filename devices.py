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
        # validate_covariance(V_real, d_real)  # optional
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

    def first_moments(self, tol: float = 1e-15) -> np.ndarray:
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
                f"first_moments: imaginary component magnitude {max_imag:.3e} exceeds tol={tol}",
                RuntimeWarning,
            )
        return res

    def second_moments(self) -> np.ndarray:
        S = np.zeros((self.n, self.n), dtype=float)
        nbar = self.first_moments()
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
        Return the n x n unitary corresponding to the beamsplitter mesh only.

        This uses the same beamsplitter convention as ``interferometer``,
        so if `instructions` came from `triangle_decomposition` or
        `square_decomposition`, this matrix should match the internal mesh
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
            if isinstance(bs, tuple) and len(bs) == 4:
                k, l, theta, phi = bs
            else:
                k = getattr(bs, "k", getattr(bs, "i", getattr(bs, "mode1", None)))
                l = getattr(bs, "l", getattr(bs, "j", getattr(bs, "mode2", None)))
                theta = getattr(bs, "theta", getattr(bs, "angle", None))
                phi = getattr(bs, "phi", getattr(bs, "phase", 0.0))
            if k is None or l is None or theta is None:
                raise TypeError("Unexpected beamsplitter entry from interferometer decomposition.")
            instr.append((int(k - 1), int(l - 1), float(theta), float(phi)))

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
    n: int, topology: str, rng: np.random.Generator | None = None, include_phases: bool = False
) -> Tuple[Instruction, ...]:
    """Generate a random mesh of beamsplitter instructions for a target topology."""
    rng = rng or np.random.default_rng()
    instructions: list[Instruction] = []
    if topology.lower() == "reck":
        for k in range(n, 1, -1):
            for j in range(1, k):
                instructions.append((j - 1, j, rng.uniform(0.0, TWOPI)))
    elif topology.lower() == "clements":
        # simple random mesh: alternating layers
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
    else:
        raise ValueError(f"Unknown topology '{topology}'.")
    return tuple(instructions)


def effective_loss_curve(n: int, rng: np.random.Generator | None = None):
    """Compute expected photon number vs loss for random meshes."""
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
