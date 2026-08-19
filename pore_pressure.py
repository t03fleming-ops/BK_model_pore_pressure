"""1D pore-pressure diffusion (Crank–Nicolson).

This module provides a `PorePressure` helper class for simulating diffusion
(e.g. pressure equilibration) on a 1D grid using an implicit Crank–Nicolson
scheme. The implicit step is stable for time steps where explicit methods
blow up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spla


def gaussian_press(x: np.ndarray, injection_pressure: float, mu: float, sigma: float = 1.0) -> np.ndarray:
    gaussian = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gaussian * injection_pressure


def build_1d_laplacian_matrix(n: int):
    """Sparse matrix matching `fault.laplacian` for a 1D vector of length `n`.

    The corresponding operator has:
    - interior:   u[i+1] - 2u[i] + u[i-1]
    - boundaries: u[1] - u[0]  and  u[n-2] - u[n-1]
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    main = -2.0 * np.ones(n)
    main[0] = -1.0
    main[-1] = -1.0
    off = np.ones(n - 1)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csc")


@dataclass
class _LinearStepperCache:
    dt: float
    lhs_factor: Callable[[np.ndarray], np.ndarray]
    rhs: "sp.csc_matrix"


class PorePressure:
    """1D diffusion model for pore pressure.

    Parameters mirror the notebook implementation currently in `c_l_faultsim_rt.ipynb`.
    `n_blocks` is treated as a *domain length*, with grid spacing `res`.

    When `hold_injection_pressure=True`, the node at `inj_site` is clamped to
    `injection_pressure` at every time step (Dirichlet condition).
    """

    def __init__(
        self,
        n_blocks: float,
        injection_pressure: float,
        form: str = "Gaussian",
        diff_coeff: float = 5.0,
        res: float = 0.001,
        hold_injection_pressure: bool = False,
    ):
        if res <= 0:
            raise ValueError(f"res must be positive, got {res}")
        if diff_coeff < 0:
            raise ValueError(f"diff_coeff must be non-negative, got {diff_coeff}")
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}")

        self.injection_pressure = float(injection_pressure)
        self.hold_injection_pressure = bool(hold_injection_pressure)
        self.diff_coeff = float(diff_coeff)
        self.res = float(res)
        self.n_blocks = float(n_blocks)

        form_dict: Dict[str, Callable[..., np.ndarray]] = {"Gaussian": gaussian_press}
        if form not in form_dict:
            raise ValueError(f"form must be one of {sorted(form_dict.keys())}, got {form!r}")
        self.form = form_dict[form]

        self.x = np.arange(0.0, self.n_blocks+self.res, self.res, dtype=float)
        if self.x.size < 3:
            raise ValueError(
                f"Grid too small: len(x)={self.x.size}. Increase n_blocks or decrease res."
            )

        self.inj_site = int(self.x.size // 2)
        self.Pressure = self.form(self.x, self.injection_pressure, self.x[self.inj_site], self.res)
        if self.hold_injection_pressure:
            self.Pressure[self.inj_site] = self.injection_pressure

        self._lap = None
        self._A = None
        self._cn_cache: Optional[_LinearStepperCache] = None

    def _ensure_operator(self) -> None:
        if self._A is not None:
            return
        self._lap = build_1d_laplacian_matrix(self.x.size)
        self._A = (self.diff_coeff / (self.res**2)) * self._lap

    def _get_crank_nicolson(self, dt: float) -> _LinearStepperCache:
        self._ensure_operator()
        if self._cn_cache is not None and np.isclose(self._cn_cache.dt, dt):
            return self._cn_cache

        I = sp.identity(self.x.size, format="csc")
        lhs = (I - (dt / 2.0) * self._A).tocsc()
        rhs = (I + (dt / 2.0) * self._A).tocsc()

        if self.hold_injection_pressure:
            # Dirichlet constraint at the injection site:
            #   P^{n+1}[inj_site] = injection_pressure
            k = int(self.inj_site)
            lhs = lhs.tolil()
            rhs = rhs.tolil()
            lhs[k, :] = 0.0
            lhs[k, k] = 1.0
            rhs[k, :] = 0.0
            rhs[k, k] = 1.0 
            lhs = lhs.tocsc()
            rhs = rhs.tocsc()

        lhs_factor = spla.factorized(lhs)
        self._cn_cache = _LinearStepperCache(dt=float(dt), lhs_factor=lhs_factor, rhs=rhs)
        return self._cn_cache

    def set_hold_injection_pressure(self, hold: bool) -> None:
        """Enable/disable the Dirichlet constraint and clear cached factorizations."""
        self.hold_injection_pressure = bool(hold)
        self._cn_cache = None

    def crank_nicolson_step(self, dt: float, P: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one time step with Crank–Nicolson (implicit, 2nd order)."""
        if P is None:
            P = self.Pressure
        if not np.isfinite(P).all():
            raise FloatingPointError("Non-finite pressure state.")

        cache = self._get_crank_nicolson(dt)
        b = cache.rhs @ P
        self.Pressure = cache.lhs_factor(b)
        return self.Pressure


    def simulate(self, t0: float, tn: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate and return `(times, pressure_history)`."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        times = np.arange(t0, tn + 1E-15, dt, dtype=float)
        hist = np.empty((self.x.size, times.size), dtype=float)
        hist[:, 0] = self.Pressure
        for i in range(1, times.size):
            hist[:, i] = self.crank_nicolson_step(dt)
        return times, hist
