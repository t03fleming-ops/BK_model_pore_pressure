"""Burridge–Knopoff spring-block fault model utilities.

This module contains small numerical utilities (friction laws, a 1D discrete
Laplacian, plotting helpers) and a `Fault` class that integrates a simplified
spring-block (Burridge–Knopoff / Carlson–Langer style) model forward in time.
"""
from typing import Optional

from numba import jit
from numpy import ndarray
import numpy as np

from scipy.integrate import LSODA
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pore_pressure import build_1d_laplacian_matrix

@jit(nopython=True, parallel=True)
def slip_friction(x, x0, alpha: float = 2.5, sigma: float = 0.01):
    """Slip weakening friction function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Current displacement.
    x0 : numpy.array or float
        The position prior to the beginning of slip.
    alpha : float
        Parameter to control the strength of friction weaken.
    sigma : float
        Instantaneous drop in friction. Must be in (0, 1).

    Returns
    -------
    numpy.ndarray or float
        Friction force (dimensionless), same shape as ``x``.

    Raises
    ------
    ValueError
        If sigma is not in the valid range (0, 1).
    """
    if sigma <= 0 or sigma >= 1:
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")

    return (1-sigma) / (1 + (alpha/(1-sigma)) * (x-x0))

@jit(nopython=True, parallel=True)
def phi(v, sigma=0.01):
    """Velocity-weakening friction function used by the model.

    Parameters
    ----------
    v : numpy.ndarray or float
        Slip velocity (typically non-negative in this model).
    sigma : float, optional
        Dimensionless friction drop parameter in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray or float
        Friction force (dimensionless), same shape as ``v``.

    Raises
    ------
    ValueError
        If sigma is not in the valid range (0, 1).
    """
    if sigma <= 0 or sigma >= 1:
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")

    return (1.0 - sigma) / (1.0 + v / (1.0 - sigma))

@jit(nopython=True, parallel=True)
def laplacian(U):
    """Compute the 1D discrete Laplacian (second difference) of ``U``.

    The interior uses the centered second difference. The boundary entries use
    a one-sided approximation consistent with "free" ends (no ghost points).

    Parameters
    ----------
    U : numpy.ndarray
        1D array of displacements (or any scalar field sampled on blocks).
        Must have at least 2 elements.

    Returns
    -------
    numpy.ndarray
        1D array of the same shape as ``U`` containing the discrete Laplacian.

    Raises
    ------
    ValueError
        If U has fewer than 2 elements.
    """
    if len(U) < 2:
        raise ValueError(f"U must have at least 2 elements, got {len(U)}")

    lap = np.zeros_like(U)
    if len(U) > 2:
        lap[1:-1] = U[2:] - 2 * U[1:-1] + U[:-2]
    lap[0] = U[1] - U[0]
    lap[-1] = U[-2] - U[-1]
    return lap

@jit(nopython=True, parallel=True)
def laplacian_mat(U, lap_mat=None):
    """Compute the 1D discrete Laplacian of ``U`` using matrix multiplication.

    Uses a precomputed Laplacian matrix or builds one if not provided. This is
    an alternative to the finite-difference implementation in :func:`laplacian`.

    Parameters
    ----------
    U : numpy.ndarray
        1D array of displacements (or any scalar field sampled on blocks).
        Must have at least 2 elements.
    lap_mat : scipy.sparse matrix, optional
        Precomputed 1D Laplacian matrix. If None, builds a new matrix using
        :func:`build_1d_laplacian_matrix`.

    Returns
    -------
    numpy.ndarray
        1D array of the same shape as ``U`` containing the discrete Laplacian.

    Raises
    ------
    ValueError
        If U has fewer than 2 elements or if lap_mat shape is incompatible with U.
    """
    if len(U) < 2:
        raise ValueError(f"U must have at least 2 elements, got {len(U)}")

    if lap_mat is None:
        lap_mat = build_1d_laplacian_matrix(int(len(U)))

    if lap_mat.shape[1] != len(U):
        raise ValueError(f"lap_mat shape {lap_mat.shape} incompatible with U length {len(U)}")

    return lap_mat @ U

class Fault:
    """1D Burridge–Knopoff spring-block fault model.

    The model state is stored as ``self.state`` with shape ``(2, n_blocks)``,
    where row 0 is a displacement ``U`` and row 1 is velocity ``v``.
    """
    def __init__(self,
                 n_blocks: int,
                 l2 : Optional[float] = None,
                 alpha :float = 2.5,
                 pll_spd: float = 0.001,
                 damping_coef : float = 0,
                 block_spacing : Optional[float] = 1,
                 p_ratio: Optional[float] = 0,
                 evl_PorePressure: bool = False,
                 normal_pressure: Optional[float] = None,
                 injection_pressure: Optional[float] = None,
                 ):
        """Create a fault model instance.

        Parameters
        ----------
        n_blocks : int
            Number of blocks in the 1D chain.
        l2 : float, optional
            Dimensionless elastic coupling (stiffness) between neighboring
            blocks. If None, continuum models must be used.
        alpha : float, optional
            Dimensionless friction/velocity scaling parameter used in
            :func:`phi`. Default is 2.5.
        pll_spd : float, optional
            Dimensionless loading (plate) speed. Default is 0.001.
        damping_coef : float, optional
            Damping coefficient for viscous damping term. Default is 0.
        block_spacing : float, optional
            Spatial spacing between blocks. Default is 1.
        p_ratio : float, optional
            Pore pressure ratio for constant pore pressure models. Default is 0.
        evl_PorePressure : bool, optional
            Whether to evolve pore pressure dynamically. Default is False.
        normal_pressure : float, optional
            Normal stress on the fault. Required if injection_pressure is specified.
        injection_pressure : float, optional
            Injection pressure for pore pressure evolution. Required if normal_pressure
            is specified.
        """

        try:
            n_blocks = int(n_blocks)
        except (ValueError, TypeError):
            raise ValueError(f'n_blocks must be an integer at least 1, got {n_blocks}')

        if n_blocks < 1:
            raise ValueError(f'n_blocks must be at least 1, got {n_blocks}')

        params = {
            "n_blocks": n_blocks,
            "l2": l2,
            "alpha": alpha,
            "pll_spd": pll_spd,
        }

        for name, val in params.items():
            if name == "l2" and val is None:
                print("l2 not set, ensure you simulate with one of the continuum models")
                continue
            if not isinstance(val, (int, float, np.number)):
                raise TypeError(f"{name} must be numeric, got {type(val)}")
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        if damping_coef < 0:
            raise ValueError(f"damping_coef must be non-negative, got {damping_coef}")

        if block_spacing is not None and block_spacing <= 0:
            raise ValueError(f"block_spacing must be positive, got {block_spacing}")

        if p_ratio is not None and p_ratio < 0:
            raise ValueError(f"p_ratio must be non-negative, got {p_ratio}")
        if injection_pressure is not None or normal_pressure is not None:
            if normal_pressure is None:
                raise ValueError("normal_pressure must be specified with injection_pressure")
            if injection_pressure is None:
                raise ValueError("injection_pressure must be specified with normal_pressure")
            if normal_pressure <= 0:
                raise ValueError(f"normal_pressure must be positive, got {normal_pressure}")
            if injection_pressure < 0:
                raise ValueError(f"injection_pressure must be non-negative, got {injection_pressure}")





        self.l2 = l2
        self.pll_spd = pll_spd
        self.n_blocks = n_blocks
        self.alpha = alpha
        self.initial_state = None
        self.n = damping_coef
        self.damping_frc = []
        self.block_spacing = block_spacing
        self.lap_mat = build_1d_laplacian_matrix(n_blocks).tocsc()
        self.p_ratio = p_ratio
        self.normal_pressure = normal_pressure
        self.inj_preasure = injection_pressure
        self.pore_pressure = None

        if evl_PorePressure:
            from pore_pressure import PorePressure
            PorePressure = PorePressure(
                n_blocks=self.n_blocks,
                injection_pressure=self.inj_preasure,
                form="Gaussian",
                diff_coeff=5.0,
                res=0.001,
                hold_injection_pressure=False,
            )
            self.pore_pressure = PorePressure
            self.U_indxs = np.arange(0, self.n_blocks + 1, 1)




    def Discrete_vel_weakening(self, t, U_dU: np.ndarray):
        """Right-hand side of the (dimensionless) Burridge–Knopoff ODE system.

        This implementation follows a Carlson–Langer style stick-slip rule:
        when a block is "sticking" (velocity non-positive and shear <= 1), its
        velocity is clamped to 0; otherwise it slips with acceleration given by
        elastic shear minus velocity-dependent friction.

        Parameters
        ----------
        t : float
            Current time.
        U_dU : numpy.ndarray
            Current state array shaped ``(2, n_blocks)`` where row 0 is
            displacement and row 1 is velocity.

        Returns
        -------
        numpy.ndarray
            Time derivative shaped ``(2, n_blocks)``: ``[dU/dt, dv/dt]``.

        Raises
        ------
        ValueError
            If U_dU shape is incorrect or if l2 is not set.
        """
        if U_dU.shape != (2, self.n_blocks):
            raise ValueError(f"U_dU must have shape (2, {self.n_blocks}), got {U_dU.shape}")

        if self.l2 is None:
            raise ValueError("l2 must be set for discrete models. Use continuum models instead.")

        if self.pore_pressure is None:
            raise ValueError("pore_pressure must be initialized for this ODE formulation")

        if self.normal_pressure is None:
            raise ValueError("normal_pressure must be set for this ODE formulation")

        U = U_dU[0, :]
        v = U_dU[1, :]

        shear = self.l2 * laplacian(U) + self.pll_spd * t - U
        d2u_dt2 = np.zeros_like(U)
        stick = (v <= 0) & (shear <= 1.0)
        v[stick & (v < 0)] = 0
        slip = ~stick

        d2u_dt2[slip] = shear[slip] - np.sign(v[slip]) * phi(2 * self.alpha * np.abs(v[slip]))
        self.damping_frc.append([shear, d2u_dt2])
        return np.vstack((v, d2u_dt2))

    def Discrete_vel_weakening_evl_pp(self, t, U_dU: np.ndarray):
        """Right-hand side of the Burridge–Knopoff ODE with evolving pore pressure.

        This implementation follows a Carlson–Langer style stick-slip rule with
        spatially-varying pore pressure that modifies the effective normal stress.
        Blocks stick when velocity is non-positive and shear <= |f0| (where f0
        depends on local pore pressure). During slip, friction is scaled by |f0|.

        Parameters
        ----------
        t : float
            Current time.
        U_dU : numpy.ndarray
            Current state array shaped ``(2, n_blocks)`` where row 0 is
            displacement and row 1 is velocity.

        Returns
        -------
        numpy.ndarray
            Time derivative shaped ``(2, n_blocks)``: ``[dU/dt, dv/dt]``.

        Raises
        ------
        ValueError
            If U_dU shape is incorrect, if l2 is not set, or if pore pressure is not initialized.
        """
        if U_dU.shape != (2, self.n_blocks):
            raise ValueError(f"U_dU must have shape (2, {self.n_blocks}), got {U_dU.shape}")

        if self.l2 is None:
            raise ValueError("l2 must be set for discrete models. Use continuum models instead.")

        if self.pore_pressure is None:
            raise ValueError("pore_pressure must be initialized for this ODE formulation")

        if self.normal_pressure is None:
            raise ValueError("normal_pressure must be set for this ODE formulation")

        U = U_dU[0, :]
        v = U_dU[1, :]

        ##way of going from pore pressure array to the pressure at each block location
        #I will use a function

        fp = self.pore_pressure.Pressure
        pp_at_blck =  np.interp(self.U_indxs*self.block_spacing + U, self.pore_pressure.x, fp)


        f0 = pp_at_blck/self.normal_pressure - 1

        shear = self.l2 * laplacian(U) + self.pll_spd * t - U
        d2u_dt2 = np.zeros_like(U)
        stick = (v <= 0) & (shear <= abs(f0))
        v[stick & (v < 0)] = 0
        slip = ~stick

        d2u_dt2[slip] = shear[slip] - np.sign(v[slip]) * abs(f0) * phi(2 * self.alpha * np.abs(v[slip]))
        self.damping_frc.append([shear, d2u_dt2])
        return np.vstack((v, d2u_dt2))


    def Continuum_vel_weakening_damped_cnst_pp(self, t, U_dU: np.ndarray):
        """Right-hand side of continuum-limit ODE with damping and constant pore pressure.

        Uses continuum scaling (Laplacian scaled by 1/block_spacing²) with velocity
        damping and constant pore pressure ratio. Stick-slip behavior is modified by
        f0 = p_ratio - 1, where p_ratio is the pore-to-normal pressure ratio.

        Parameters
        ----------
        t : float
            Current time.
        U_dU : numpy.ndarray
            Current state array shaped ``(2, n_blocks)`` where row 0 is
            displacement and row 1 is velocity.

        Returns
        -------
        numpy.ndarray
            Time derivative shaped ``(2, n_blocks)``: ``[dU/dt, dv/dt]``.

        Raises
        ------
        ValueError
            If U_dU shape is incorrect or if block_spacing is not set.
        """
        if U_dU.shape != (2, self.n_blocks):
            raise ValueError(f"U_dU must have shape (2, {self.n_blocks}), got {U_dU.shape}")

        if self.block_spacing is None or self.block_spacing <= 0:
            raise ValueError(f"block_spacing must be positive for continuum models, got {self.block_spacing}")

        U = U_dU[0, :]
        v = U_dU[1, :]

        f0 = self.p_ratio - 1

        shear = (1 / self.block_spacing ** 2) * laplacian(U) + self.pll_spd * t - U
        d2u_dt2 = np.zeros_like(U)
        stick = (v <= 0) & (shear <= abs(f0))
        v[stick & (v < 0)] = 0
        slip = ~stick

        d2u_dt2[slip] = shear[slip] - np.sign(v[slip]) * abs(f0) * phi(2 * self.alpha * np.abs(v[slip])) + (self.n/self.block_spacing**2)*laplacian(v)[slip]
        return np.vstack((v, d2u_dt2))

    def Continuum_vel_weakening_damped(self, t, U_dU: np.ndarray):
        """Right-hand side of continuum-limit ODE with velocity damping.

        Uses continuum scaling (Laplacian scaled by 1/block_spacing²) and includes
        a velocity damping term proportional to the Laplacian of velocity. Standard
        Carlson–Langer stick-slip with threshold shear stress of 1.0.

        Parameters
        ----------
        t : float
            Current time.
        U_dU : numpy.ndarray
            Current state array shaped ``(2, n_blocks)`` where row 0 is
            displacement and row 1 is velocity.

        Returns
        -------
        numpy.ndarray
            Time derivative shaped ``(2, n_blocks)``: ``[dU/dt, dv/dt]``.
        """
        if U_dU.shape != (2, self.n_blocks):
            raise ValueError(f"U_dU must have shape (2, {self.n_blocks}), got {U_dU.shape}")

        if self.block_spacing is None or self.block_spacing <= 0:
            raise ValueError(f"block_spacing must be positive for continuum models, got {self.block_spacing}")

        U = U_dU[0, :]
        v = U_dU[1, :]
        shear = (1/self.block_spacing**2) * laplacian(U) + self.pll_spd * t - U
        d2u_dt2 = np.zeros_like(U)
        stick = (v <= 0) & (shear <= 1.0)
        v[stick & (v < 0)] = 0
        slip = ~stick

        d2u_dt2[slip] = shear[slip] - np.sign(v[slip]) * phi(2 * self.alpha * np.abs(v[slip])) + (self.n/self.block_spacing**2)*laplacian(v)[slip]
        #self.damping_frc.append([abs(-2 * np.sqrt(self.l2) * v)*n,
        #                        abs(shear),
        #                         phi_nef(2 * self.alpha * v),
        #                         v])

        return np.vstack((v, d2u_dt2))


    def Discrete_slip_weakening(self, t, U_dU: np.ndarray):
        """Right-hand side of Burridge–Knopoff ODE with slip-weakening friction.

        Uses Carlson–Langer stick-slip logic but replaces velocity-weakening friction
        with slip-weakening friction. Friction depends on total slip (U - U0) since
        the block began slipping, where U0 is updated each time a block sticks.

        Parameters
        ----------
        t : float
            Current time.
        U_dU : numpy.ndarray
            Current state array shaped ``(2, n_blocks)`` where row 0 is
            displacement and row 1 is velocity.

        Returns
        -------
        numpy.ndarray
            Time derivative shaped ``(2, n_blocks)``: ``[dU/dt, dv/dt]``.

        Raises
        ------
        ValueError
            If U_dU shape is incorrect or if l2 is not set.
        """
        if U_dU.shape != (2, self.n_blocks):
            raise ValueError(f"U_dU must have shape (2, {self.n_blocks}), got {U_dU.shape}")

        if self.l2 is None:
            raise ValueError("l2 must be set for discrete models. Use continuum models instead.")

        U = U_dU[0, :]
        v = U_dU[1, :]
        shear = self.l2 * laplacian(U) + self.pll_spd * t - U
        d2u_dt2 = np.zeros_like(U)
        stick = (v <= 0) & (shear <= 1.0)
        v[stick & (v < 0)] = 0
        self.U0[stick] = U[stick]
        slip = ~stick

        d2u_dt2[slip] = shear[slip] -  slip_friction(U[slip], self.U0[slip], alpha=self.alpha, sigma=0.01)

        return np.vstack((v, d2u_dt2))



    def RK4(self, t, t_step, x_v: np.ndarray, f) -> ndarray:
        """Advance the state by one step using classic 4th-order Runge–Kutta.

        Parameters
        ----------
        t : float
            Current time.
        t_step : float
            Time step size ``dt``.
        x_v : numpy.ndarray
            State to advance (typically shaped ``(2, n_blocks)``). This array
            is updated in-place and also returned.
        f : callable
            Function implementing the ODE right-hand side: ``f(t, x_v)``.

        Returns
        -------
        numpy.ndarray
            The updated state (same object as ``x_v``).

        Raises
        ------
        FloatingPointError
            If the input state contains NaN/Inf (usually indicating
            instability).
        """

        if np.any(np.isnan(x_v)) or np.any(np.isinf(x_v)):
            raise FloatingPointError(f"Numerical instability detected at t={t}. State contains NaN/Inf.")

        k1 = f(t, x_v)
        k2 = f(t + t_step/2, x_v + t_step * k1 / 2)
        k3 = f(t + t_step/2, x_v + t_step * k2 / 2)
        k4 = f(t + t_step, x_v + t_step * k3)
        x_v += (t_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_v

    def set_initial(self ,state: str = 'Homogenous', ODE : str = 'C_L_ODE_fwrd'):

        """Initialize ``self.state`` with a displacement pattern and select ODE formulation.

        Parameters
        ----------
        state : str or numpy.ndarray, optional
            If a string, selects a built-in initial condition:
            - ``'Homogenous'``: all displacements set to 0
            - ``'Random'``: uniform random displacements in ``[-0.1, 0.1]``

            If an array, it is used directly as the initial displacement (must
            have shape ``(n_blocks,)``).
        ODE : str, optional
            Name of the ODE formulation to use. Must be one of:
            'Discrete_vel_weakening' -> Discrete BK model with velocity-weakening friction,
            'Discrete_vel_weakening_evl_pp' -> Discrete BK model with evolving pore pressure,
            'Continuum_vel_weakening_damped_cnst_pp' -> Continuum BK model with constant pore pressure,
            'Continuum_vel_weakening_damped' -> Continuum BK model with velocity damping,
            'Discrete_slip_weakening' -> Discrete BK model with slip-weakening friction.
            Default is ``'Discrete_vel_weakening'``.

        Raises
        ------
        ValueError
            If state string is invalid or array has wrong shape, or if ODE name is invalid.
        """

        if isinstance(state, str):
            if state == 'Homogenous':
                state = np.zeros(shape=(self.n_blocks))
            elif state == 'Random':
                state = np.random.uniform(-0.1,0.1, size=(self.n_blocks))
            else:
                raise ValueError(f'Displacement spacing must be Homogenous or Random, got {state}')
        elif isinstance(state, np.ndarray):
            if state.shape != (self.n_blocks,):
                raise ValueError(f'state array must have shape ({self.n_blocks},), got {state.shape}')
        else:
            raise TypeError(f'state must be a string or numpy array, got {type(state)}')
        ODE_grid = {'C_L_ODE_fwrd' : self.Discrete_vel_weakening,
                    'C_L_ODE_fwrd_evl_pp' : self.Discrete_vel_weakening_evl_pp,
                    'C_L_ODE_fwrd_continuum_damped_cnst_pp' : self.Continuum_vel_weakening_damped_cnst_pp,
                    'C_L_ODE_fwrd_continuum_damped' : self.Continuum_vel_weakening_damped,
                    'C_L_ODE_slip_weakening' : self.Discrete_slip_weakening,
                    'Discrete_vel_weakening' : self.Discrete_vel_weakening,
                    'Discrete_vel_weakening_evl_pp' : self.Discrete_vel_weakening_evl_pp,
                    'Continuum_vel_weakening_damped_cnst_pp' : self.Continuum_vel_weakening_damped_cnst_pp,
                    'Continuum_vel_weakening_damped' : self.Continuum_vel_weakening_damped,
                    'Discrete_slip_weakening' : self.Discrete_slip_weakening,
                    }
        if ODE not in ODE_grid.keys():
            raise ValueError(f'ODE must be one of {ODE_grid.keys()}')
        self.ODE = ODE_grid[ODE]

        v = np.zeros(shape=(self.n_blocks))
        self.state = np.vstack([state, v])
        self.initial_state = np.vstack([state, v])
        self.U0 = state


    def simulate(self,t0, tn, dt):
        """Simulate the fault dynamics and return the full time history.

        Uses RK4 time integration with the selected ODE formulation. If pore pressure
        evolution is enabled, also advances the pore pressure field using Crank-Nicolson
        and translates the pressure domain with the plate speed.

        Parameters
        ----------
        t0 : float
            Start time.
        tn : float
            End time (non-inclusive; uses ``np.arange(t0, tn, dt)``).
        dt : float
            Fixed time step for integration.

        Returns
        -------
        numpy.ndarray
            History array shaped ``(2, n_blocks, n_steps)`` where ``n_steps`` is
            the number of saved states (initial state + integration steps).

        Raises
        ------
        ValueError
            If initial state is not set, if time parameters are invalid, or if dt is non-positive.
        RuntimeError
            If ODE has not been selected.
        """
        if self.state is None:
            raise ValueError("Initial state not set. Call set_initial() first.")

        if not hasattr(self, 'ODE') or self.ODE is None:
            raise RuntimeError("ODE formulation not selected. Call set_initial() first.")

        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        if tn <= t0:
            raise ValueError(f"End time ({tn}) must be greater than start time ({t0})")

        times = np.arange(t0, tn, dt)
        hist = np.empty((2, self.n_blocks, times.size + 1), dtype=np.float32)
        hist[:, :, 0] = self.state

        for i,t in enumerate(times, start=1):
            self.state = self.RK4(t, dt, self.state, self.ODE)
            hist[:, :, i] = self.state

            if self.pore_pressure is not None:
                self.pore_pressure.crank_nicolson_step(dt)
                self.pore_pressure.xp += dt*self.pll_spd





        return hist
