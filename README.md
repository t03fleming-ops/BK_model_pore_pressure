# VSRP-Code — Pore Pressure Effects on BK Model b-Values

Numerical simulation code by Toby Fleming ([github.com/t03fleming-ops/BK_model_pore_pressure](https://github.com/t03fleming-ops/BK_model_pore_pressure)) for studying how **fluid injection (pore pressure)** changes **earthquake statistics** in a **1D Burridge–Knopoff (BK) spring-block fault model**.

For architecture diagrams and a full logic walkthrough, see [CODE_LOGIC.md](CODE_LOGIC.md).

---

## Research Goal

Enhanced Geothermal Systems (EGS) can induce earthquakes when fluid is injected into the subsurface. This code asks: **how does pore pressure change the Gutenberg–Richter (GR) frequency–magnitude distribution**, especially the **b-value slope**

The model researched is the continuum Carlson–Langer velocity-weakening BK fault with constant pore pressure. Other configurations are:
- **Discrete_vel_weakening** Discrete model using velocity weakening function.
- **Discrete_vel_weakening_evl_pp'** Discrete model using velocity weakening function with pore pressure evolving with diffusion equation.
- **Continuum_vel_weakening_damped_cnst_pp** Continuum limit of BK model with viscous damping and velocity weakening friction with homogenous pore pressure.
- **Continuum_vel_weakening_damped** Continuum limit of BK model with viscous damping and velocity weakening friction.
-**Discrete_slip_weakening** Discrete BK model using slip weakening friction.

Events are detected from velocity spikes, sized by total slip, and plotted on a GR curve to extract b-values.

---

## Quick Start

```bash
cd BK_model_pore_pressure-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimal simulation example:

```python
from fault import Fault
from event_detection import detect_events_from_hist, plot_GR

# Create a continuum fault with constant pore pressure ratio 0.6
fault = Fault(
    n_blocks=int(200 / 0.0625),
    alpha=1.0,
    pll_spd=0.001,
    damping_coef=0.02,
    block_spacing=0.0625,
    p_ratio=0.6,
)
fault.set_initial("Random", ODE="Continuum_vel_weakening_damped_cnst_pp")

dt = 0.001
hist = fault.simulate(0, 500, dt)  # short test run, hist[1] works in detect_events_from_hist for reduce memory usage.
events = detect_events_from_hist(hist, dt=dt, threshold_factor=0)
b = plot_GR(events, bin_size=0.33, b_value=True, b_range = (-12,2), cutoff_at_peak=True, element_size = 0.0625)
```

Long production runs use `Simulation.ipynb` with chunked saving and `joblib` parallelism.

---

## Simulation Workflow

1. **Configure** `Fault` with physics parameters (`alpha`, `p_ratio`, `block_spacing`, etc.)
2. **Initialize** random displacements and select an ODE formulation via `set_initial()`
3. **Integrate** forward in time with RK4 (`simulate()`), optionally evolving pore pressure each step
4. **Save** velocity chunks to manage disk use
5. **Detect events** scipy connected-component labeling
6. **Analyze** GR curves and b-values with `plot_GR()`

---

## Parameter Reference

| Parameter | Code name | Meaning | Typical value |
|-----------|-----------|---------|---------------|
| Friction steepness | `alpha` | How fast friction weakens with slip speed | 1 or 3 |
| Plate speed | `pll_spd` | Dimensionless loading rate | 0.001 |
| Fault length (elements) | `n_blocks` | Number of spring-block elements | 3200 |
| Viscous damping | `damping_coef` | Stabilizes continuum limit | 0.02 |
| Pore pressure ratio | `p_ratio` | `pore_pressure / normal_stress` | 0–0.9 |
| Mesh spacing | `block_spacing` | Element size `a` in continuum limit | 0.0625 |
| Time step | `dt` | RK4 step size | 0.001 |
| Spring constant ratio| 'l' | Ratio of leaf and inter block springs | 9 |

---

## File Guide — Code Walkthrough

Each section below shows a logical code group with a markdown comment immediately above explaining what it does.

---

### `fault.py`

> **Module purpose:** Burridge–Knopoff spring-block dynamics with multiple friction and pore-pressure formulations.

> **Lines 18–46:** Slip-weakening friction — strength drops as slip `(x - x0)` increases.

```python
@jit(nopython=True, parallel=True)
def slip_friction(x, x0, alpha=2.5, sigma=0.01):
    return (1 - sigma) / (1 + (alpha / (1 - sigma)) * (x - x0))
```

> **Lines 48–72:** Velocity-weakening friction `phi(v)` used by Carlson–Langer stick-slip dynamics.

```python
@jit(nopython=True, parallel=True)
def phi(v, sigma=0.01):
    return (1.0 - sigma) / (1.0 + v / (1.0 - sigma))
```

> **Lines 74–105:** Discrete 1D Laplacian — elastic coupling between neighboring blocks.

```python
lap[1:-1] = U[2:] - 2 * U[1:-1] + U[:-2]
lap[0] = U[1] - U[0]
lap[-1] = U[-2] - U[-1]
```

> **Lines 150–261:** `Fault.__init__` — store physics parameters; optionally create a `PorePressure` diffuser when `evl_PorePressure=True`.

```python
self.lap_mat = build_1d_laplacian_matrix(n_blocks).tocsc()
if evl_PorePressure:
    self.pore_pressure = PorePressure(
        n_blocks=self.n_blocks,
        injection_pressure=self.inj_preasure,
        form="Gaussian", diff_coeff=5.0, res=0.001,
    )
    self.U_indxs = np.arange(0, self.n_blocks + 1, 1)
```

> **Lines 377–420:** Continuum ODE with damping + **constant pore pressure** — main formulation used in the notebook.

```python
f0 = self.p_ratio - 1
shear = (1 / self.block_spacing ** 2) * laplacian(U) + self.pll_spd * t - U
stick = (v <= 0) & (shear <= abs(f0))
d2u_dt2[slip] = shear[slip] - np.sign(v[slip]) * abs(f0) * phi(2 * self.alpha * np.abs(v[slip]))
    + (self.n / self.block_spacing ** 2) * laplacian(v)[slip]
```

> **Lines 317–374:** Continuum/discrete ODE with **evolving pore pressure** — interpolate pressure onto block positions.

```python
fp = self.pore_pressure.Pressure
pp_at_blck = np.interp(self.U_indxs * self.block_spacing + U, self.pore_pressure.xp, fp)
f0 = pp_at_blck / self.normal_pressure - 1
stick = (v <= 0) & (shear <= abs(f0))
```

> **Lines 511–546:** Classic 4th-order Runge–Kutta time stepper.

```python
k1 = f(t, x_v)
k2 = f(t + t_step / 2, x_v + t_step * k1 / 2)
k3 = f(t + t_step / 2, x_v + t_step * k2 / 2)
k4 = f(t + t_step, x_v + t_step * k3)
x_v += (t_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
```

> **Lines 548–606:** Pick initial displacement pattern (`Homogenous` or `Random`) and map ODE name to method.

```python
ODE_grid = {
    "C_L_ODE_fwrd_continuum_damped_cnst_pp": self.Continuum_vel_weakening_damped_cnst_pp,
    "C_L_ODE_fwrd_evl_pp": self.Discrete_vel_weakening_evl_pp,
    ...
}
self.ODE = ODE_grid[ODE]
self.state = np.vstack([state, v])
```

> **Lines 609–664:** Main simulation loop — RK4 steps plus optional pore-pressure diffusion and advection.

```python
for t in np.arange(t0, tn, dt):
    self.state = self.RK4(t, dt, self.state, self.ODE)
    if self.pore_pressure is not None:
        self.pore_pressure.crank_nicolson_step(dt)
        self.pore_pressure.xp += dt * self.pll_spd
return np.stack(hist, axis=2)
```

---

### `event_detection.py`

> **Module purpose:** Turn velocity histories into earthquake catalogs and GR statistics.

> **Lines 17–35:** Median Absolute Deviation for robust slip threshold estimation.

```python
@jit(nopython=True, parallel=True)
def _mad(data):
    median = np.median(data)
    return np.median(np.abs(data - median))
```

> **Lines 150–292:** Core event detection — threshold velocity, label connected slip patches, compute event size.

```python
threshold = float(1.4826 * _mad(vel) * threshold_factor)
mask = np.abs(vel) > threshold
# ... group connected (block, time) pixels into events ...
displacement = float(disp_step[blocks, times].sum())
events.append((t_start, t_end, displacement, nucleation_block, np.unique(blocks).size))
```

> **Lines 399–524:** Gutenberg–Richter plot — magnitude = ln(slip), fit b-value from log-frequency slope.

```python
mags = np.log(events[:, 2]) / np.log(log_base)
R_mu, mu = np.histogram(mags, bins=bins)
slope, intercept, r, p, se = linregress(mu[l_mask], np.log(R_mu[l_mask]))
# b-value = -slope
```

> **Lines 644–724:** Load saved velocity chunks from disk and concatenate detected events.

```python
for filename in os.listdir(full_directory):
    data = np.load(file_path)
    event = detect_events_from_hist(data, dt=dt, t0=t0, ...)
    events.extend(event)
return np.vstack(events)
```
---

### `pore_pressure.py`

> **Module purpose:** Solve 1D pore-pressure diffusion on a fine grid using Crank–Nicolson.

> **Lines 20–22:** Build a Gaussian initial pressure profile centered at the injection site.

```python
def gaussian_press(x, injection_pressure, mu, sigma=1.0):
    gaussian = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gaussian * injection_pressure
```

> **Lines 25–39:** Construct the sparse 1D Laplacian matrix (interior second difference, one-sided at boundaries).

```python
def build_1d_laplacian_matrix(n: int):
    main = -2.0 * np.ones(n)
    main[0] = -1.0
    main[-1] = -1.0
    off = np.ones(n - 1)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csc")
```

> **Lines 59–96:** Initialize grid spacing, injection site, and starting pressure field; optionally clamp injection node.

```python
self.x = np.arange(0.0, self.n_blocks, self.res, dtype=float)
self.inj_site = int(self.x.size // 2)
self.Pressure = self.form(self.x, self.injection_pressure, self.x[self.inj_site], self.res)
if self.hold_injection_pressure:
    self.Pressure[self.inj_site] = self.injection_pressure
```

> **Lines 108–132:** Build and cache Crank–Nicolson operators `(I - dt/2 A) P^{n+1} = (I + dt/2 A) P^n`.

```python
I = sp.identity(self.x.size, format="csc")
lhs = (I - (dt / 2.0) * self._A).tocsc()
rhs = (I + (dt / 2.0) * self._A).tocsc()
lhs_factor = spla.factorized(lhs)
```

> **Lines 139–149:** Advance one diffusion time step by solving the linear system.

```python
cache = self._get_crank_nicolson(dt)
b = cache.rhs @ P
self.Pressure = cache.lhs_factor(b)
return self.Pressure
```

> **Lines 152–161:** Run a full time series and return `(times, pressure_history)`.

```python
for i in range(1, times.size):
    hist[:, i] = self.crank_nicolson_step(dt)
return times, hist
```

---
### `Simulation.ipynb` 

> **A few simulation functions using joblib's parallelisation.** Defines `run_sim()` functions that sweep pore pressure ratios and performs simulations in chunks.

> ```python
param_dict = {
    1: [1, 0.001, 200/0.0625, 0.02, 0, 0.0625],
    2: [1, 0.001, 200/0.0625, 0.02, 0.6, 0.0625],
    ...
}
fault = Fault(params[2], alpha=params[0], pll_spd=params[1], ...)
```

> **Chunked simulation loop:** simulate 75 time units, save velocity, detect events, delete raw chunks.

```python
for time in range(0, tn, chunk):
    fault.state = np.load(f"event_data_SRF/.../end_state_{sim_num}.npy")
    vel = fault.simulate(time, time + chunk, ts)[1]
    np.save(f"event_data_SRF/.../event_sample_SRF_{time}", vel)
    events = find_events(sim_num, t0=time, threshold_factor=0)
```

> **Parallel execution:** run multiple parameter sets concurrently.

```python
from joblib import Parallel, delayed
Parallel(n_jobs=4)(delayed(run_sim)(i) for i in range(1, 5))
```

---

### `c_l_faultsim_rt.ipynb`

> **Collection of function uses, used during research.** Defines `run_sim()` functions that sweep pore pressure ratios.

> **Parameter dict:** `[alpha, pll_spd, n_blocks, damping, p_ratio, block_spacing]`

```python
param_dict = {
    1: [1, 0.001, 200/0.0625, 0.02, 0, 0.0625],
    2: [1, 0.001, 200/0.0625, 0.02, 0.6, 0.0625],
    ...
}
fault = Fault(params[2], alpha=params[0], pll_spd=params[1], ...)
```

> **Chunked simulation loop:** simulate 75 time units, save velocity, detect events, delete raw chunks.

```python
for time in range(0, tn, chunk):
    fault.state = np.load(f"event_data_SRF/.../end_state_{sim_num}.npy")
    vel = fault.simulate(time, time + chunk, ts)[1]
    np.save(f"event_data_SRF/.../event_sample_SRF_{time}", vel)
    events = find_events(sim_num, t0=time, threshold_factor=0)
```

> **Parallel execution:** run multiple parameter sets concurrently.

```python
from joblib import Parallel, delayed
Parallel(n_jobs=4)(delayed(run_sim)(i) for i in range(1, 5))
```

---

## Output Files

```
event_data_SRF/
  Original_friction__clamped_velocity/
    Sim_{run}/
      end_state_{run}.npy          # checkpoint (U, v) state
      event_sample_SRF_{time}      # velocity chunk
    Pore Pressure in continuum limit/
      Constant pore pressure/
        events_SRF_PPcont_alpha=..._ppr=....npy
Figures/
  GR_{alpha}.png
```

These directories are gitignored; create them before running long simulations.

---

## References

- Burridge & Knopoff (1967), *Model and theoretical seismicity*
- Carlson & Langer (1989), *Mechanical model of an earthquake fault*
- Myers & Langer (1993); Shaw (1994) — viscous damping for continuum limit
- Mori & Kawamura (2006, 2008) — BK continuum validation
- Hubbert & Rubey (1959) — effective stress and pore pressure

---

## Related Documentation

- [CODE_LOGIC.md](CODE_LOGIC.md) — architecture diagrams and detailed module logic
- `BK and Pore pressure.docx` (parent folder) — Toby's full research write-up
