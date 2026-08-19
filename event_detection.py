"""
Event detection and analysis for Burridge-Knopoff fault simulations.

Provides functions to detect earthquake events from velocity histories,
visualize event maps, and analyze magnitude distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Optional, Union
from numba import jit


@jit(nopython=True, parallel=True)
def _mad(data: np.ndarray) -> np.ndarray:
    """
    Compute Median Absolute Deviation (MAD) for robust threshold estimation.

    Parameters
    ----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to compute MAD

    Returns
    -------
    ndarray
        Median absolute deviation
    """
    median = np.median(data)
    return np.median(np.abs(data - median))


def _iter_events_sparse(mask: np.ndarray, block_neighbors: int,
                        time_neighbors: int):
    """
    Identify connected event components using sparse iteration (fallback).

    Parameters
    ----------
    mask : ndarray, shape (n_blocks, n_steps)
        Boolean mask of slipping blocks
    block_neighbors : int
        Spatial connectivity radius (must be non-negative)
    time_neighbors : int
        Temporal connectivity radius (must be non-negative)

    Yields
    ------
    tuple of (blocks, times)
        Arrays of block indices and time indices for each event

    Raises
    ------
    ValueError
        If mask is not 2D or if neighbors parameters are negative.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if block_neighbors < 0:
        raise ValueError(f"block_neighbors must be non-negative, got {block_neighbors}")
    if time_neighbors < 0:
        raise ValueError(f"time_neighbors must be non-negative, got {time_neighbors}")

    n_blocks, n_steps = mask.shape
    on = np.argwhere(mask)
    if on.size == 0:
        return
    on_set = {(int(b), int(t)) for b, t in on}
    while on_set:
        b0, t0 = on_set.pop()
        stack = [(b0, t0)]
        blocks = []
        times = []
        while stack:
            b, t = stack.pop()
            blocks.append(b)
            times.append(t)
            b_min = max(0, b - block_neighbors)
            b_max = min(n_blocks, b + block_neighbors + 1)
            t_min = max(0, t - time_neighbors)
            t_max = min(n_steps, t + time_neighbors + 1)
            for nb in range(b_min, b_max):
                for nt in range(t_min, t_max):
                    neighbor = (nb, nt)
                    if neighbor in on_set:
                        on_set.remove(neighbor)
                        stack.append(neighbor)
        yield np.asarray(blocks), np.asarray(times)


def _iter_events_scipy(mask: np.ndarray, block_neighbors: int,
                       time_neighbors: int):
    """
    Identify connected event components using scipy (fast method).

    Parameters
    ----------
    mask : ndarray, shape (n_blocks, n_steps)
        Boolean mask of slipping blocks
    block_neighbors : int
        Spatial connectivity radius (must be non-negative)
    time_neighbors : int
        Temporal connectivity radius (must be non-negative)

    Yields
    ------
    tuple of (blocks, times)
        Arrays of block indices and time indices for each event

    Raises
    ------
    ValueError
        If mask is not 2D or if neighbors parameters are negative.
    ImportError
        If scipy is not available.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if block_neighbors < 0:
        raise ValueError(f"block_neighbors must be non-negative, got {block_neighbors}")
    if time_neighbors < 0:
        raise ValueError(f"time_neighbors must be non-negative, got {time_neighbors}")

    from scipy.ndimage import label

    structure = np.ones(
        (2 * block_neighbors + 1, 2 * time_neighbors + 1), dtype=bool
    )
    labels, n_labels = label(mask, structure=structure, output=np.int32)
    del structure

    if n_labels == 0:
        return

    coords = np.argwhere(mask)
    label_ids = labels[mask]
    order = np.argsort(label_ids)
    coords = coords[order]
    label_ids = label_ids[order]



    start = 0
    for idx in range(1, len(label_ids) + 1):
        if idx == len(label_ids) or label_ids[idx] != label_ids[start]:
            group = coords[start:idx]
            yield group[:, 0], group[:, 1]
            start = idx

def _iter_events_scipy_updated(mask: np.ndarray, block_neighbors: int,
                       time_neighbors: int):
    """
    Identify connected event components using scipy (fast method).

    Parameters
    ----------
    mask : ndarray, shape (n_blocks, n_steps)
        Boolean mask of slipping blocks
    block_neighbors : int
        Spatial connectivity radius (must be non-negative)
    time_neighbors : int
        Temporal connectivity radius (must be non-negative)

    Yields
    ------
    tuple of (blocks, times)
        Arrays of block indices and time indices for each event

    Raises
    ------
    ValueError
        If mask is not 2D or if neighbors parameters are negative.
    ImportError
        If scipy is not available.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if block_neighbors < 0:
        raise ValueError(f"block_neighbors must be non-negative, got {block_neighbors}")
    if time_neighbors < 0:
        raise ValueError(f"time_neighbors must be non-negative, got {time_neighbors}")

    from scipy.ndimage import label, find_objects

    structure = np.ones(
        (2 * block_neighbors + 1, 2 * time_neighbors + 1), dtype=bool
    )
    labels, n_labels = label(mask, structure=structure, output=np.int32)
    del structure

    if n_labels == 0:
        return


    objects = find_objects(labels)

    for i, slc in enumerate(objects):
        if slc is None:
            continue
        label_id = i + 1
        block_off, time_off = slc[0].start, slc[1].start
        # Only look within this event's bounding box, not the whole array
        local = labels[slc] == label_id
        b, t = np.nonzero(local)
        yield b + block_off, t + time_off


def detect_events_from_hist(
        hist: np.ndarray,
        dt: float,
        t0: float = 0.0,
        vel_index: int = 1,
        threshold: Optional[float] = None,
        threshold_factor: float = 1,
        block_neighbors: int = 1,
        time_neighbors: int = 1,
        min_duration: int = 1,
        use_scipy: bool = True,
        return_threshold: bool = False,
        return_event_map: bool = False, ) -> Union[List[Tuple[float, float, float, int]],
Tuple[List[Tuple[float, float, float, int]], np.ndarray]]:
    """
    Detect earthquake events from velocity history.

    Parameters
    ----------
    hist : ndarray
        Simulation history, either:
        - 3D shape (state, n_blocks, n_steps): state[vel_index] = velocity
        - 2D shape (n_blocks, n_steps): velocity directly
    dt : float
        Time step
    t0 : float, default=0.0
        Starting time offset
    vel_index : int, default=1
        Index of velocity in state dimension (for 3D hist)
    threshold : float, optional
        Absolute velocity threshold. If None, computed as 1.4826 * MAD * threshold_factor.
        Must be positive if provided.
    threshold_factor : float, default=1
        Factor for MAD-based threshold (used only if threshold=None). Must be positive.
    block_neighbors : int, default=1
        Spatial connectivity radius for event grouping. Must be non-negative.
    time_neighbors : int, default=1
        Temporal connectivity radius for event grouping. Must be non-negative.
    min_duration : int, default=1
        Minimum event duration in time steps (unused currently, for future filtering)
    use_scipy : bool, default=True
        Use scipy.ndimage.label for fast event detection (fallback to sparse if unavailable)
    return_threshold : bool, default=False
        If True, return the computed threshold value along with events
    return_event_map : bool, default=False
        If True, also return 2D event map with event IDs

    Returns
    -------
    events : list of tuples
        Each event is (t_start, t_end, total_displacement, nucleation_block, block_count)
    threshold : float, optional
        Only returned if return_threshold=True. The threshold value used for detection.
    event_map : ndarray, shape (n_blocks, n_steps), optional
        Only returned if return_event_map=True. Values are event IDs (0 = no event)

    Raises
    ------
    ValueError
        If dt <= 0, hist has wrong shape, vel_index out of range, threshold <= 0,
        threshold_factor <= 0, or neighbor parameters are negative

    Notes
    -----
    Events are identified as connected components of blocks with |velocity| > threshold.
    Displacement is computed as sum(|velocity| * dt) over all blocks/times in the event.
    MAD scaling factor 1.4826 converts MAD to match std dev for Gaussian data.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    if threshold_factor < 0:
        raise ValueError(f"threshold_factor must be positive, got {threshold_factor}")

    if block_neighbors < 0:
        raise ValueError(f"block_neighbors must be non-negative, got {block_neighbors}")

    if time_neighbors < 0:
        raise ValueError(f"time_neighbors must be non-negative, got {time_neighbors}")

    data = np.asarray(hist)
    if data.ndim == 2:
        vel = data
    elif data.ndim == 3:
        if data.shape[0] == 1 and vel_index == 1:
            vel_index = 0
        if vel_index < 0 or vel_index >= data.shape[0]:
            raise ValueError(
                f"vel_index={vel_index} out of range for hist shape {data.shape}"
            )
        vel = data[vel_index]
    else:
        raise ValueError(
            f"hist must be 2D (n_blocks, n_steps) or 3D (state, n_blocks, n_steps), "
            f"got shape {data.shape}"
        )

    if threshold is None:
        threshold = float(1.4826 * _mad(vel) * threshold_factor)

    mask = np.abs(vel) > threshold
    disp_step = np.abs(vel) * dt

    if use_scipy:
        try:
            iterator = _iter_events_scipy_updated(mask, block_neighbors, time_neighbors)
        except (ImportError, Exception):
            iterator = _iter_events_sparse(mask, block_neighbors, time_neighbors)
    else:
        iterator = _iter_events_sparse(mask, block_neighbors, time_neighbors)


    events = []
    event_map = None
    event_id = 1
    if return_event_map:
        event_map = np.zeros_like(vel, dtype=int)

    for blocks, times in iterator:
        start_idx = int(times.min())
        end_idx = int(times.max())
        displacement = float(disp_step[blocks, times].sum())
        t_start = t0 + start_idx * dt
        t_end = t0 + (end_idx + 1) * dt

        nucleation_candidates = np.sort(blocks[times == start_idx])
        nucleation_block = int(nucleation_candidates[0])

        events.append((t_start, t_end, displacement, nucleation_block, np.unique(blocks).size))

        if return_event_map:
            event_map[blocks, times] = event_id
            event_id += 1

    if return_event_map:
        return events, event_map
    if return_threshold:
        return events, threshold
    if return_event_map and return_threshold:
        return events, event_map, threshold
    return events



def plot_event_map(
        event_map: np.ndarray,
        dt: Optional[float] = None,
        t0: float = 0.0,
        figsize: Tuple[int, int] = (15, 10),
        cmap: str = 'viridis',
        save_path: Optional[str] = None,
        title: str = '',
        binary: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize event detection map showing which blocks slip during each event.

    Parameters
    ----------
    event_map : ndarray, shape (n_blocks, n_timesteps)
        2D array where 0 = no event, positive integers = event IDs
    dt : float, optional
        Time step for x-axis labels. If None, uses step indices
    t0 : float, default=0.0
        Starting time offset
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    cmap : str, default='tab20'
        Matplotlib colormap for events (0 is always white). Ignored if binary=True.
    save_path : str, optional
        If provided, saves figure to this path
    label : str, default=''
        Additional label for plot (currently unused)
    binary : bool, default=False
        If True, ignore individual event IDs and plot all event pixels
        (event_map > 0) as black, with non-event pixels as white.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Raises
    ------
    ValueError
        If event_map is not 2D

    Notes
    -----
    Background (event ID = 0) is shown in white.
    Each event gets a unique color from the colormap, unless binary=True,
    in which case all events are shown in black.
    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt


    if event_map.ndim != 2:
        raise ValueError(f"event_map must be 2D, got shape {event_map.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    n_events = int(event_map.max())

    if binary:
        event_map = (event_map > 0).astype(int)
        custom_cmap = ListedColormap(['white', 'black'])
        vmin, vmax = 0, 1
    else:
        if n_events > 0:
            base_cmap = plt.get_cmap(cmap, n_events).colors
            np.random.shuffle(base_cmap)
            base_cmap = np.concatenate((np.array([[1, 1, 1, 1]]), base_cmap))
            custom_cmap = ListedColormap(base_cmap)

        else:
            custom_cmap = ListedColormap(['white'])
        vmin, vmax = 0, n_events

    im = ax.imshow(
        event_map, aspect='auto', cmap=custom_cmap,
        interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=35)
    ax.set_xlabel('Time Step' if dt is None else 'Time')
    ax.set_ylabel('Element')

    if dt is not None:
        n_steps = event_map.shape[1]
        times = t0 + np.arange(n_steps) * dt
        n_ticks = min(10, n_steps)
        tick_indices = np.linspace(0, n_steps - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{times[i]:.2f}' for i in tick_indices])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_history(data,
                 cmap : str = 'viridis',
                 return_fig=False,
                 v_zlim : tuple=None,
                 x_zlim : tuple=None):
    """Plot a simulation history as 3D surfaces.

    Parameters
    ----------
    data : numpy.ndarray
        History array shaped ``(2, n_blocks, n_steps)`` where ``data[0]`` is
        displacement and ``data[1]`` is velocity.
    cmap : str, default='viridis'
        matplotlib colormap for surface.
    return_fig : bool, default=False
        If True, return the created matplotlib figures instead of only showing
        the plots.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.figure.Figure] or None
        ``(fig_vel, fig_pos)`` when ``return_fig=True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If data is not 3D with shape (2, n_blocks, n_steps).
    """
    import matplotlib.pyplot as plt
    plt.tight_layout()
    if len(data.shape) != 3:
        raise ValueError(f"data must be 3D, got shape {data.shape}")
    if data.shape[0] != 2:
        raise ValueError(f"data must have shape (2, n_blocks, n_steps), got {data.shape}")
    n_blocks = data.shape[1]
    n_steps = data.shape[2]

    rest_pos = np.arange(n_blocks)
    time = np.arange(n_steps)

    vel = data[1, :, :].T
    pos = data[0, :, :].T

    X, Y = np.meshgrid(time, rest_pos, indexing='ij')

    fig1 = plt.figure(figsize=(14,7))
    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.set_zlabel('Velocity', labelpad=10, size=20); ax1.set_xlabel('Time step', size=20); ax1.set_ylabel('Element', size=20)
    surf = ax1.plot_surface(
        X, Y, vel,
        cmap=cmap, linewidth=0, antialiased=True
    )
    fig2 = plt.figure(figsize=(14,7))
    ax2 = fig2.add_subplot(122, projection='3d')
    ax2.set_zlabel('Displacement', labelpad=10, size=20);ax2.set_xlabel('Time step', size=20);ax2.set_ylabel('Element', size=20)
    surf = ax2.plot_surface(X, Y, pos, cmap=cmap,
                            linewidth=0, antialiased=True
                            )
    if v_zlim: ax1.set_zlim(v_zlim)
    if x_zlim: ax2.set_zlim(x_zlim)

    if return_fig:
        return fig1, fig2


def clear_directory(
        directory: str,
        exceptions: Optional[Union[str, List[str]]] = None
) -> int:
    """
    Delete all files in the directory, preserving specified exception files.

    Parameters
    ----------
    directory : str
        Path to directory to clear
    exceptions : str or list of str, optional
        Filename(s) to preserve (do not delete)

    Returns
    -------
    int
        Number of files deleted

    Raises
    ------
    ValueError
        If the directory doesn't exist, is not a directory, or exceptions have the wrong type.
    OSError
        If file deletion fails

    Notes
    -----
    Only removes files, not subdirectories.
    Hidden files (e.g., .DS_Store) are deleted unless in exceptions.
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    # Convert exceptions to set for fast lookup
    if exceptions is None:
        exceptions_set = set()
    elif isinstance(exceptions, str):
        exceptions_set = {exceptions}
    elif isinstance(exceptions, (list, tuple)):
        exceptions_set = set(exceptions)
    else:
        raise ValueError(
            f"exceptions must be string or list of strings, got {type(exceptions)}"
        )

    deleted_count = 0
    for filename in os.listdir(directory):
        if filename in exceptions_set:
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                raise OSError(f"Failed to delete {file_path}: {e}")

    return deleted_count


def find_events(
        run: int,
        t0: float,
        directory: str = r"C:\Users\Toby\PycharmProjects\VSRP-Code\sim_run_temp_files\Sim_",
        dt: float = 0.001,
        threshold_factor: float = 1.0,
        return_event_map: bool = False,
) -> np.ndarray:
    """
    Extract event data from saved simulation chunks.

    Parameters
    ----------
    run : int
        Simulation run number
    t0 : float
        Start time offset for event detection
    directory : str, default='event_data_SRF/Original_friction__clamped_velocity/Sim_'
        Directory path prefix (run number appended)
    dt : float, default=0.001
        Time step used in simulation. Must be positive.
    threshold_factor : float, default=1.0
        Threshold factor for event detection. Must be positive.
    return_event_map : bool, default=False
        If True, also plot event maps for each chunk

    Returns
    -------
    ndarray
        Array of events with shape (N_events, 5) containing
        (t_start, t_end, displacement, nucleation_block, block_count)

    Raises
    ------
    ValueError
        If directory doesn't exist, dt <= 0, or threshold_factor <= 0
    OSError
        If file loading fails

    Notes
    -----
    Searches directory for .npy files, detects events in each,
    and concatenates all event data.
    Skips .DS_Store and end_state files.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    if threshold_factor < 0:
        raise ValueError(f"threshold_factor must be positive, got {threshold_factor}")

    full_directory = directory + f'{run}'

    if not os.path.exists(full_directory):
        raise ValueError(f"Directory does not exist: {full_directory}")

    events = []

    for filename in os.listdir(full_directory):
        # Skip metadata files
        if filename == '.DS_Store' or filename == f'end_state_{run}.npy':
            continue

        file_path = os.path.join(full_directory, filename)
        try:
            data = np.load(file_path)
        except (OSError, ValueError) as e:
            raise OSError(f"Failed to load {file_path}: {e}")

        event = detect_events_from_hist(
            data, dt=dt, t0=t0,
            threshold_factor=threshold_factor,
            return_event_map=return_event_map
        )

        if not return_event_map:
            events.extend(event)
        else:
            events.append(event[0])
            plot_event_map(event[1])
    return np.vstack(events)

