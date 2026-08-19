from typing import Tuple, Optional
import numpy as np
import os
import matplotlib.pyplot as plt
from natsort import realsorted
from scipy.stats import linregress
from matplotlib import ticker

def comp_avrg_event_duration(dir: str,
                             mag_range: Optional[tuple] or Optional[float] = None,
                             log_base: Optional[float] = np.e,
                             major_tick_interval: Optional[float] = 0.1,
                             marker_style: Optional[str] = 'o',
                             linestyle: Optional[str] = '-',
                             color: Optional[str] = 'b',
                             cmap : Optional[str] = None,
                             label: Optional[str] = '',
                             filter_slow_slip : bool = False,
                             fig_ax = None):
    """Plot mean event duration against pore-pressure ratio for event files.

    Loads each ``.npy`` file in ``dir`` (skipping ``.DS_Store``), where each
    file corresponds to a simulation run at a particular pore-pressure ratio
    encoded in its filename (``..._ppr=<value>.npy``). Column 2 of each
    event array is converted to a magnitude via ``log(size) / log(log_base)``.
    Events are optionally restricted to slow-slip events (start time > 5000
    and duration < 75) and to a magnitude range, then the mean duration
    (``t_end - t_start``) of the remaining events is computed per file and
    plotted against pore-pressure ratio.

    Parameters
    ----------
    dir : str
        Path to the directory containing one ``.npy`` event file per
        pore-pressure ratio, with the ratio encoded at the end of each
        filename (after the last ``=``).
    mag_range : tuple or float or int, optional
        Magnitude filter applied before averaging. A ``(lower, upper)``
        tuple keeps events with ``lower < magnitude < upper``; a single
        float or int keeps events with ``magnitude > mag_range``; ``None``
        (default) keeps all events.
    log_base : float, optional
        Base of the logarithm used to convert event size (column 2) into
        magnitude. Defaults to the natural log base ``e``.
    major_tick_interval : float, optional
        Spacing, in pore-pressure-ratio units, between major x-axis ticks.
        Default is 0.1.
    marker_style : str, optional
        Matplotlib marker style used for the plotted points. Default 'o'.
    linestyle : str, optional
        Matplotlib line style connecting the plotted points. Default '-'.
    color : str, optional
        Matplotlib color for the plotted line/markers. Default 'b'.
    cmap : str, optional
        Colormap name. Currently accepted but not applied.
    label : str, optional
        Legend label for the plotted line. Default is an empty string.
    filter_slow_slip : bool, optional
        If True, restrict each file's events to those with
        ``t_start > 5000`` and ``(t_end - t_start) < 75`` before computing
        the mean duration. Default False.
    fig_ax : tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        Existing figure and axes to draw on. If None (default), a new
        figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plotted mean-duration curve.

    Raises
    ------
    TypeError
        If ``mag_range`` is not a tuple, float, int, or None; if
        ``marker_style`` or ``color`` is not a string; or if ``log_base``
        cannot be converted to a float.
    """
    if not isinstance(mag_range, tuple) and not isinstance(mag_range, (float,int)) and not mag_range == None:
        raise TypeError("mag_range must be either a tuple or a float")
    if  not isinstance(marker_style, str) or not isinstance(color, str):
        raise TypeError('marker_style, major_tick_interval, and color must be strings')
    try:
        log_base = float(log_base)
    except:
        raise TypeError("log_base must be a number")
    if fig_ax: fig = fig_ax[0] ; ax = fig_ax[1] ;ax.set_ylabel('Event duration', size=20);ax.set_xlabel('Pore Pressure ratio', size=20)
    else: fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.grid(visible=True, which='major', linestyle='--')
    pprs = []
    time_avrgs = []

    for filename in realsorted(os.listdir(dir)):
        if filename == ".DS_Store":
            continue
        pprs.append(float(filename.split('=')[-1].removesuffix('.npy')))
        a = filename.split('=')[-2].removesuffix('_ppr')

        events = np.load(dir + '/' + filename, allow_pickle=True)
        events = np.vstack(events)
        with np.errstate(divide="ignore", invalid="ignore"):
            events[:, 2] = np.log(events[:, 2]) / np.log(log_base)
        if filter_slow_slip:
            events = events[events[:,0] > 5000]
            events = events[(events[:,1] - events[:,0]) < 75]
        if type(mag_range) == tuple:
            lower = mag_range[0];
            upper = mag_range[1]
            rel_events = events[(events[:, 2] > lower) & (events[:, 2] < upper)]

        if type(mag_range) == float or type(mag_range) == int:
            rel_events = events[events[:, 2] > mag_range]
        elif not mag_range:
            rel_events = events

        times = rel_events[:, 1] - rel_events[:, 0]
        time_avrgs.append(np.mean(times))




    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_interval))
    ax.plot(pprs, time_avrgs, marker=marker_style, linestyle=linestyle, c=color, label=label)
    return fig, ax

def recurrence_time(
        events,
        r: Optional[float] = None,
        min_mag: float = 2,
        log_base: float = np.e,
        epicenter_col: int = 3,
        element_size: float = 0.0625,
        simulation_time_step: Optional[float] = None,
):
    """Return local recurrence times between qualifying events.

    Events are sorted by start time (column 0), and a magnitude is computed
    for each as ``log(events[:, 2] * element_size, log_base)``. Only events
    with a finite magnitude at or above ``min_mag`` are considered. If ``r``
    is None, recurrence times are the positive consecutive time differences
    between qualifying events. If ``r`` is provided, for each qualifying
    event the function instead finds the next-in-time qualifying event whose
    epicenter (column ``epicenter_col``, scaled by ``element_size``) lies
    within distance ``r`` of the current event's epicenter, and returns the
    time gap to that event. If ``simulation_time_step`` is given, recurrence
    times shorter than it are discarded.

    Parameters
    ----------
    events : numpy.ndarray
        2D array of event records with at least 4 columns. Column 0 is the
        event start time and column 2 is the event size used for the
        magnitude calculation.
    r : float, optional
        Maximum epicentral distance (in the same units as
        ``epicenter_col * element_size``) for a later event to count as a
        local recurrence of an earlier one. If None (default), recurrence
        times are computed without any spatial restriction.
    min_mag : float, optional
        Minimum magnitude threshold for an event to qualify. Default 2.
    log_base : float, optional
        Base of the logarithm used to compute magnitude. Default is the
        natural log base ``e``.
    epicenter_col : int, optional
        Column index in ``events`` holding the epicenter location, used only
        when ``r`` is provided. Default 3.
    element_size : float, optional
        Scale factor applied to event size (for magnitude) and to the
        epicenter column (for distance). Default 0.0625.
    simulation_time_step : float, optional
        If provided, recurrence times smaller than this value are dropped
        from the result.

    Returns
    -------
    numpy.ndarray
        1D array of recurrence times (float). Empty if fewer than two
        qualifying events are found.

    Raises
    ------
    ValueError
        If ``events`` is not a 2D array with at least 4 columns; if ``r`` is
        provided but ``events`` does not contain ``epicenter_col``; or if
        ``log_base`` is non-positive or equal to 1.
    """
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError("events must be a 2D array with at least 4 columns")
    if r is not None and events.shape[1] <= epicenter_col:
        raise ValueError(f"events does not contain epicenter_col={epicenter_col}")
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be positive and not equal to 1")

    events = events[np.argsort(events[:, 0])]
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitudes = np.log(events[:, 2] * element_size) / np.log(log_base)
    relevant_rows = np.flatnonzero(np.isfinite(magnitudes) & (magnitudes >= min_mag))

    if relevant_rows.size < 2:
        return np.array([], dtype=float)

    if r is None:
        recurrence_t = np.diff(events[relevant_rows, 0])
        return recurrence_t[recurrence_t > 0]

    epicenters = events[:, epicenter_col] * element_size
    recurrence_t = []
    for n, row in enumerate(relevant_rows[:-1]):
        later_rows = relevant_rows[n + 1:]
        later_rows = later_rows[events[later_rows, 0] > events[row, 0]]
        if later_rows.size == 0:
            continue
        local_rows = later_rows[np.abs(epicenters[later_rows] - epicenters[row]) <= r]
        if local_rows.size:
            recurrence_t.append(events[local_rows[0], 0] - events[row, 0])

    recurrence_t = np.asarray(recurrence_t, dtype=float)

    if simulation_time_step:
        recurrence_t = recurrence_t[recurrence_t >= simulation_time_step]

    return recurrence_t


def log_binned_recurrence_distribution(
        recurrence_t,
        bins_per_decade: int = 12,
        normalize_by_mean: bool = True,
):
    """Return local recurrence times between qualifying events.

    Events are sorted by start time (column 0), and a magnitude is computed
    for each as ``log(events[:, 2] * element_size, log_base)``. Only events
    with a finite magnitude at or above ``min_mag`` are considered. If ``r``
    is None, recurrence times are the positive consecutive time differences
    between qualifying events. If ``r`` is provided, for each qualifying
    event the function instead finds the next-in-time qualifying event whose
    epicenter (column ``epicenter_col``, scaled by ``element_size``) lies
    within distance ``r`` of the current event's epicenter, and returns the
    time gap to that event. If ``simulation_time_step`` is given, recurrence
    times shorter than it are discarded.

    Parameters
    ----------
    events : numpy.ndarray
        2D array of event records with at least 4 columns. Column 0 is the
        event start time, and column 2 is the event size used for the
        magnitude calculation.
    r : float, optional
        Maximum epicentral distance (in the same units as
        ``epicenter_col * element_size``) for a later event to count as a
        local recurrence of an earlier one. If None (default), recurrence
        times are computed without any spatial restriction.
    min_mag : float, optional
        Minimum magnitude threshold for an event to qualify. Default 2.
    log_base : float, optional
        Base of the logarithm used to compute magnitude. Default is the
        natural log base ``e``.
    epicenter_col : int, optional
        Column index in ``events`` holding the epicenter location, used only
        when ``r`` is provided. Default 3.
    element_size : float, optional
        Scale factor applied to event size (for magnitude) and to the
        epicenter column (for distance). Default 0.0625.
    simulation_time_step : float, optional
        If provided, recurrence times smaller than this value are dropped
        from the result.

    Returns
    -------
    numpy.ndarray
        1D array of recurrence times (float). Empty if fewer than two
        qualifying events are found.

    Raises
    ------
    ValueError
        If ``events`` is not a 2D array with at least 4 columns; if ``r`` is
        provided but ``events`` does not contain ``epicenter_col``; or if
        ``log_base`` is non-positive or equal to 1.
    """
    recurrence_t = np.asarray(recurrence_t, dtype=float)
    recurrence_t = recurrence_t[np.isfinite(recurrence_t) & (recurrence_t > 0)]
    if recurrence_t.size < 2:
        return np.array([]), np.array([]), np.array([]), np.nan

    mean_t = recurrence_t.mean()
    values = recurrence_t / mean_t if normalize_by_mean else recurrence_t
    vmin, vmax = values.min(), values.max()
    decades = np.log10(vmax) - np.log10(vmin)
    n_bins = int(np.ceil(decades * bins_per_decade))
    bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins + 1)
    density, edges = np.histogram(values, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    #keep = density > 0
    #return centers[keep], density[keep], edges[:-1][keep], mean_t
    return centers, density, edges[:-1], mean_t


def preceding_event_locality(
        events,
        min_mag: float = 2,
        preceeding_time: float or Tuple[float, float] = 33,
        log_base: float = np.e,
        element_size: float = 0.0625,
        epicenter_col: int = 3,
):
    """Return distances from main events to preceding events in a time window.

    Events are sorted by start time (column 0) and a magnitude is computed
    for each as ``log(events[:, 2] * element_size, log_base)``; those with a
    finite magnitude at or above ``min_mag`` are treated as "main" events.
    For each main event, every earlier event whose time gap to the main
    event falls within ``preceeding_time`` (a maximum look-back time if a
    single float, or an inclusive ``(lower, upper)`` window if a tuple) is
    treated as a preceding event. Its epicentral distance (column
    ``epicenter_col``, scaled by ``element_size``) to the main event's
    epicenter is recorded.

    Parameters
    ----------
    events : array_like
        2D array of event records containing at least ``epicenter_col + 1``
        columns. Column 0 is the event start time and column 2 is the event
        size used for the magnitude calculation.
    min_mag : float, optional
        Minimum magnitude threshold for an event to be treated as a main
        event. Default 2.
    preceeding_time : float or tuple of (float, float), optional
        Time window, measured backward from each main event, in which
        earlier events are counted as preceding events. A single float is
        treated as an upper bound (0, preceeding_time]; a tuple is treated
        as an inclusive-lower/exclusive-upper window
        ``[lower, upper]``. Default 33.
    log_base : float, optional
        Base of the logarithm used to compute magnitude. Default is the
        natural log base ``e``.
    element_size : float, optional
        Scale factor applied to event size (for magnitude) and to the
        epicenter column (for distance). Default 0.0625.
    epicenter_col : int, optional
        Column index in ``events`` holding the epicenter location. Default 3.

    Returns
    -------
    list of float
        Epicentral distances between each main event and every preceding
        event found within its time window, pooled across all main events.

    Raises
    ------
    ValueError
        If ``events`` is not a 2D array or does not contain
        ``epicenter_col``.
    """
    events = np.asarray(events, dtype=float)
    if events.ndim != 2 or events.shape[1] <= epicenter_col:
        raise ValueError(f"events must be a 2D array containing epicenter_col={epicenter_col}")

    events = events[np.argsort(events[:, 0])]
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitudes = np.log(events[:, 2] * element_size) / np.log(log_base)
    rel_events_row = np.flatnonzero(np.isfinite(magnitudes) & (magnitudes >= min_mag))

    r = []
    epicenters = events[:, epicenter_col] * element_size
    for row in rel_events_row:
        dt = events[row, 0] - events[:row, 0]
        if isinstance(preceeding_time, tuple):
            mask = (dt <= preceeding_time[1]) & (preceeding_time[0] <= dt) & (dt > 0)
        else:
            mask = (dt <= preceeding_time) & (dt > 0)
        r.extend(np.abs(epicenters[:row][mask] - epicenters[row]).tolist())
    return r

def plot_GR(
        events: np.ndarray,
        bin_size: float = 0.05,
        b_value: bool = False,
        b_range: Tuple[int, int] = (-8, 3),
        n_blocks: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 10),
        log_base: Optional[float] = np.e,
        title: Optional[str] = 'Gutenberg-Richter Distribution',
        fig_ax: Optional[Tuple[object, object]] = None,
        label: Optional[str] = None,
        linestyle: Optional[str] = '--',
        marker: Optional[str] = '.',
        markersize: Optional[float] = 5,
        linewidth: Optional[float] = 1,
        color: Optional[str] = 'b',
        alpha: Optional[float] = 1.0,
        cutoff: Optional[float] = None,
        cutoff_at_peak: Optional[bool] = None,
        element_size: Optional[float] = None,
) -> Optional[float]:
    """Plot a Gutenberg-Richter frequency-magnitude distribution.

      Event magnitudes are computed from column 2 of ``events`` (optionally
      scaled by ``element_size``) as ``log(size) / log(log_base)``, then
      optionally restricted to values above ``cutoff``. Magnitudes are binned
      with width ``bin_size``, counts are optionally normalized by
      ``n_blocks``, and empty bins are dropped. If ``cutoff_at_peak`` is set,
      bins below the peak-count bin are discarded. The resulting log-frequency
      vs. magnitude curve is plotted on ``fig_ax`` (or a new figure). If
      ``b_value`` is True, a linear regression of ``log(count)`` on magnitude
      is fit over ``b_range`` and its negative slope (the b-value) is
      returned; the fit line is drawn when a new figure was created, and
      appended to ``label`` when plotting onto an existing ``fig_ax``.

      Parameters
      ----------
      events : numpy.ndarray
          2D array of event records; column 2 holds the event size used to
          compute magnitude. Modified in place when ``element_size`` is given.
      bin_size : float, optional
          Width of the magnitude bins used for the frequency histogram.
          Default 0.05.
      b_value : bool, optional
          If True, fit a line to ``log(count)`` vs. magnitude over
          ``b_range`` and return its negative slope as the Gutenberg-Richter
          b-value. Default False.
      b_range : tuple of (int, int), optional
          Magnitude range over which the b-value linear fit is performed.
          Default ``(-8, 3)``.
      n_blocks : int, optional
          If provided, this value divides bin counts to normalize
          frequency (e.g., by number of simulation blocks or realizations).
      figsize : tuple of (int, int), optional
          Figure size used when a new figure is created (i.e. ``fig_ax`` is
          None). Default ``(15, 10)``.
      log_base : float, optional
          Base of the logarithm used both for computing magnitude and for the
          y-axis frequency scale. Default is the natural log base ``e``.
      title : str, optional
          Title applied to a newly created axes. Default
          ``'Gutenberg-Richter Distribution'``.
      fig_ax : tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes), optional
          Existing figure and axes to draw on. If None (default), a new
          figure and axes are created with grid lines and tick locators
          preconfigured.
      label : str, optional
          Legend label for the plotted curve. When ``b_value`` is True and
          ``fig_ax`` is provided, the fitted b-value is appended to this
          label.
      linestyle : str, optional
          Matplotlib line style for the plotted curve. Default '--'.
      marker : str, optional
          Matplotlib marker style for the plotted curve. Default '.'.
      markersize : float, optional
          Marker size for the plotted curve. Default 5.
      linewidth : float, optional
          Line width for the plotted curve. Default 1.
      color : str, optional
          Color of the plotted curve. Default 'b'.
      alpha : float, optional
          Opacity of the plotted curve. Default 1.0.
      cutoff : float, optional
          If provided, magnitudes at or below this value are excluded before
          binning.
      cutoff_at_peak : bool, optional
          If True, bins below the bin with the maximum count are discarded
          before plotting and (if applicable) fitting.
      element_size : float, optional
          If provided, event sizes in ``events[:, 2]`` are multiplied by this
          value before computing magnitude.

      Returns
      -------
      float or None
          The Gutenberg-Richter b-value (negative of the fitted slope) if
          ``b_value`` is True and a fit was performed; otherwise None.

      Raises
      ------
      ValueError
          If ``events`` is empty, or if ``bin_size`` is not positive.
      """
    if events.size == 0:
        raise ValueError("events array is empty")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")

    if not fig_ax:
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(visible=True, which='minor', linestyle='--')
        ax.grid(visible=True, which='major', linestyle='-')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.5))
        ax.minorticks_on()
    else:
        fig, ax = fig_ax
    events[:, 2] = events[:, 2] * element_size if element_size else events[:, 2]
    mags = np.log(events[:, 2]) / np.log(log_base)
    mags = mags[mags > cutoff] if cutoff else mags

    slope = None
    bins = np.arange(np.min(mags), np.max(mags) + bin_size, bin_size)
    R_mu, mu = np.histogram(mags, bins=bins)
    mu = mu[:-1]
    mask = R_mu > 0
    mu = mu[mask]
    R_mu = R_mu[mask] if not n_blocks else R_mu[mask] / n_blocks
    if cutoff_at_peak:
        mu = mu[np.where(R_mu == np.max(R_mu))[0][0]:]
        R_mu = R_mu[np.where(R_mu == np.max(R_mu))[0][0]:]

    if b_value:
        l_mask = (mu >= b_range[0]) & (mu <= b_range[1])
        slope, intercept, r, p, se = linregress(mu[l_mask], np.log(R_mu[l_mask]))
        if not fig_ax:
            ax.plot(mu[l_mask], slope * mu[l_mask] + intercept, 'r', alpha=0.5)
            ax.text(-2, 2, s=f'b value = {-1 * slope:.2f}', fontsize=25)

        else:
            label += f'\nb value: {slope:.2f}'

    if log_base == 10:
        ax.plot(mu, np.log10(R_mu), label=label, linestyle=linestyle, marker=marker, markersize=markersize,
                linewidth=linewidth, alpha=alpha, color=color)
        ax.set_ylabel('log₁₀(R(μ))', size=20)
    else:
        ax.plot(mu, np.log(R_mu), label=label, linestyle=linestyle, marker=marker, markersize=markersize,
                linewidth=linewidth, alpha=alpha, color=color)
        ax.set_ylabel('ln(R(μ))', size=20)

    ax.set_xlabel('Magnitude (μ)', size=15)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if slope:
        return slope
