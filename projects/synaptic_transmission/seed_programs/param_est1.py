import numpy as np


def parameter_estimator(data):
    """Estimate Neher-model parameters from depletion and recovery summaries."""
    release = np.asarray(data["release"], dtype=np.float64)
    time = np.asarray(data["time"], dtype=np.float64)
    if release.ndim == 1:
        release = release[None, :]
        time = time[None, :]

    eps = np.finfo(np.float64).eps
    protocols = []
    intervals = []
    first_values = []
    fractional_drops = []
    recovery_fractions = []
    recovery_times = []

    for y_row, t_row in zip(release, time):
        valid = np.isfinite(y_row) & np.isfinite(t_row)
        y = y_row[valid]
        t = t_row[valid]
        if y.size == 0:
            continue
        protocols.append(y)
        first_values.append(y[0])
        if y.size < 2:
            continue

        dt = np.diff(t)
        intervals.extend(dt[dt > 0.0])
        fractional_drops.extend(np.maximum(1.0 - y[1:] / np.maximum(y[:-1], eps), 0.0))

        gap_index = int(np.argmax(dt))
        depleted = y[gap_index]
        available_recovery = max(y[0] - depleted, eps)
        recovery_fractions.append(
            np.clip((y[gap_index + 1] - depleted) / available_recovery, 0.0, 1.0)
        )
        recovery_times.append(dt[gap_index])

    values = np.concatenate(protocols) if protocols else np.array([1.0])
    intervals = np.asarray(intervals, dtype=np.float64)
    first_release = max(float(np.median(first_values)), eps)
    typical_interval = max(float(np.median(intervals)), eps) if intervals.size else 1.0
    longest_interval = max(float(np.max(intervals)), typical_interval) if intervals.size else typical_interval
    shortest_interval = max(float(np.min(intervals)), eps) if intervals.size else typical_interval

    depletion = float(np.clip(np.max(fractional_drops), 0.05, 0.95)) if fractional_drops else 0.5
    recovery = float(np.clip(np.median(recovery_fractions), eps, 1.0)) if recovery_fractions else 0.5
    recovery_time = max(float(np.median(recovery_times)), typical_interval) if recovery_times else longest_interval

    # A recovered fraction r after delay t gives the first-order rate -log(1-r)/t.
    recovery_rate = -np.log(max(1.0 - recovery, eps)) / recovery_time
    resting_rate = max(recovery_rate, 1.0 / longest_interval)
    decay_time = np.sqrt(shortest_interval * longest_interval)

    Pf = depletion
    TS = first_release / Pf
    median_release = max(float(np.median(values)), eps)
    LS = median_release / Pf
    ES = TS * recovery
    Ntotal = ES + LS + TS

    # Response variability defines the arbitrary effective-calcium unit.  Only
    # ratios of this scale are identifiable from release observations.
    calcium_scale = max(float(np.std(values)), median_release, eps)
    pulse_transfer = float(np.clip(depletion * (1.0 - recovery), eps, 0.95))
    return {
        "b1": float(resting_rate * (1.0 - recovery)),
        "b2": float(resting_rate * (1.0 - depletion)),
        "b3": float(decay_time),
        "Pf": Pf,
        "k": pulse_transfer,
        "k_1": float(calcium_scale * median_release / first_release),
        "deltaca": calcium_scale,
        "cadecay": float(decay_time),
        "Crest": float(median_release),
        "k2rest": float(resting_rate * depletion),
        "k1rest": float(resting_rate * recovery),
        "s1": pulse_transfer,
        "s2": float(np.clip(depletion * (1.0 - recovery), eps, 0.95)),
        "Ntotal": float(Ntotal),
    }
