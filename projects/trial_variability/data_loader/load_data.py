from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import jax.numpy as jnp
from edgar.data.neural import filtering, normalization, signal
from edgar.data.neural.signal import extract_stimulus_related


def load_data(
    data_path: str,
    activity_thresh: float = 0.4,
    conc_thresh: float = 0.55,
    n_bins: int = 256,
    show_plots: bool = False,
    shuffle_trials: bool = True,
    normalization_scheme: str = "vector",  # Options: "vector" or "peak"
    random_seed: int = 42,
):
    """
    Load and preprocess neural data for trial-to-trial variability modeling.

    Returns data in (n_samples, n_trials, n_cells) shape, where n_samples=1 for population modeling.
    """
    print(f"Loading data from {data_path}...")
    if "BZ015" in data_path:
        responses_raw, angles_raw = _load_bz015_data(
            data_path
        )  # (n_repeats, n_trials, n_cells), (n_repeats, n_trials)
    elif "stringer" in data_path:
        responses_raw, angles_raw = _load_stringer_data(
            data_path
        )  # (n_repeats, n_trials, n_cells), (n_repeats, n_trials)
    else:
        raise ValueError(
            f"Unrecognized dataset in path: {data_path}, must contain 'BZ015' or 'stringer'"
        )

    print(
        f"Raw responses: (n_repeats = {len(responses_raw)}, n_trials = {responses_raw[0].shape[0]}, n_cells = {responses_raw[0].shape[1]})"
    )
    print(
        f"Raw angles: (n_repeats = {len(angles_raw)}, n_trials = {angles_raw[0].shape[0]})"
    )
    responses_filtered = _filter_cells(
        responses_raw, angles_raw, activity_thresh, conc_thresh
    )

    # 2. Normalize, (optionally shuffle), partition
    resp_all = np.vstack(responses_filtered)
    ang_all = np.concatenate(angles_raw)
    print(
        f"Filtered (stacked) responses: (n_trials = {resp_all.shape[0]}, n_cells = {resp_all.shape[1]})"
    )
    print(f"Filtered (stacked) angles: (n_trials = {ang_all.shape[0]})")
    if resp_all.shape[0] // 2 < resp_all.shape[1]:
        warnings.warn(
            f"Number of trials per partition ({resp_all.shape[0] // 2}) is less than number of cells ({resp_all.shape[1]}). This may lead to problems with peer prediction as "
        )

    if normalization_scheme == "vector":
        resp_all = normalization.by_vector_norm(resp_all, axis=0)
    elif normalization_scheme == "peak":
        resp_all = normalization.by_peak(resp_all, axis=0)

    if shuffle_trials:
        # Shuffle trials globally across all repeats before partitioning
        rng = np.random.default_rng(random_seed)
        shuffled_idx = rng.permutation(len(resp_all))
        resp_all = resp_all[shuffled_idx]
        ang_all = ang_all[shuffled_idx]

    # Partition back into discovery and validation using a 50/50 split of the trials
    n_disc = len(resp_all) // 2
    resp_disc = resp_all[:n_disc]
    ang_disc = ang_all[:n_disc]
    resp_val = resp_all[n_disc:]
    ang_val = ang_all[n_disc:]

    # 3. Calculate Signal (Tuning Curves)
    sig_disc, bin_centers, avg_resp_disc = _get_signal(resp_disc, ang_disc, n_bins)
    sig_val, _, avg_resp_val = _get_signal(resp_val, ang_val, n_bins)

    # 4. Masking out of bottom right for train data
    disc_train = _apply_corner_mask(resp_disc, sig_disc, ang_disc)
    disc_test = {"response": resp_disc, "signal": sig_disc, "stimulus": ang_disc}
    val_train = _apply_corner_mask(resp_val, sig_val, ang_val)
    val_test = {"response": resp_val, "signal": sig_val, "stimulus": ang_val}

    # 5. Add Sample Dimension to ALL fields
    for d in [disc_train, disc_test, val_train, val_test]:
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v[np.newaxis, ...]

    # 6. Fingerprinting (Evaluation Set)
    eval_data = {**disc_train, "_sample_indices": np.array([0])}

    if show_plots:
        _plot_partitions(disc_train, disc_test, val_train, val_test)
        _plot_tuning_verification(
            np.vstack([resp_disc, resp_val]),
            np.concatenate([ang_disc, ang_val]),
            bin_centers,
            avg_resp_disc,
            random_seed,
        )

    return (
        (_to_jax(disc_train), _to_jax(disc_test)),
        (_to_jax(val_train), _to_jax(val_test)),
        _to_jax(eval_data),
    )


def _to_jax(d: dict) -> dict:
    """Recursively convert numpy arrays in a dictionary to JAX arrays."""
    return {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}


def _load_stringer_data(data_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    data = np.load(data_path, allow_pickle=True).item()
    responses_raw = extract_stimulus_related(data, n_pcs=0)  # (n_cells, n_trials)
    responses_raw = responses_raw.T  # (n_trials, n_cells)
    angles_raw = data["istim"]  # (n_trials,)
    return [responses_raw], [angles_raw]


def _load_bz015_data(data_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load BZ015 neural responses and stimulus angles from files."""
    _temp_dir = None
    # If a tarball is provided, extract it to a temporary directory and load from that (used with gcp)
    if data_path.endswith(".tar.gz") or data_path.endswith(".tgz"):
        _temp_dir = tempfile.TemporaryDirectory()
        print(f"Extracting dataset archive {data_path} to {_temp_dir.name}...")
        with tarfile.open(data_path, "r:gz") as tar:
            tar.extractall(path=_temp_dir.name)
        # BZ015 folder will be directly under the temporary directory
        data_path = str(Path(_temp_dir.name) / "BZ015")

    try:
        # 1. Load and Filter
        data_paths = (
            (
                data_path + "/BZ015_2025-07-03_2/BZ015_2025-07-03_2_dspikes.npy",
                data_path + "/BZ015_2025-07-03_2/2025-07-03_2_BZ015_Block.mat",
            ),
            (
                data_path + "/BZ015_2025-07-03_3/BZ015_2025-07-03_3_dspikes.npy",
                data_path + "/BZ015_2025-07-03_3/2025-07-03_3_BZ015_Block.mat",
            ),
            (
                data_path + "/BZ015_2025-07-03_5/BZ015_2025-07-03_5_dspikes.npy",
                data_path + "/BZ015_2025-07-03_5/2025-07-03_5_BZ015_Block.mat",
            ),
        )
        responses_raw, angles_raw = _load_raw_bz015_data(
            data_paths
        )  # (n_repeats, n_trials, n_cells), (n_repeats, n_trials)
    finally:  # ensure this runs even if an exception is raised, ensuring temp files are cleaned up
        if _temp_dir is not None:
            _temp_dir.cleanup()
    return responses_raw, angles_raw


def _load_raw_bz015_data(
    data_paths: tuple[tuple[str, str], ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load neural responses and stimulus angles from files."""
    responses, angles = [], []
    for d_file, m_file in data_paths:
        resp = np.load(d_file)  # (n_trials, n_cells)
        mat_data = sp.io.loadmat(m_file, simplify_cells=True)
        ang = np.array(
            [entry["gratingOrient"] for entry in mat_data["block"]["paramsValues"]]
        )

        # Filter out invalid stimulus values
        mask = ang != 1
        responses.append(resp[mask])
        angles.append(ang[mask])

    # Convert angles to radians and ensure they are within [0, 2*pi)
    for i in range(len(angles)):
        angles[i] = np.deg2rad(angles[i])
        angles[i][angles[i] >= 2 * np.pi] = 2 * np.pi - 1e-5

    return responses, angles


def _filter_cells(
    responses_list: list[np.ndarray],
    angles_list: list[np.ndarray],
    activity_thresh: float,
    conc_thresh: float,
) -> list[np.ndarray]:
    """Filter cells based on global activity and vector concentration."""
    all_responses = np.vstack(responses_list)
    all_angles = np.concatenate(angles_list)

    conc = filtering.vector_concentration(all_responses.T, all_angles)
    activity = filtering.activity(all_responses.T)

    good_cells_mask = (conc > conc_thresh) & (activity > activity_thresh)
    return [r[:, good_cells_mask] for r in responses_list]


def _get_signal(
    resp: np.ndarray, ang: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the binned mean signal (tuning curves) using only train trials to prevent data leakage."""
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit binned tuning curves using ONLY the first-half training trials
    trial_mid = len(resp) // 2
    avg_resp = signal.binned_mean(ang[:trial_mid], resp[:trial_mid].T, bin_centers)

    # Map these train-only tuning curves back to ALL trials of this partition
    _, bin_indices = signal.binned_mean(ang, resp.T, bin_centers, return_indices=True)
    sig_all = avg_resp[:, bin_indices].T

    return sig_all, bin_centers, avg_resp


def _apply_corner_mask(resp: np.ndarray, sig: np.ndarray, ang: np.ndarray):
    """Create train data by masking the bottom-right corner of the response matrix."""
    n_trials, n_cells = resp.shape
    trial_mid, cell_mid = n_trials // 2, n_cells // 2

    # Train: Mask out bottom right corner
    resp_train = resp.copy()
    resp_train[trial_mid:, cell_mid:] = np.nan
    # Use 0.0 for signal to avoid NaN gradients during optimization
    sig_train = sig.copy()
    sig_train[trial_mid:, cell_mid:] = 0.0

    return {"response": resp_train, "signal": sig_train, "stimulus": ang}


def _plot_partitions(disc_train, disc_test, val_train, val_test):
    # Set up 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # --- UNSORTED PLOTS ---
    # Col 0: Train Unsorted, Col 1: Test Unsorted
    partitions_original = [
        ("Discovery Train", disc_train, axes[0, 0]),
        ("Discovery Test", disc_test, axes[0, 1]),
        ("Validation Train", val_train, axes[1, 0]),
        ("Validation Test", val_test, axes[1, 1]),
    ]

    for name, data, ax in partitions_original:
        resp = data["response"][0]
        im = ax.imshow(resp, aspect="auto", cmap="viridis")
        ax.set_title(
            f"{name}\nShape: {resp.shape}\nMean: {np.nanmean(resp):.4e}, Std: {np.nanstd(resp):.4e}",
            fontsize=10,
        )
        ax.set_xlabel("Cells")
        ax.set_ylabel("Trials")
        fig.colorbar(im, ax=ax, label="Normalized Response")

    # --- SORTED PLOTS ---
    # Compute sorting indices for Discovery from disc_test
    # disc_resp_test = disc_test["response"][0]
    disc_sig_test = disc_test["signal"][0]
    disc_stims_test = disc_test["stimulus"][0]

    trial_sort_disc = np.argsort(disc_stims_test)
    complex_sum_disc = np.nansum(
        disc_sig_test * np.exp(1j * disc_stims_test)[:, np.newaxis], axis=0
    )
    pref_angles_disc = np.angle(complex_sum_disc) % (2 * np.pi)
    cell_sort_disc = np.argsort(pref_angles_disc)

    # Compute sorting indices for Validation from val_test
    # val_resp_test = val_test["response"][0]
    val_sig_test = val_test["signal"][0]
    val_stims_test = val_test["stimulus"][0]

    trial_sort_val = np.argsort(val_stims_test)
    complex_sum_val = np.nansum(
        val_sig_test * np.exp(1j * val_stims_test)[:, np.newaxis], axis=0
    )
    pref_angles_val = np.angle(complex_sum_val) % (2 * np.pi)
    cell_sort_val = np.argsort(pref_angles_val)

    partitions_sorted = [
        (
            "Discovery Train (Sorted)",
            disc_train,
            trial_sort_disc,
            cell_sort_disc,
            axes[0, 2],
        ),
        (
            "Discovery Test (Sorted)",
            disc_test,
            trial_sort_disc,
            cell_sort_disc,
            axes[0, 3],
        ),
        (
            "Validation Train (Sorted)",
            val_train,
            trial_sort_val,
            cell_sort_val,
            axes[1, 2],
        ),
        (
            "Validation Test (Sorted)",
            val_test,
            trial_sort_val,
            cell_sort_val,
            axes[1, 3],
        ),
    ]

    for name, data, trial_idx, cell_idx, ax in partitions_sorted:
        resp = data["response"][0]
        resp_sorted = resp[trial_idx][:, cell_idx]
        im = ax.imshow(resp_sorted, aspect="auto", cmap="viridis")
        ax.set_title(
            f"{name}\nShape: {resp_sorted.shape}\nMean: {np.nanmean(resp_sorted):.4e}, Std: {np.nanstd(resp_sorted):.4e}",
            fontsize=10,
        )
        ax.set_xlabel("Cells (Sorted by Preferred Angle)")
        ax.set_ylabel("Trials (Sorted by Stimulus Angle)")
        fig.colorbar(im, ax=ax, label="Normalized Response")

    plt.suptitle(
        "Trial variability data partitions (Original vs Sorted by Angle)", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("trial_variability_partitions.png")
    plt.close()


def _plot_tuning_verification(all_resp, all_ang, bin_centers, avg_resp, seed):
    rng = np.random.default_rng(seed)
    cell_idx = rng.integers(0, all_resp.shape[1])
    plt.figure(figsize=(10, 6))
    plt.scatter(
        all_ang,
        all_resp[:, cell_idx],
        alpha=0.3,
        s=10,
        label="Raw Responses",
        color="gray",
    )
    plt.plot(
        bin_centers,
        avg_resp[cell_idx, :],
        color="red",
        linewidth=2,
        label="Binned Mean (Tuning Curve)",
    )
    plt.title(f"Tuning Curve Verification - Cell {cell_idx}")
    plt.xlabel("Theta (radians)")
    plt.ylabel("Normalized Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trial_variability_tuning_curve.png")
    plt.close()


def loss_fn_train(model_output, data):
    filter = ~jnp.isnan(
        data["response"]
    )  # response set to NaN where we don't want to evaluate the loss
    return _filtered_loss_fn(model_output, data, filter)


def loss_fn_test(model_output, data):
    n_trials, n_cells = data["response"].shape[-2], data["response"].shape[-1]
    filter = jnp.zeros((n_trials, n_cells), dtype=bool)
    filter = filter.at[n_trials // 2 :, n_cells // 2 :].set(
        True
    )  # Set bottom-right corner to True, so only evaluate the loss there
    return _filtered_loss_fn(model_output, data, filter)


def _filtered_loss_fn(model_output, data, filter):
    """
    Mean squared error loss, only computed where the filter is True.
    Due to backpropagation in jax.grad need to ensure that there are no NaNs during the forward pass through the loss.
    Expects data['response'] of shape (n_samples, n_trials, n_cells).
    Returns (n_samples,) array of losses.
    """
    clean_response = jnp.where(
        filter, data["response"], 0.0
    )  # set the response to zero where filter is False, to avoid evaluating NaNs
    safe_model_output = jnp.where(
        filter, model_output, 0.0
    )  # set the prediction to zero where filter is False
    diff_sq = (
        clean_response - safe_model_output
    ) ** 2  # squared error where filter is True, zero where filter is False
    # Compute mean across trials and cells for unmasked entries
    total_error = jnp.sum(diff_sq, axis=(-2, -1))
    valid_count = jnp.sum(filter, axis=(-2, -1))

    return 1e4 * total_error / valid_count
