"""
05_statistics.py
================
Statistical analysis for the somatosensory roving oddball.

Analyses (single-subject, epoch-level):
    1. Permutation tests: deviant vs. standard epochs at contralateral ROI
       — separately for increase and decrease directions
    2. Bootstrap confidence intervals for the difference waves
    3. Mean amplitude comparisons across N40, P50, P300 windows
    4. Direction asymmetry: increase vs. decrease difference waves

Since this is single-subject data, statistics are computed across
individual epochs (treating each epoch as an observation).

Outputs:
    derivatives/stats/     — results tables (CSV)
    derivatives/figures/   — statistical plots
"""

import numpy as np
import mne
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

project_dir = Path(__file__).resolve().parent.parent
epochs_dir = project_dir / "derivatives" / "epochs"
stats_dir = project_dir / "derivatives" / "stats"
figures_dir = project_dir / "derivatives" / "figures"

stats_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Channels of interest
CONTRA_CHANNELS = ["C4", "CP4", "FC4"]

# Time windows (seconds)
COMPONENTS = {
    "N40":  (0.030, 0.045),
    "P50":  (0.050, 0.070),
    "P300": (0.250, 0.400),
}

# Bootstrap parameters
N_BOOTSTRAP = 5000
BOOTSTRAP_CI = 95
RNG = np.random.default_rng(42)

# Permutation test parameters
N_PERMUTATIONS = 10000

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        1. LOAD EPOCHS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("=" * 60)
print("Loading cleaned epochs")
print("=" * 60)

epochs = mne.read_epochs(epochs_dir / "ID01_cleaned-epo.fif", preload=True)

# Get channel indices for ROI
ch_idx = [epochs.ch_names.index(ch) for ch in CONTRA_CHANNELS]

print(f"  ROI channels: {CONTRA_CHANNELS}")
for cond in epochs.event_id:
    print(f"  {cond}: {len(epochs[cond])} epochs")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║     2. EXTRACT SINGLE-EPOCH MEAN AMPLITUDES                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Extracting single-epoch mean amplitudes at ROI")
print("=" * 60)


def get_epoch_amplitudes(epochs_obj, condition, ch_indices, tmin, tmax):
    """
    Extract mean amplitude per epoch in a time window, averaged over
    the specified channels. Returns array of shape (n_epochs,) in µV.
    """
    data = epochs_obj[condition].get_data()  # (n_epochs, n_channels, n_times)
    times = epochs_obj.times
    time_mask = (times >= tmin) & (times <= tmax)
    # Average over channels and time → one value per epoch
    return data[:, ch_indices, :][:, :, time_mask].mean(axis=(1, 2)) * 1e6


# ╔══════════════════════════════════════════════════════════════════════╗
# ║     3. PERMUTATION t-TESTS: DEVIANT vs. STANDARD                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Permutation t-tests: deviant vs. matched standard")
print("=" * 60)

results_lines = []
results_lines.append(
    "direction,component,tmin,tmax,mean_std_uV,mean_dev_uV,mean_diff_uV,"
    "cohens_d,t_stat,p_value_perm"
)

comparisons = [
    ("Increase (64→128)", "std_for_increase", "deviant_increase"),
    ("Decrease (128→64)", "std_for_decrease", "deviant_decrease"),
]

for direction_label, std_cond, dev_cond in comparisons:
    print(f"\n  --- {direction_label} ---")

    for comp_name, (tmin, tmax) in COMPONENTS.items():
        amps_std = get_epoch_amplitudes(epochs, std_cond, ch_idx, tmin, tmax)
        amps_dev = get_epoch_amplitudes(epochs, dev_cond, ch_idx, tmin, tmax)

        # Cohen's d (independent samples)
        pooled_std = np.sqrt(
            ((len(amps_std) - 1) * amps_std.std(ddof=1) ** 2
             + (len(amps_dev) - 1) * amps_dev.std(ddof=1) ** 2)
            / (len(amps_std) + len(amps_dev) - 2)
        )
        cohens_d = (amps_dev.mean() - amps_std.mean()) / pooled_std if pooled_std > 0 else 0.0

        # Observed t-statistic
        t_obs, _ = stats.ttest_ind(amps_dev, amps_std)

        # Permutation test
        combined = np.concatenate([amps_dev, amps_std])
        n_dev = len(amps_dev)
        count_extreme = 0

        for _ in range(N_PERMUTATIONS):
            perm = RNG.permutation(combined)
            t_perm, _ = stats.ttest_ind(perm[:n_dev], perm[n_dev:])
            if abs(t_perm) >= abs(t_obs):
                count_extreme += 1

        p_perm = (count_extreme + 1) / (N_PERMUTATIONS + 1)  # +1 correction

        print(f"    {comp_name}: std={amps_std.mean():.2f} µV, "
              f"dev={amps_dev.mean():.2f} µV, "
              f"diff={amps_dev.mean() - amps_std.mean():.2f} µV, "
              f"d={cohens_d:.3f}, t={t_obs:.3f}, p_perm={p_perm:.4f}")

        results_lines.append(
            f"{direction_label},{comp_name},{tmin},{tmax},"
            f"{amps_std.mean():.4f},{amps_dev.mean():.4f},"
            f"{amps_dev.mean() - amps_std.mean():.4f},"
            f"{cohens_d:.4f},{t_obs:.4f},{p_perm:.4f}"
        )

# ╔══════════════════════════════════════════════════════════════════════╗
# ║     4. DIRECTION ASYMMETRY: INCREASE vs. DECREASE                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Direction asymmetry: increase vs. decrease deviants")
print("=" * 60)

results_lines.append("")  # blank line separator
results_lines.append(
    "comparison,component,tmin,tmax,mean_inc_uV,mean_dec_uV,mean_diff_uV,"
    "cohens_d,t_stat,p_value_perm"
)

for comp_name, (tmin, tmax) in COMPONENTS.items():
    # Compare deviant amplitudes across directions
    amps_dev_inc = get_epoch_amplitudes(
        epochs, "deviant_increase", ch_idx, tmin, tmax
    )
    amps_dev_dec = get_epoch_amplitudes(
        epochs, "deviant_decrease", ch_idx, tmin, tmax
    )

    pooled_std = np.sqrt(
        ((len(amps_dev_inc) - 1) * amps_dev_inc.std(ddof=1) ** 2
         + (len(amps_dev_dec) - 1) * amps_dev_dec.std(ddof=1) ** 2)
        / (len(amps_dev_inc) + len(amps_dev_dec) - 2)
    )
    cohens_d = (
        (amps_dev_inc.mean() - amps_dev_dec.mean()) / pooled_std
        if pooled_std > 0 else 0.0
    )

    t_obs, _ = stats.ttest_ind(amps_dev_inc, amps_dev_dec)

    combined = np.concatenate([amps_dev_inc, amps_dev_dec])
    n_inc = len(amps_dev_inc)
    count_extreme = 0

    for _ in range(N_PERMUTATIONS):
        perm = RNG.permutation(combined)
        t_perm, _ = stats.ttest_ind(perm[:n_inc], perm[n_inc:])
        if abs(t_perm) >= abs(t_obs):
            count_extreme += 1

    p_perm = (count_extreme + 1) / (N_PERMUTATIONS + 1)

    print(f"  {comp_name}: inc={amps_dev_inc.mean():.2f} µV, "
          f"dec={amps_dev_dec.mean():.2f} µV, "
          f"diff={amps_dev_inc.mean() - amps_dev_dec.mean():.2f} µV, "
          f"d={cohens_d:.3f}, t={t_obs:.3f}, p_perm={p_perm:.4f}")

    results_lines.append(
        f"inc_vs_dec_deviant,{comp_name},{tmin},{tmax},"
        f"{amps_dev_inc.mean():.4f},{amps_dev_dec.mean():.4f},"
        f"{amps_dev_inc.mean() - amps_dev_dec.mean():.4f},"
        f"{cohens_d:.4f},{t_obs:.4f},{p_perm:.4f}"
    )

# ╔══════════════════════════════════════════════════════════════════════╗
# ║     5. BOOTSTRAP CONFIDENCE INTERVALS FOR DIFFERENCE WAVES        ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print(f"Bootstrap {BOOTSTRAP_CI}% CIs for difference wave amplitudes")
print("=" * 60)

ci_lines = []
ci_lines.append("direction,component,tmin,tmax,mean_diff_uV,ci_lower,ci_upper")

for direction_label, std_cond, dev_cond in comparisons:
    print(f"\n  --- {direction_label} ---")

    for comp_name, (tmin, tmax) in COMPONENTS.items():
        amps_std = get_epoch_amplitudes(epochs, std_cond, ch_idx, tmin, tmax)
        amps_dev = get_epoch_amplitudes(epochs, dev_cond, ch_idx, tmin, tmax)

        # Bootstrap the mean difference
        boot_diffs = []
        for _ in range(N_BOOTSTRAP):
            boot_std = RNG.choice(amps_std, size=len(amps_std), replace=True)
            boot_dev = RNG.choice(amps_dev, size=len(amps_dev), replace=True)
            boot_diffs.append(boot_dev.mean() - boot_std.mean())

        boot_diffs = np.array(boot_diffs)
        alpha = (100 - BOOTSTRAP_CI) / 2
        ci_low = np.percentile(boot_diffs, alpha)
        ci_high = np.percentile(boot_diffs, 100 - alpha)
        mean_diff = amps_dev.mean() - amps_std.mean()

        print(f"    {comp_name}: diff={mean_diff:.2f} µV, "
              f"{BOOTSTRAP_CI}% CI=[{ci_low:.2f}, {ci_high:.2f}]")

        ci_lines.append(
            f"{direction_label},{comp_name},{tmin},{tmax},"
            f"{mean_diff:.4f},{ci_low:.4f},{ci_high:.4f}"
        )

# ╔══════════════════════════════════════════════════════════════════════╗
# ║     6. PLOT: BAR CHART OF MEAN AMPLITUDES                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Plotting bar charts")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Mean Amplitude by Component and Direction (Contralateral ROI)",
             fontsize=13)

for ax_idx, (comp_name, (tmin, tmax)) in enumerate(COMPONENTS.items()):
    ax = axes[ax_idx]

    # Gather means and SEMs
    labels = []
    means = []
    sems = []

    for cond_label, cond_key in [
        ("Std↑", "std_for_increase"),
        ("Dev↑", "deviant_increase"),
        ("Std↓", "std_for_decrease"),
        ("Dev↓", "deviant_decrease"),
    ]:
        amps = get_epoch_amplitudes(epochs, cond_key, ch_idx, tmin, tmax)
        labels.append(cond_label)
        means.append(amps.mean())
        sems.append(amps.std(ddof=1) / np.sqrt(len(amps)))

    x = np.arange(len(labels))
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]
    hatches = ["", "", "//", "//"]

    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{comp_name} ({tmin*1000:.0f}–{tmax*1000:.0f} ms)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")

fig.tight_layout()
fig.savefig(figures_dir / "stats_bar_amplitudes.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║     7. PLOT: DIFFERENCE WAVE WITH BOOTSTRAP CI                    ║
# ╔══════════════════════════════════════════════════════════════════════╝

print("  Plotting difference wave with bootstrap CI band...")

# Compute epoch-level difference wave time courses at ROI
def epoch_timecourse_at_roi(epochs_obj, condition, ch_indices):
    """Return (n_epochs, n_times) in µV, averaged over ROI channels."""
    data = epochs_obj[condition].get_data()[:, ch_indices, :]
    return data.mean(axis=1) * 1e6  # average over channels → µV


for direction_label, std_cond, dev_cond in comparisons:
    tc_std = epoch_timecourse_at_roi(epochs, std_cond, ch_idx)
    tc_dev = epoch_timecourse_at_roi(epochs, dev_cond, ch_idx)

    # Pointwise mean difference
    n_std, n_dev = tc_std.shape[0], tc_dev.shape[0]
    mean_diff = tc_dev.mean(axis=0) - tc_std.mean(axis=0)

    # Bootstrap CI at each time point
    n_times = len(epochs.times)
    boot_low = np.zeros(n_times)
    boot_high = np.zeros(n_times)

    boot_diffs_all = np.zeros((N_BOOTSTRAP, n_times))
    for b in range(N_BOOTSTRAP):
        b_std = tc_std[RNG.choice(n_std, size=n_std, replace=True)].mean(axis=0)
        b_dev = tc_dev[RNG.choice(n_dev, size=n_dev, replace=True)].mean(axis=0)
        boot_diffs_all[b] = b_dev - b_std

    alpha = (100 - BOOTSTRAP_CI) / 2
    boot_low = np.percentile(boot_diffs_all, alpha, axis=0)
    boot_high = np.percentile(boot_diffs_all, 100 - alpha, axis=0)

    times_ms = epochs.times * 1000

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(times_ms, boot_low, boot_high, alpha=0.25, color="steelblue",
                    label=f"{BOOTSTRAP_CI}% CI")
    ax.plot(times_ms, mean_diff, color="steelblue", linewidth=1.5,
            label="Mean difference")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    for comp_name, (t0, t1) in COMPONENTS.items():
        ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.08, color="orange")
        ax.text((t0 + t1) / 2 * 1000, ax.get_ylim()[1] * 0.9, comp_name,
                ha="center", fontsize=8, color="darkorange")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Difference Wave with {BOOTSTRAP_CI}% Bootstrap CI — "
                 f"{direction_label}")
    ax.legend(fontsize=9)
    fig.tight_layout()

    safe_label = direction_label.replace(" ", "_").replace("→", "-").replace("(", "").replace(")", "")
    fig.savefig(figures_dir / f"stats_diff_ci_{safe_label}.png", dpi=150)
    plt.close(fig)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     8. SAVE RESULTS                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Saving results")
print("=" * 60)

# Save permutation test results
with open(stats_dir / "permutation_test_results.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
print(f"  Permutation results: {stats_dir / 'permutation_test_results.csv'}")

# Save bootstrap CIs
with open(stats_dir / "bootstrap_ci_results.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(ci_lines))
print(f"  Bootstrap CIs:       {stats_dir / 'bootstrap_ci_results.csv'}")

print("\nSTATISTICAL ANALYSIS COMPLETE")
print(f"  Results: {stats_dir}")
print(f"  Figures: {figures_dir}")
