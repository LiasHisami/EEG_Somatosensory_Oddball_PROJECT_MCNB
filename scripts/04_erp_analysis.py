"""
04_erp_analysis.py
==================
ERP computation and visualization for the somatosensory roving oddball.

Loads cleaned epochs from 03_preprocessing.py and produces:
    1. Grand-average ERPs for all 4 conditions
    2. Difference waves (deviant − matched standard) for each direction
    3. Direction comparison (increase vs decrease difference waves)
    4. Topographic maps at N40, P50, and P300 time windows
    5. Joint plots (waveform + topography)
    6. GFP (Global Field Power) for the difference waves

Key channels for somatosensory (left wrist → right hemisphere):
    Primary:   C4, CP4, FC4 (contralateral somatosensory cortex)
    Midline:   Cz, CPz, FCz
    Secondary: C3, CP3, FC3 (ipsilateral, for comparison)

Outputs:
    derivatives/evokeds/   — evoked FIF files
    derivatives/figures/   — all publication-quality figures
"""

import numpy as np
import mne
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

project_dir = Path(__file__).resolve().parent.parent
epochs_dir = project_dir / "derivatives" / "epochs"
evokeds_dir = project_dir / "derivatives" / "evokeds"
figures_dir = project_dir / "derivatives" / "figures"

evokeds_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Channels of interest — contralateral somatosensory cortex (right hemisphere)
CONTRA_CHANNELS = ["C4", "CP4", "FC4"]
MIDLINE_CHANNELS = ["Cz", "CPz", "FCz"]
IPSI_CHANNELS = ["C3", "CP3", "FC3"]
ROI_CHANNELS = CONTRA_CHANNELS + MIDLINE_CHANNELS  # primary ROI

# Time windows for components (seconds)
N40_WINDOW = (0.030, 0.050)   # N40 somatosensory component
P50_WINDOW = (0.040, 0.070)   # P50 somatosensory component
P300_WINDOW = (0.250, 0.400)  # P300 cognitive component

# Topography snapshot times (seconds)
TOPO_TIMES = [0.040, 0.055, 0.100, 0.150, 0.200, 0.300]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        1. LOAD EPOCHS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("=" * 60)
print("Loading cleaned epochs")
print("=" * 60)

epochs = mne.read_epochs(epochs_dir / "ID01_cleaned-epo.fif", preload=True)
print(epochs)
print(f"  Conditions: {list(epochs.event_id.keys())}")
for cond in epochs.event_id:
    print(f"    {cond}: {len(epochs[cond])} epochs")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    2. COMPUTE EVOKEDS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Computing evokeds")
print("=" * 60)

# Individual condition averages
evk_std_inc = epochs["std_for_increase"].average()
evk_dev_inc = epochs["deviant_increase"].average()
evk_std_dec = epochs["std_for_decrease"].average()
evk_dev_dec = epochs["deviant_decrease"].average()

# Set readable comments
evk_std_inc.comment = "std_for_increase"
evk_dev_inc.comment = "deviant_increase"
evk_std_dec.comment = "std_for_decrease"
evk_dev_dec.comment = "deviant_decrease"

# Difference waves: deviant − matched standard
diff_increase = mne.combine_evoked(
    [evk_dev_inc, evk_std_inc], weights=[1, -1]
)
diff_increase.comment = "diff_increase (dev_inc − std_inc)"

diff_decrease = mne.combine_evoked(
    [evk_dev_dec, evk_std_dec], weights=[1, -1]
)
diff_decrease.comment = "diff_decrease (dev_dec − std_dec)"

# Overall deviant vs standard (collapsed across direction)
evk_std_all = mne.combine_evoked(
    [evk_std_inc, evk_std_dec], weights="equal"
)
evk_std_all.comment = "standard_all"

evk_dev_all = mne.combine_evoked(
    [evk_dev_inc, evk_dev_dec], weights="equal"
)
evk_dev_all.comment = "deviant_all"

diff_all = mne.combine_evoked(
    [evk_dev_all, evk_std_all], weights=[1, -1]
)
diff_all.comment = "diff_all (deviant − standard)"

print("  Evokeds computed for all conditions")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              3. PLOT: CONDITION COMPARISON AT ROI                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Plotting ERPs")
print("=" * 60)

# --- 3a. Intensity INCREASE: standard vs deviant at contralateral ROI ---
fig = mne.viz.plot_compare_evokeds(
    {
        "Standard (low)": evk_std_inc,
        "Deviant (high)": evk_dev_inc,
    },
    picks=CONTRA_CHANNELS,
    combine="mean",
    title="Intensity Increase (64→128) — Contralateral ROI",
    show=False,
)
fig[0].savefig(figures_dir / "erp_increase_contra.png", dpi=150,
               bbox_inches="tight")

# --- 3b. Intensity DECREASE: standard vs deviant at contralateral ROI ---
fig = mne.viz.plot_compare_evokeds(
    {
        "Standard (high)": evk_std_dec,
        "Deviant (low)": evk_dev_dec,
    },
    picks=CONTRA_CHANNELS,
    combine="mean",
    title="Intensity Decrease (128→64) — Contralateral ROI",
    show=False,
)
fig[0].savefig(figures_dir / "erp_decrease_contra.png", dpi=150,
               bbox_inches="tight")

# --- 3c. All 4 conditions overlaid at contralateral ROI ---
fig = mne.viz.plot_compare_evokeds(
    {
        "Std for increase (low)": evk_std_inc,
        "Deviant increase (high)": evk_dev_inc,
        "Std for decrease (high)": evk_std_dec,
        "Deviant decrease (low)": evk_dev_dec,
    },
    picks=CONTRA_CHANNELS,
    combine="mean",
    title="All Conditions — Contralateral ROI (C4, CP4, FC4)",
    show=False,
)
fig[0].savefig(figures_dir / "erp_all_conditions_contra.png", dpi=150,
               bbox_inches="tight")

# --- 3d. Collapsed standard vs deviant ---
fig = mne.viz.plot_compare_evokeds(
    {
        "Standard (all)": evk_std_all,
        "Deviant (all)": evk_dev_all,
    },
    picks=CONTRA_CHANNELS,
    combine="mean",
    title="Overall Standard vs Deviant — Contralateral ROI",
    show=False,
)
fig[0].savefig(figures_dir / "erp_collapsed_contra.png", dpi=150,
               bbox_inches="tight")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              4. PLOT: DIFFERENCE WAVES                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# --- 4a. Difference waves: increase vs decrease direction ---
fig = mne.viz.plot_compare_evokeds(
    {
        "Diff: increase (64→128)": diff_increase,
        "Diff: decrease (128→64)": diff_decrease,
    },
    picks=CONTRA_CHANNELS,
    combine="mean",
    title="Difference Waves by Direction — Contralateral ROI",
    show=False,
)
fig[0].savefig(figures_dir / "erp_diff_by_direction_contra.png", dpi=150,
               bbox_inches="tight")

# --- 4b. Difference waves at midline ---
fig = mne.viz.plot_compare_evokeds(
    {
        "Diff: increase (64→128)": diff_increase,
        "Diff: decrease (128→64)": diff_decrease,
    },
    picks=MIDLINE_CHANNELS,
    combine="mean",
    title="Difference Waves by Direction — Midline (Cz, CPz, FCz)",
    show=False,
)
fig[0].savefig(figures_dir / "erp_diff_by_direction_midline.png", dpi=150,
               bbox_inches="tight")

# --- 4c. Overall difference wave ---
fig = mne.viz.plot_compare_evokeds(
    {"Deviant − Standard": diff_all},
    picks=CONTRA_CHANNELS,
    combine="mean",
    title="Overall Difference Wave — Contralateral ROI",
    show=False,
)
fig[0].savefig(figures_dir / "erp_diff_overall_contra.png", dpi=150,
               bbox_inches="tight")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              5. TOPOGRAPHIC MAPS                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("  Plotting topographic maps...")

# --- 5a. Difference wave topographies at key time points ---
fig = diff_increase.plot_topomap(
    times=TOPO_TIMES, average=0.010, show=False,
)
fig.suptitle("Diff Increase (64→128)")
fig.savefig(figures_dir / "topo_diff_increase.png", dpi=150,
            bbox_inches="tight")

fig = diff_decrease.plot_topomap(
    times=TOPO_TIMES, average=0.010, show=False,
)
fig.suptitle("Diff Decrease (128→64)")
fig.savefig(figures_dir / "topo_diff_decrease.png", dpi=150,
            bbox_inches="tight")

fig = diff_all.plot_topomap(
    times=TOPO_TIMES, average=0.010, show=False,
)
fig.suptitle("Diff Overall")
fig.savefig(figures_dir / "topo_diff_overall.png", dpi=150,
            bbox_inches="tight")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              6. JOINT PLOTS                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("  Plotting joint plots...")

fig = diff_increase.plot_joint(title="Diff Increase (64→128)", show=False)
fig.savefig(figures_dir / "joint_diff_increase.png", dpi=150,
            bbox_inches="tight")

fig = diff_decrease.plot_joint(title="Diff Decrease (128→64)", show=False)
fig.savefig(figures_dir / "joint_diff_decrease.png", dpi=150,
            bbox_inches="tight")

fig = diff_all.plot_joint(title="Overall Difference Wave", show=False)
fig.savefig(figures_dir / "joint_diff_overall.png", dpi=150,
            bbox_inches="tight")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              7. GFP (GLOBAL FIELD POWER)                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("  Plotting GFP...")

fig, ax = plt.subplots(figsize=(10, 4))
times = diff_increase.times * 1000  # convert to ms

gfp_inc = diff_increase.data.std(axis=0)
gfp_dec = diff_decrease.data.std(axis=0)
gfp_all = diff_all.data.std(axis=0)

ax.plot(times, gfp_inc * 1e6, label="Increase (64→128)", linewidth=1.5)
ax.plot(times, gfp_dec * 1e6, label="Decrease (128→64)", linewidth=1.5)
ax.plot(times, gfp_all * 1e6, label="Overall", linewidth=2, color="black",
        linestyle="--")

ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

# Shade component windows
ax.axvspan(N40_WINDOW[0] * 1000, N40_WINDOW[1] * 1000,
           alpha=0.1, color="blue", label="N40 window")
ax.axvspan(P50_WINDOW[0] * 1000, P50_WINDOW[1] * 1000,
           alpha=0.1, color="green", label="P50 window")
ax.axvspan(P300_WINDOW[0] * 1000, P300_WINDOW[1] * 1000,
           alpha=0.1, color="red", label="P300 window")

ax.set_xlabel("Time (ms)")
ax.set_ylabel("GFP (µV)")
ax.set_title("Global Field Power — Difference Waves")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(times[0], times[-1])
fig.tight_layout()
fig.savefig(figures_dir / "gfp_difference_waves.png", dpi=150)
plt.close(fig)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║         8. MEAN AMPLITUDE EXTRACTION (ROI × TIME WINDOW)          ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Mean amplitudes at contralateral ROI (C4, CP4, FC4)")
print("=" * 60)

def mean_amplitude(evoked, channels, tmin, tmax):
    """Extract mean amplitude in a time window at specified channels."""
    ch_idx = [evoked.ch_names.index(ch) for ch in channels]
    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    return evoked.data[np.ix_(ch_idx, time_mask)].mean() * 1e6  # µV

components = {
    "N40": N40_WINDOW,
    "P50": P50_WINDOW,
    "P300": P300_WINDOW,
}

conditions = {
    "Std for increase (low)": evk_std_inc,
    "Deviant increase (high)": evk_dev_inc,
    "Std for decrease (high)": evk_std_dec,
    "Deviant decrease (low)": evk_dev_dec,
    "Diff increase": diff_increase,
    "Diff decrease": diff_decrease,
    "Diff overall": diff_all,
}

print(f"\n{'Condition':<30} {'N40 (µV)':>10} {'P50 (µV)':>10} {'P300 (µV)':>10}")
print("-" * 62)
for cond_name, evoked in conditions.items():
    amps = {}
    for comp_name, (t0, t1) in components.items():
        amps[comp_name] = mean_amplitude(evoked, CONTRA_CHANNELS, t0, t1)
    print(f"{cond_name:<30} {amps['N40']:>10.2f} {amps['P50']:>10.2f} "
          f"{amps['P300']:>10.2f}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      9. SAVE EVOKEDS                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Saving evokeds")
print("=" * 60)

all_evokeds = [
    evk_std_inc, evk_dev_inc, evk_std_dec, evk_dev_dec,
    diff_increase, diff_decrease,
    evk_std_all, evk_dev_all, diff_all,
]

mne.write_evokeds(
    evokeds_dir / "ID01_somatosensory_oddball-ave.fif",
    all_evokeds,
    overwrite=True,
)

print(f"  Saved {len(all_evokeds)} evokeds to {evokeds_dir}")
print(f"  Figures saved to {figures_dir}")
print("\nERP ANALYSIS COMPLETE")
