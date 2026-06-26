"""
02_first_erp.py
===============
Quick first-pass ERP pipeline — loads raw BDF, applies minimal
preprocessing, epochs, averages, and plots standard vs. deviant ERPs.

This is a rapid-iteration script for sanity-checking the data.
For the full preprocessing pipeline (with ICA, etc.), see 03_preprocessing.py.

Fixes applied vs. original version:
    - Channel types set for EXG and Status channels
    - BioSemi64 montage applied
    - Bandpass filter applied *before* re-referencing
    - Amplitude-based epoch rejection added
    - All figures saved to derivatives/figures/
"""

import numpy as np
import mne
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
project_dir = Path(__file__).resolve().parent.parent
raw_path = project_dir / "original_data" / "01EEG" / "SPNCartoons_ID01.bdf"

figures_dir = project_dir / "derivatives" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

evokeds_dir = project_dir / "derivatives" / "evokeds"
evokeds_dir.mkdir(parents=True, exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────────────
raw = mne.io.read_raw_bdf(raw_path, preload=True)

# ── Set channel types ─────────────────────────────────────────────────
# EXG1/EXG2 → EOG (vertical/horizontal), EXG3-EXG8 → misc, Status → stim
raw.set_channel_types(
    {
        "EXG1": "eog",
        "EXG2": "eog",
        "EXG3": "misc",
        "EXG4": "misc",
        "EXG5": "misc",
        "EXG6": "misc",
        "EXG7": "misc",
        "EXG8": "misc",
    }
)

# ── Set montage ───────────────────────────────────────────────────────
montage = mne.channels.make_standard_montage("biosemi64")
raw.set_montage(montage, on_missing="warn")

# ── Bandpass filter (BEFORE re-referencing) ───────────────────────────
raw.filter(l_freq=0.1, h_freq=30.0, fir_design="firwin")

# ── Re-reference to average (EEG channels only) ──────────────────────
raw.set_eeg_reference("average", projection=False)

# ── Find stimulus events ─────────────────────────────────────────────
events = mne.find_events(raw, stim_channel="Status", shortest_event=1)
stim_events = events[np.isin(events[:, 2], [64, 128])]

print(f"Trigger 64 count:  {np.sum(stim_events[:, 2] == 64)}")
print(f"Trigger 128 count: {np.sum(stim_events[:, 2] == 128)}")

# ── Identify matched standards and deviants ───────────────────────────
matched_standards = []
deviants = []

for i in range(1, len(stim_events)):
    if stim_events[i, 2] != stim_events[i - 1, 2]:
        matched_standards.append(stim_events[i - 1].copy())
        deviants.append(stim_events[i].copy())

matched_standards = np.array(matched_standards)
deviants = np.array(deviants)

print(f"Matched standards: {len(matched_standards)}")
print(f"Deviants:          {len(deviants)}")

# ── Build new event array with readable IDs ───────────────────────────
standard_events = matched_standards.copy()
deviant_events = deviants.copy()

standard_events[:, 2] = 1
deviant_events[:, 2] = 2

new_events = np.vstack([standard_events, deviant_events])
new_events = new_events[np.argsort(new_events[:, 0])]

event_id = {
    "matched_standard": 1,
    "deviant": 2,
}

# ── Epoching ──────────────────────────────────────────────────────────
epochs = mne.Epochs(
    raw,
    events=new_events,
    event_id=event_id,
    tmin=-0.1,
    tmax=0.4,
    baseline=(-0.1, 0),
    picks="eeg",
    preload=True,
    reject=dict(eeg=150e-6),   # 150 µV peak-to-peak rejection
)

print(epochs)

# ── Downsample ────────────────────────────────────────────────────────
epochs.resample(200)

# ── Average ERPs ──────────────────────────────────────────────────────
evoked_standard = epochs["matched_standard"].average()
evoked_deviant = epochs["deviant"].average()

difference = mne.combine_evoked(
    [evoked_deviant, evoked_standard],
    weights=[1, -1],
)

# ── Plot & save ───────────────────────────────────────────────────────
# Butterfly plots
fig = evoked_standard.plot(show=False)
fig.savefig(figures_dir / "erp_standard_butterfly.png", dpi=150)

fig = evoked_deviant.plot(show=False)
fig.savefig(figures_dir / "erp_deviant_butterfly.png", dpi=150)

fig = difference.plot(show=False)
fig.savefig(figures_dir / "erp_difference_butterfly.png", dpi=150)

# Comparison at key channels
fig = mne.viz.plot_compare_evokeds(
    {
        "matched_standard": evoked_standard,
        "deviant": evoked_deviant,
    },
    picks=["Fz", "FCz", "Cz"],
    combine="mean",
    show=False,
)
fig[0].savefig(figures_dir / "erp_comparison_frontocentral.png", dpi=150)

# Single-channel comparison at Cz
fig = mne.viz.plot_compare_evokeds(
    {
        "matched_standard": evoked_standard,
        "deviant": evoked_deviant,
    },
    picks="Cz",
    show=False,
)
fig[0].savefig(figures_dir / "erp_comparison_Cz.png", dpi=150)

# Joint plots (waveform + topography)
fig = evoked_standard.plot_joint(show=False)
fig.savefig(figures_dir / "erp_standard_joint.png", dpi=150)

fig = evoked_deviant.plot_joint(show=False)
fig.savefig(figures_dir / "erp_deviant_joint.png", dpi=150)

fig = difference.plot_joint(show=False)
fig.savefig(figures_dir / "erp_difference_joint.png", dpi=150)

# ── Save evokeds ──────────────────────────────────────────────────────
mne.write_evokeds(
    evokeds_dir / "first_pass-ave.fif",
    [evoked_standard, evoked_deviant, difference],
    overwrite=True,
)

print(f"\nFigures saved to: {figures_dir}")
print(f"Evokeds saved to: {evokeds_dir}")