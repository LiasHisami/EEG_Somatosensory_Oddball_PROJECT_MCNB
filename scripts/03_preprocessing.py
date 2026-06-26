"""
03_preprocessing.py
===================
Full EEG preprocessing pipeline for the somatosensory roving oddball
(SPNCartoons) paradigm.

Paradigm:
    Electrical stimulation to the LEFT wrist at two intensities:
        Trigger 64  = low intensity
        Trigger 128 = high intensity
    Roving oddball: one intensity repeats (standards), then switches (deviant).
    Cartoons serve as passive distraction.

Event coding (4 conditions):
    1 = matched_standard_for_increase  (last 64 before a 64→128 switch)
    2 = deviant_increase               (the 128 in a 64→128 switch)
    3 = matched_standard_for_decrease  (last 128 before a 128→64 switch)
    4 = deviant_decrease               (the 64 in a 128→64 switch)

Steps:
    1.  Load raw BDF
    2.  Set channel types (EXG → EOG/misc)
    3.  Set BioSemi64 montage
    4.  Bandpass filter (0.1–30 Hz)
    5.  Re-reference to average
    6.  ICA for EOG artifact removal
    7.  Extract events — direction-aware standard/deviant coding
    8.  Epoch (−0.2 to 0.5 s, baseline −0.2 to 0 s)
    9.  Reject noisy epochs (150 µV peak-to-peak)
   10.  Downsample to 256 Hz
   11.  Save cleaned epochs

Outputs:
    derivatives/epochs/    — cleaned epochs as MNE FIF
    derivatives/figures/   — preprocessing QC figures
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Paths
project_dir = Path(__file__).resolve().parent.parent
raw_path = project_dir / "original_data" / "01EEG" / "SPNCartoons_ID01.bdf"

epochs_dir = project_dir / "derivatives" / "epochs"
epochs_dir.mkdir(parents=True, exist_ok=True)

figures_dir = project_dir / "derivatives" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Filter settings
L_FREQ = 0.1        # Hz – high-pass
H_FREQ = 30.0       # Hz – low-pass

# ICA settings
ICA_N_COMPONENTS = 20       # number of ICA components to fit
ICA_RANDOM_STATE = 42       # for reproducibility
ICA_HIGHPASS = 1.0           # Hz – temporary high-pass for ICA fitting

# Epoch settings
TMIN = -0.2          # s – pre-stimulus baseline start
TMAX = 0.5           # s – post-stimulus end
BASELINE = (-0.2, 0) # baseline correction window
REJECT = dict(eeg=250e-6)  # 250 µV peak-to-peak rejection threshold

# Downsampling
RESAMPLE_FREQ = 256  # Hz

# Trigger codes
TRIG_LOW = 64        # low-intensity stimulation
TRIG_HIGH = 128      # high-intensity stimulation

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         1. LOAD RAW DATA                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

raw = mne.io.read_raw_bdf(raw_path, preload=True)
print(f"  Channels: {len(raw.ch_names)}")
print(f"  Sampling rate: {raw.info['sfreq']} Hz")
print(f"  Duration: {raw.times[-1]:.1f} s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     2. SET CHANNEL TYPES                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 2: Setting channel types")
print("=" * 60)

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

eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
print(f"  EEG channels: {len(eeg_channels)}")
print(f"  EOG channels: {len(mne.pick_types(raw.info, eog=True))}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        3. SET MONTAGE                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 3: Setting montage")
print("=" * 60)

montage = mne.channels.make_standard_montage("biosemi64")
raw.set_montage(montage, on_missing="warn")
print("  Montage: biosemi64")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       4. BANDPASS FILTER                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print(f"STEP 4: Bandpass filter ({L_FREQ}–{H_FREQ} Hz)")
print("=" * 60)

raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    5. AVERAGE RE-REFERENCE                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 5: Average re-reference")
print("=" * 60)

raw.set_eeg_reference("average", projection=False)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       6. ICA ARTIFACT REMOVAL                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 6: ICA for EOG artifact removal")
print("=" * 60)

# Fit ICA on a 1 Hz high-pass copy (better ICA decomposition)
raw_for_ica = raw.copy().filter(l_freq=ICA_HIGHPASS, h_freq=None)

ica = ICA(
    n_components=ICA_N_COMPONENTS,
    random_state=ICA_RANDOM_STATE,
    max_iter="auto",
)
ica.fit(raw_for_ica, picks="eeg")

print(f"  ICA fitted with {ica.n_components_} components")

# Auto-detect EOG-related components
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=["EXG1", "EXG2"])
ica.exclude = eog_indices
print(f"  EOG components detected: {eog_indices}")

# Plot ICA components for QC
fig = ica.plot_components(
    picks=range(min(ICA_N_COMPONENTS, ica.n_components_)), show=False
)
if isinstance(fig, list):
    for idx, f in enumerate(fig):
        f.savefig(figures_dir / f"ica_components_{idx}.png", dpi=150)
else:
    fig.savefig(figures_dir / "ica_components.png", dpi=150)

# Plot EOG scores
fig = ica.plot_scores(eog_scores, show=False)
fig.savefig(figures_dir / "ica_eog_scores.png", dpi=150)

# Apply ICA to the original (0.1 Hz high-pass) data
ica.apply(raw)
print("  ICA applied — EOG artifacts removed")

# Clean up temporary copy
del raw_for_ica

# ── Mark noisy channels as bad and interpolate ───────────────────────
# Fp1, Fp2, Fpz, AF3 are dominated by residual eye artifacts after ICA.
# O2 shows persistent high-amplitude noise.
# All are far from the somatosensory ROI and cause excessive epoch
# rejection.  Interpolating reconstructs them from neighbors so they
# still contribute to topographic maps.
bad_channels = ["Fp1", "Fp2", "Fpz", "AF3", "O2"]
raw.info["bads"] = bad_channels
raw.interpolate_bads(reset_bads=True)
print(f"  Interpolated bad channels: {bad_channels}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              7. EXTRACT EVENTS (DIRECTION-AWARE)                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 7: Extracting events (direction-aware)")
print("=" * 60)

events = mne.find_events(raw, stim_channel="Status", shortest_event=1)
stim_events = events[np.isin(events[:, 2], [TRIG_LOW, TRIG_HIGH])]

print(f"  Trigger {TRIG_LOW} (low) count:  {np.sum(stim_events[:, 2] == TRIG_LOW)}")
print(f"  Trigger {TRIG_HIGH} (high) count: {np.sum(stim_events[:, 2] == TRIG_HIGH)}")
print(f"  Total stimuli:          {len(stim_events)}")

# Identify direction-specific matched-standard / deviant pairs
#   Intensity increase: 64 → 128  (low → high)
#   Intensity decrease: 128 → 64  (high → low)

std_for_increase = []   # last low before low→high switch
dev_increase = []       # the high in a low→high switch
std_for_decrease = []   # last high before high→low switch
dev_decrease = []       # the low in a high→low switch

for i in range(1, len(stim_events)):
    prev_code = stim_events[i - 1, 2]
    curr_code = stim_events[i, 2]

    if prev_code == TRIG_LOW and curr_code == TRIG_HIGH:
        # Intensity INCREASE: 64 → 128
        std_for_increase.append(stim_events[i - 1].copy())
        dev_increase.append(stim_events[i].copy())
    elif prev_code == TRIG_HIGH and curr_code == TRIG_LOW:
        # Intensity DECREASE: 128 → 64
        std_for_decrease.append(stim_events[i - 1].copy())
        dev_decrease.append(stim_events[i].copy())

std_for_increase = np.array(std_for_increase)
dev_increase = np.array(dev_increase)
std_for_decrease = np.array(std_for_decrease)
dev_decrease = np.array(dev_decrease)

print(f"  Intensity increase (64→128):")
print(f"    Matched standards: {len(std_for_increase)}")
print(f"    Deviants:          {len(dev_increase)}")
print(f"  Intensity decrease (128→64):")
print(f"    Matched standards: {len(std_for_decrease)}")
print(f"    Deviants:          {len(dev_decrease)}")

# Assign new event codes
EVENT_ID = {
    "std_for_increase": 1,     # last 64 before 64→128 switch
    "deviant_increase": 2,     # the 128 in a 64→128 switch
    "std_for_decrease": 3,     # last 128 before 128→64 switch
    "deviant_decrease": 4,     # the 64 in a 128→64 switch
}

std_for_increase[:, 2] = EVENT_ID["std_for_increase"]
dev_increase[:, 2] = EVENT_ID["deviant_increase"]
std_for_decrease[:, 2] = EVENT_ID["std_for_decrease"]
dev_decrease[:, 2] = EVENT_ID["deviant_decrease"]

new_events = np.vstack([std_for_increase, dev_increase,
                         std_for_decrease, dev_decrease])
new_events = new_events[np.argsort(new_events[:, 0])]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         8. EPOCHING                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 8: Epoching")
print("=" * 60)

epochs = mne.Epochs(
    raw,
    events=new_events,
    event_id=EVENT_ID,
    tmin=TMIN,
    tmax=TMAX,
    baseline=BASELINE,
    picks="eeg",
    preload=True,
    reject=REJECT,
)

print(epochs)
for cond in EVENT_ID:
    n = len(epochs[cond])
    print(f"  {cond}: {n} epochs")
print(f"  Drop log: {epochs.drop_log_stats():.1f}% rejected")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       9. DOWNSAMPLE                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print(f"STEP 9: Downsampling to {RESAMPLE_FREQ} Hz")
print("=" * 60)

epochs.resample(RESAMPLE_FREQ)
print(f"  New sampling rate: {epochs.info['sfreq']} Hz")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       10. SAVE OUTPUTS                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("STEP 10: Saving cleaned epochs")
print("=" * 60)

epochs_fname = epochs_dir / "ID01_cleaned-epo.fif"
epochs.save(epochs_fname, overwrite=True)
print(f"  Saved: {epochs_fname}")

# Save drop log plot
fig = epochs.plot_drop_log(show=False)
fig.savefig(figures_dir / "preprocessing_drop_log.png", dpi=150)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Cleaned epochs: {epochs_fname}")
print(f"  QC figures:     {figures_dir}")
