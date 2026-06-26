# EEG Practical Final Project

## Background
The project uses a somatosensory roving oddball paradigm to investigate prediction and mismatch responses in the somatosensory system. 
The participant received electrical stimulation to their left wrist at two intensities (trigger values 64 and 128) while watching cartoons, which serve as a passive distraction task to keep attention away from the stimuli.
There were 12 blocks of 6min, with short breaks in between.

In a roving paradigm, one stimulus intensity repeats several times, allowing the brain to establish an expectation (standard). The intensity then switches, producing an unexpected (deviant) stimulus.

## Hypothesis
**The Effects of Stimulus Intensity Depend on Both the Direction of Change and ERP Component.**
Prediction error responses may differ depending on whether stimulus intensity increases or decreases. 
Furthermore, stimulus intensity is expected to influence early sensory processing components more strongly than later cognitive components.

Expected findings:
- N40 amplitudes will differ between intensity-increase deviants (64 → 128) and intensity-decrease deviants (128 → 64).
- The deviant–standard difference wave may be larger for one transition direction than the other, indicating asymmetric prediction error processing.
- P50 amplitudes will be modulated by stimulus intensity, reflecting differences in early sensory processing.
- In contrast, P300 amplitudes will show little or no modulation by stimulus intensity.

## Data
- **Participant**: ID01
- **System**: BioSemi Active-Two, 64 EEG channels + 8 external (EXG1–EXG8)
- **Sampling rate**: 2048 Hz
- **Format**: BDF
- **Paradigm**: Somatosensory roving oddball (left wrist electrical stimulation)
- **Trigger 64**: Low-intensity stimulation
- **Trigger 128**: High-intensity stimulation
- **Blocks**: 12, ~6 minutes each, variable ISI (~1.0–2.0 s)

## Event Coding
| Code | Condition | Description |
|------|-----------|-------------|
| 1 | `std_for_increase` | Last low-intensity (64) stimulus before a 64→128 switch |
| 2 | `deviant_increase` | High-intensity (128) stimulus in a 64→128 switch |
| 3 | `std_for_decrease` | Last high-intensity (128) stimulus before a 128→64 switch |
| 4 | `deviant_decrease` | Low-intensity (64) stimulus in a 128→64 switch |

---

## Analysis Pipeline

### 1. Preprocessing (`03_preprocessing.py`)
The goal is to clean the data without distorting the somatosensory response.
1. **Load raw BDF data** and set BioSemi64 montage.
2. **Bandpass filter (0.1–30 Hz)** to remove slow drifts and high-frequency noise.
3. **Average re-reference** using the 64 EEG channels (excluding EXG).
4. **ICA (Independent Component Analysis)** to identify and remove eye blink artifacts automatically.
5. **Channel Interpolation**: Marked excessively noisy channels (`Fp1`, `Fp2`, `Fpz`, `AF3`, `O2`) as bad and interpolated them. This prevents them from causing unnecessary epoch rejections, as they are far from our somatosensory Region of Interest (ROI).
6. **Epoching**: Extract data from −200 to 500 ms around each stimulus.
7. **Artifact Rejection**: Drop any epoch where a channel exceeds 250 µV (results in a very healthy ~6.7% rejection rate).
8. **Save cleaned epochs** to `derivatives/epochs/`.

### 2. ERP Analysis (`04_erp_analysis.py`)
Computes Evoked responses (averages) and creates visual plots. Focuses on the **Contralateral ROI** (`C4`, `CP4`, `FC4`) since stimulation was on the left wrist.
- **Key Output 1:** `derivatives/figures/erp_increase_contra.png` & `erp_decrease_contra.png` — Shows the raw Standard vs Deviant waveforms.
- **Key Output 2:** `derivatives/figures/erp_diff_by_direction_contra.png` — Plots the (Deviant - Standard) difference wave. Visually confirms the asymmetry hypothesis (the blue and orange lines go in opposite directions).
- **Key Output 3:** `derivatives/figures/topo_diff_overall.png` — Shows the brain-wide distribution of the mismatch response.

### 3. Statistics (`05_statistics.py`)
Because EEG data is non-normally distributed and this is a single-subject dataset, we used non-parametric **Permutation t-tests** across single epochs (10,000 permutations) rather than standard parametric tests. 
- **Key Output 1:** `derivatives/figures/stats_bar_amplitudes.png` — A bar chart showing the mean amplitude of each component (N40, P50, P300) with error bars.
- **Key Output 2:** `derivatives/figures/stats_diff_ci_increase.png` & `stats_diff_ci_decrease.png` — Difference waves plotted with 95% Bootstrap Confidence Intervals. Anywhere the shaded band doesn't cross zero is statistically significant.
- **Key Output 3:** `derivatives/stats/permutation_test_results.csv` — Contains the raw p-values and Cohen's d effect sizes.

---

## Results & Key Images to Review
The data **fully supports the Direction Asymmetry hypothesis**: the brain does not process all "surprises" equally; the response is heavily modulated by the physical direction of the intensity change.

To see the proof, look at these exact images (generated in `derivatives/figures/` after running the scripts):
1. **`stats_bar_amplitudes.png`**: The best summary image. It shows that for an Increase (64→128), the Deviant is always much *more positive* than the Standard. For a Decrease (128→64), the Deviant is always *more negative* (or closer to zero) than the Standard. 
2. **`stats_diff_ci_increase.png`**: Shows the massive positive prediction error (P300) when the shock is unexpectedly strong. The blue shaded area is the 95% Confidence Interval.
3. **`stats_diff_ci_decrease.png`**: Shows the early negative prediction error (N40) when the shock is unexpectedly weak.

---

## Data Directory Setup (For Group Members)
Because the raw EEG data files are very large (gigabytes), they are **not** included in this GitHub repository. Before running the scripts, you must manually create the data folders and place the raw `.bdf` file inside.

Your project folder must look exactly like this:
```
EEG_PRACTICAL/
├── original_data/            <-- YOU MUST CREATE THIS FOLDER
│   └── 01EEG/                <-- YOU MUST CREATE THIS FOLDER
│       └── ID01_...bdf       <-- PLACE THE RAW BDF FILE HERE
├── scripts/                  (From GitHub)
├── README.md                 (From GitHub)
└── environment.yml           (From GitHub)
```
The scripts will automatically create the `derivatives/` folder and output all figures there.
## Environment Setup (For Group Members)
If you are downloading this code for the first time, you need to create the exact same Python environment to run the scripts. 

Assuming you have Anaconda or Miniconda installed, run this command in your terminal inside the project folder:
```bash
conda env create -f environment.yml
```
This will automatically download and install `mne`, `numpy`, `scipy`, `matplotlib`, and `pandas` into a new environment called `eeg_mne`.

## Running the Pipeline
Once the environment is created, activate it and run the scripts in order:
```bash
conda activate eeg_mne
python scripts/03_preprocessing.py
python scripts/04_erp_analysis.py
python scripts/05_statistics.py
```
