# EEG Practical Final Project

## Background
The project uses a somatosensory roving oddball paradigm to investigate prediction and mismatch responses in the somatosensory system. 
The participant received electrical stimulation to their left wrist at two intensities (trigger values 64 and 128) while watching cartoons, which serve as a passive distraction task to keep attention away from the stimuli.
There were 12 blocks of 6min, with short breaks in between.

In a roving paradigm, one stimulus intensity repeats several times, allowing the brain to establish an expectation (standard). The intensity then switches, producing an unexpected (deviant) stimulus.

## Hypothesis
**Do somatosensory responses to unexpected intensity changes differ between low-to-high and high-to-low transitions?**
We hypothesize that early components (N40, P50) may be strongly influenced by physical stimulus intensity, whereas later responses (P300) may show direction-dependent effects after accounting for physical intensity.

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

### 3. Progressive Statistical Analysis (`05_statistics.py` & `06_regression_analysis.py`)
Because EEG data is non-normally distributed and this is a single-subject dataset, we began our analysis with non-parametric **Permutation t-tests** across single epochs (10,000 permutations). 
- **`05_statistics.py`:** Generates difference waves, bootstrap confidence intervals, and runs permutation tests comparing deviants vs standards. It initially compared increase-deviants directly against decrease-deviants to test for direction asymmetry.
- **The Confound:** We identified that comparing an increase deviant (physically a 128µA stimulus) directly to a decrease deviant (physically a 64µA stimulus) confounds *prediction error* with *physical intensity*.
- **`06_regression_analysis.py`:** To solve this, we implemented a **Simple Linear Regression** (`Amplitude ~ Deviance * Intensity`) on the single-epoch amplitudes for each time window. The interaction term (`Deviance:Intensity`) tests for pure direction asymmetry while controlling for the physical intensity of the stimuli. We also ran ROI sensitivity checks across different electrode clusters.

---

## Results & Key Findings
This single-subject analysis provides preliminary evidence for direction-dependent somatosensory responses:

1. **Early Components (N40, P50):** Initial permutation tests suggested direction differences across multiple time windows. However, the regression analysis indicated that the N40 and P50 interaction effects were no longer statistically reliable after accounting for physical intensity ($p > 0.11$). This suggests the apparent early direction differences may largely reflect physical intensity differences.
2. **Late Cognitive Component (P300):** The strongest evidence for a direction-dependent effect was observed in the P300 window. The regression model showed a significant interaction effect ($p = 0.0076$), consistent with a later direction-dependent mismatch response. However, these findings should be interpreted cautiously given the single-subject design, single-epoch statistics, and multiple-comparisons burden.

### Key Outputs to Review:
- **`derivatives/stats/regression_results.csv`**: Shows the interaction p-values that disentangle intensity from prediction error.
- **`stats_bar_amplitudes.png`**: Visually demonstrates the large P300 for the Increase condition.
- **`stats_diff_ci_Increase_64-128.png`** & **`stats_diff_ci_Decrease_128-64.png`**: Difference waves showing the mismatch responses.

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
python scripts/06_regression_analysis.py
```
