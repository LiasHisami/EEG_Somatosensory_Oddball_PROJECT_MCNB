"""
06_regression_analysis.py
=========================
Applies simple Ordinary Least Squares (OLS) regression on single-epoch 
mean amplitudes to properly disentangle the Prediction Error (Deviance) 
from the physical stimulus properties (Intensity).

Model:
    Amplitude ~ Deviance + Intensity + Deviance * Intensity

Where:
    - Deviance: 0 = Standard, 1 = Deviant
    - Intensity: 0 = Low (64µA), 1 = High (128µA)

The interaction term (Deviance:Intensity) tests whether the prediction error 
response is genuinely modulated by the direction of intensity change.

Also performs ROI Sensitivity checks across:
    1. Primary ROI (C4, CP4, FC4)
    2. C4 Only
    3. Broad Contralateral ROI (C2, C4, FC2, FC4, CP2, CP4)

Outputs:
    derivatives/stats/regression_results.csv
"""

import numpy as np
import mne
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

project_dir = Path(__file__).resolve().parent.parent
epochs_dir = project_dir / "derivatives" / "epochs"
stats_dir = project_dir / "derivatives" / "stats"
stats_dir.mkdir(parents=True, exist_ok=True)

# ROI Definitions for Sensitivity Checks
ROIS = {
    "Primary": ["C4", "CP4", "FC4"],
    "C4_Only": ["C4"],
    "Broad_Contra": ["C2", "C4", "FC2", "FC4", "CP2", "CP4"],
}

# Time windows
COMPONENTS = {
    "N40":  (0.030, 0.045),
    "P50":  (0.050, 0.070),
    "P300": (0.250, 0.400),
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        1. LOAD EPOCHS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("=" * 60)
print("Loading cleaned epochs for Regression")
print("=" * 60)

epochs = mne.read_epochs(epochs_dir / "ID01_cleaned-epo.fif", preload=True)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  2. BUILD DATAFRAME & RUN MODELS                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

results_lines = []
results_lines.append("ROI,Component,Interaction_Coef,Interaction_P,MainEff_Deviance_P,MainEff_Intensity_P")

print("\nRunning OLS Regression Models: Amplitude ~ Deviance * Intensity")
print("-" * 60)

for roi_name, channels in ROIS.items():
    print(f"\nEvaluating ROI: {roi_name} ({', '.join(channels)})")
    
    # Check if all channels are in epochs (we interpolated some in 03, but not these)
    valid_channels = [ch for ch in channels if ch in epochs.ch_names]
    ch_idx = [epochs.ch_names.index(ch) for ch in valid_channels]

    for comp_name, (tmin, tmax) in COMPONENTS.items():
        
        # We need to construct a dataset for this specific component/ROI
        rows = []
        
        # Define condition mapping
        # Tuple: (ConditionName, Deviance(0=Std,1=Dev), Intensity(0=Low,1=High))
        conditions = [
            ("std_for_increase", 0, 0),  # 64µA Standard
            ("deviant_increase", 1, 1),  # 128µA Deviant
            ("std_for_decrease", 0, 1),  # 128µA Standard
            ("deviant_decrease", 1, 0),  # 64µA Deviant
        ]
        
        for cond_name, is_deviant, is_high in conditions:
            data = epochs[cond_name].get_data() # (n_epochs, n_channels, n_times)
            times = epochs.times
            time_mask = (times >= tmin) & (times <= tmax)
            
            # Mean over selected channels and time window -> (n_epochs,)
            amps = data[:, ch_idx, :][:, :, time_mask].mean(axis=(1, 2)) * 1e6 # µV
            
            for amp in amps:
                rows.append({
                    "Amplitude": amp,
                    "Deviance": is_deviant,
                    "Intensity": is_high
                })
                
        df = pd.DataFrame(rows)
        
        # Fit the regression model
        model = smf.ols("Amplitude ~ Deviance * Intensity", data=df).fit()
        
        # Extract p-values and coefficients
        p_dev = model.pvalues["Deviance"]
        p_int = model.pvalues["Intensity"]
        p_inter = model.pvalues["Deviance:Intensity"]
        coef_inter = model.params["Deviance:Intensity"]
        
        print(f"  {comp_name:4s} | Interaction p={p_inter:.4f} (Coef: {coef_inter:+.2f} µV)")
        
        results_lines.append(f"{roi_name},{comp_name},{coef_inter:.4f},{p_inter:.4f},{p_dev:.4f},{p_int:.4f}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           3. SAVE RESULTS                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

out_path = stats_dir / "regression_results.csv"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

print("\n" + "=" * 60)
print(f"Regression analysis complete. Results saved to {out_path}")
print("=" * 60)
