"""
01_define_events.py
===================
Exploratory script: extracts stimulus events from the raw BDF file and
identifies matched-standard / deviant pairs.

Trigger codes:
    64  — stimulus type A
    128 — stimulus type B
A "deviant" is any stimulus whose trigger differs from the immediately
preceding one; the preceding stimulus is its "matched standard".
"""

import mne
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
project_dir = Path(__file__).resolve().parent.parent
raw_path = project_dir / "original_data" / "01EEG" / "SPNCartoons_ID01.bdf"

# ── Load (header only) ────────────────────────────────────────────────
raw = mne.io.read_raw_bdf(raw_path, preload=False)

# ── Extract stimulus events ───────────────────────────────────────────
events = mne.find_events(raw, stim_channel="Status", shortest_event=1)
stim_events = events[np.isin(events[:, 2], [64, 128])]

print(f"Trigger 64 count:  {np.sum(stim_events[:, 2] == 64)}")
print(f"Trigger 128 count: {np.sum(stim_events[:, 2] == 128)}")

# ── Identify matched-standard / deviant pairs ─────────────────────────
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