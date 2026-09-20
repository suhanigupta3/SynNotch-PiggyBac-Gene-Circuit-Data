"""Resting-state EEG simulation for Section 4 of the manuscript.

Generates 24 sham (high-inflammation) and 24 treatment-simulated
(lower-inflammation) records, 30 s of 19-channel data at 256 Hz.

Parameter ranges are exactly those stated in Section 4.2. Background noise and
drift are generated identically in the two conditions, so that all systematic
separation between them arises from the four condition-specific features.
"""

import numpy as np
from montage import N_CHANNELS, indices, POSTERIOR, OCCIPITOTEMPORAL, FRONTOPARIETAL

FS = 256.0            # Hz
DURATION = 30.0       # seconds
N_SAMPLES = int(FS * DURATION)
N_PER_GROUP = 24

NOISE_SD = 18.0                    # microvolts
DRIFT_AMP = (4.0, 14.0)            # microvolts
DRIFT_FREQ = (0.05, 0.2)           # Hz

PARAMS = {
    "sham": {
        "delta_amp": (20.0, 32.0),
        "delta_freq": (1.5, 3.0),
        "alpha_amp": (6.0, 10.0),
        "alpha_freq": (9.0, 10.5),
        "occipitotemporal_amp": (15.0, 22.0),
        "frontoparietal_amp": (3.0, 6.0),
    },
    "treatment": {
        "delta_amp": (9.0, 16.0),
        "delta_freq": (1.5, 3.0),
        "alpha_amp": (15.0, 24.0),
        "alpha_freq": (9.0, 10.5),
        "occipitotemporal_amp": (5.0, 10.0),
        "frontoparietal_amp": (14.0, 22.0),
    },
}

SHARED_OT_FREQ = 1.8   # Hz
SHARED_FP_FREQ = 2.5   # Hz


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return rng.uniform(lo, hi)


def simulate_participant(condition, seed):
    """One participant's 19 x N_SAMPLES record, in microvolts."""
    rng = np.random.default_rng(seed)
    p = PARAMS[condition]
    t = np.arange(N_SAMPLES) / FS

    # Background: white noise, generated identically in both conditions.
    x = rng.normal(0.0, NOISE_SD, size=(N_CHANNELS, N_SAMPLES))

    # Low-frequency drift, per channel, also identical across conditions.
    for ch in range(N_CHANNELS):
        x[ch] += _u(rng, DRIFT_AMP) * np.sin(
            2 * np.pi * _u(rng, DRIFT_FREQ) * t + rng.uniform(0, 2 * np.pi)
        )

    # Feature 1: delta-band oscillation on every channel, independent phase.
    for ch in range(N_CHANNELS):
        x[ch] += _u(rng, p["delta_amp"]) * np.sin(
            2 * np.pi * _u(rng, p["delta_freq"]) * t + rng.uniform(0, 2 * np.pi)
        )

    # Feature 2: posterior alpha, independent phase per channel.
    for ch in indices(POSTERIOR):
        x[ch] += _u(rng, p["alpha_amp"]) * np.sin(
            2 * np.pi * _u(rng, p["alpha_freq"]) * t + rng.uniform(0, 2 * np.pi)
        )

    # Feature 3: shared 1.8 Hz component across occipitotemporal electrodes.
    # One phase for the whole set: this is what creates coherence between them.
    ot_amp = _u(rng, p["occipitotemporal_amp"])
    ot_phase = rng.uniform(0, 2 * np.pi)
    ot_wave = np.sin(2 * np.pi * SHARED_OT_FREQ * t + ot_phase)
    for ch in indices(OCCIPITOTEMPORAL):
        x[ch] += ot_amp * ot_wave

    # Feature 4: shared 2.5 Hz component across frontoparietal electrodes.
    fp_amp = _u(rng, p["frontoparietal_amp"])
    fp_phase = rng.uniform(0, 2 * np.pi)
    fp_wave = np.sin(2 * np.pi * SHARED_FP_FREQ * t + fp_phase)
    for ch in indices(FRONTOPARIETAL):
        x[ch] += fp_amp * fp_wave

    return x


def seed_for(condition, participant_index):
    """Deterministic distinct seed per participant.

    The seeds used for the originally reported run were not preserved. This
    scheme is documented, deterministic and distinct per participant, which is
    what Section 4.2 specifies; see README.md for the consequences.
    """
    base = 1000 if condition == "sham" else 2000
    return base + participant_index


def simulate_cohort():
    """Returns (data, labels): data is (48, 19, N_SAMPLES), labels 0=sham 1=treatment."""
    records, labels = [], []
    for condition, label in (("sham", 0), ("treatment", 1)):
        for i in range(N_PER_GROUP):
            records.append(simulate_participant(condition, seed_for(condition, i)))
            labels.append(label)
    return np.asarray(records), np.asarray(labels)


if __name__ == "__main__":
    data, labels = simulate_cohort()
    print("cohort:", data.shape, "labels:", np.bincount(labels))
