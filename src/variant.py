"""Tests one genuine ambiguity in the Section 4.2 description.

Section 4.2 says the delta-band oscillation reflects "thalamocortical slowing"
but does not say whether it is an independent oscillator per electrode or a
single shared generator projected to the scalp. Both are defensible readings.

Variant A (as implemented in simulate.py): independent per channel.
Variant B: one shared frequency and phase, per-channel amplitude.

This is a methodological question, not a search: the result is reported either
way.
"""

import numpy as np

import analysis
from montage import N_CHANNELS, indices, POSTERIOR, OCCIPITOTEMPORAL, FRONTOPARIETAL
from simulate import (FS, N_SAMPLES, N_PER_GROUP, NOISE_SD, DRIFT_AMP,
                      DRIFT_FREQ, PARAMS, SHARED_OT_FREQ, SHARED_FP_FREQ, _u)

TARGET = dict(n_sig=58, median=2.34, dmin=0.74, dmax=4.62)


def simulate_shared_delta(condition, seed):
    rng = np.random.default_rng(seed)
    p = PARAMS[condition]
    t = np.arange(N_SAMPLES) / FS
    x = rng.normal(0.0, NOISE_SD, size=(N_CHANNELS, N_SAMPLES))

    for ch in range(N_CHANNELS):
        x[ch] += _u(rng, DRIFT_AMP) * np.sin(
            2 * np.pi * _u(rng, DRIFT_FREQ) * t + rng.uniform(0, 2 * np.pi))

    # Variant B: one shared delta generator, per-channel amplitude.
    d_freq = _u(rng, p["delta_freq"])
    d_phase = rng.uniform(0, 2 * np.pi)
    d_wave = np.sin(2 * np.pi * d_freq * t + d_phase)
    for ch in range(N_CHANNELS):
        x[ch] += _u(rng, p["delta_amp"]) * d_wave

    for ch in indices(POSTERIOR):
        x[ch] += _u(rng, p["alpha_amp"]) * np.sin(
            2 * np.pi * _u(rng, p["alpha_freq"]) * t + rng.uniform(0, 2 * np.pi))

    ot_a, ot_p = _u(rng, p["occipitotemporal_amp"]), rng.uniform(0, 2 * np.pi)
    ot = np.sin(2 * np.pi * SHARED_OT_FREQ * t + ot_p)
    for ch in indices(OCCIPITOTEMPORAL):
        x[ch] += ot_a * ot

    fp_a, fp_p = _u(rng, p["frontoparietal_amp"]), rng.uniform(0, 2 * np.pi)
    fp = np.sin(2 * np.pi * SHARED_FP_FREQ * t + fp_p)
    for ch in indices(FRONTOPARIETAL):
        x[ch] += fp_a * fp

    return x


def run(seed_fn):
    recs, labels = [], []
    for condition, label in (("sham", 0), ("treatment", 1)):
        for i in range(N_PER_GROUP):
            recs.append(simulate_shared_delta(condition, seed_fn(condition, i)))
            labels.append(label)
    data, labels = np.asarray(recs), np.asarray(labels)
    X = analysis.build_features(data)
    st = analysis.compare_conditions(X, labels)
    d = np.abs(st["d"][st["reject"]])
    return dict(n_sig=int(st["reject"].sum()), median=float(np.median(d)),
                dmin=float(d.min()), dmax=float(d.max()))


if __name__ == "__main__":
    print("target: %d pairs, median %.2f, range %.2f-%.2f\n"
          % (TARGET["n_sig"], TARGET["median"], TARGET["dmin"], TARGET["dmax"]))
    schemes = {
        "1000+i / 2000+i": lambda c, i: (1000 if c == "sham" else 2000) + i,
        "i, 0..47":        lambda c, i: i if c == "sham" else N_PER_GROUP + i,
        "i, 1..48":        lambda c, i: (i + 1) if c == "sham" else N_PER_GROUP + i + 1,
    }
    for name, fn in schemes.items():
        r = run(fn)
        print("variant B, %-18s pairs=%3d  median=%.3f  min=%.3f  max=%.3f"
              % (name, r["n_sig"], r["median"], r["dmin"], r["dmax"]))
