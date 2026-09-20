"""Bounded attempt to recover the seed convention used for the published run.

Only conventions a person would plausibly have written are tried, and a scheme
is accepted only if it reproduces EVERY published value simultaneously:
58 significant pairs, median |d| = 2.34, min 0.74, max 4.62.

Matching all four at once by chance is vanishingly unlikely, so a hit would be
genuine recovery. Matching one or two is not evidence of anything and is not
accepted. Nothing here searches for a seed that happens to give 58.
"""

import numpy as np
import analysis
import simulate
from simulate import PARAMS, N_PER_GROUP

TARGET = dict(n_sig=58, median=2.34, dmin=0.74, dmax=4.62)

SCHEMES = {
    "default_rng(1000+i) / (2000+i)": lambda c, i: (1000 if c == "sham" else 2000) + i,
    "default_rng(i), 0..47":          lambda c, i: i if c == "sham" else N_PER_GROUP + i,
    "default_rng(i), 1..48":          lambda c, i: (i + 1) if c == "sham" else N_PER_GROUP + i + 1,
    "default_rng(42+i)":              lambda c, i: 42 + (i if c == "sham" else N_PER_GROUP + i),
    "default_rng(i) per group, 0..23": lambda c, i: i,
    "default_rng(100+i) / (200+i)":   lambda c, i: (100 if c == "sham" else 200) + i,
    "default_rng(2024+i)":            lambda c, i: 2024 + (i if c == "sham" else N_PER_GROUP + i),
}

LEGACY = {"legacy RandomState(i), 0..47", "legacy RandomState(i), 1..48"}


def simulate_participant_legacy(condition, seed):
    """Same generative model driven by the legacy RandomState API."""
    rng = np.random.RandomState(seed)

    class Shim:
        uniform = staticmethod(rng.uniform)
        normal = staticmethod(rng.normal)

    orig = np.random.default_rng
    np.random.default_rng = lambda s: Shim
    try:
        return simulate.simulate_participant(condition, seed)
    finally:
        np.random.default_rng = orig


def cohort(seed_fn, legacy=False):
    recs, labels = [], []
    for condition, label in (("sham", 0), ("treatment", 1)):
        for i in range(N_PER_GROUP):
            s = seed_fn(condition, i)
            f = simulate_participant_legacy if legacy else simulate.simulate_participant
            recs.append(f(condition, s))
            labels.append(label)
    return np.asarray(recs), np.asarray(labels)


def evaluate(data, labels):
    X = analysis.build_features(data)
    st = analysis.compare_conditions(X, labels)
    d = np.abs(st["d"][st["reject"]])
    if d.size == 0:
        return None
    return dict(n_sig=int(st["reject"].sum()), median=float(np.median(d)),
                dmin=float(d.min()), dmax=float(d.max()))


def matches(r):
    return (r["n_sig"] == TARGET["n_sig"]
            and abs(r["median"] - TARGET["median"]) < 0.005
            and abs(r["dmin"] - TARGET["dmin"]) < 0.005
            and abs(r["dmax"] - TARGET["dmax"]) < 0.005)


if __name__ == "__main__":
    print("target: %d pairs, median %.2f, range %.2f-%.2f\n"
          % (TARGET["n_sig"], TARGET["median"], TARGET["dmin"], TARGET["dmax"]))
    hits = []
    for name, fn in SCHEMES.items():
        for legacy in (False, True):
            tag = name + (" [legacy RandomState]" if legacy else "")
            r = evaluate(*cohort(fn, legacy))
            if r is None:
                print("%-46s no significant pairs" % tag)
                continue
            ok = matches(r)
            hits.append(tag) if ok else None
            print("%-46s pairs=%3d  median=%.3f  min=%.3f  max=%.3f  %s"
                  % (tag, r["n_sig"], r["median"], r["dmin"], r["dmax"],
                     "*** FULL MATCH ***" if ok else ""))
    print()
    print("full matches: %s" % (hits if hits else "none"))
