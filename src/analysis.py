"""Section 4 analysis pipeline: filtering, coherence, statistics, classification.

Uses the self-contained NumPy implementation in numpy_impl.py. Run as a script
to reproduce every value reported in Section 4.3 and write results/ outputs.
"""

import json
import numpy as np

import numpy_impl as N
from montage import CHANNELS, N_CHANNELS, pair_list, pair_labels
from simulate import simulate_cohort, FS

BAND = (0.5, 4.0)
NPERSEG = 512          # 2 s at 256 Hz
NOVERLAP = 256         # 50 percent
N_COMPONENTS = 10
Q = 0.05


def coherence_features(record):
    """171 delta-band coherence values for one 19 x N record."""
    b, a = N.butter_bandpass(4, BAND[0], BAND[1], FS)
    filt = N.filtfilt(b, a, record)
    pairs = pair_list()
    out = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        freqs, coh = N.welch_coherence(filt[i], filt[j], FS, NPERSEG, NOVERLAP)
        band = (freqs >= BAND[0]) & (freqs <= BAND[1])
        out[k] = coh[band].mean()
    return out


def build_features(data):
    return np.asarray([coherence_features(r) for r in data])


def compare_conditions(X, labels):
    """Per-pair t-tests, BH FDR control, and effect sizes."""
    sham, treat = X[labels == 0], X[labels == 1]
    t = np.empty(X.shape[1])
    p = np.empty(X.shape[1])
    for k in range(X.shape[1]):
        t[k], p[k] = N.ttest_ind(treat[:, k], sham[:, k])
    reject, p_adj = N.benjamini_hochberg(p, Q)
    d = np.asarray([N.cohens_d(treat[:, k], sham[:, k]) for k in range(X.shape[1])])
    diff = treat.mean(0) - sham.mean(0)
    return dict(t=t, p=p, p_adj=p_adj, reject=reject, d=d, diff=diff)


def loocv(X, y, n_components=N_COMPONENTS):
    """Leave-one-out cross-validation. Standardization and PCA are fitted on
    the training partition only, inside each fold."""
    n = len(y)
    pred = np.empty(n, dtype=int)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr, ytr, Xte = X[mask], y[mask], X[i:i + 1]
        mu, sd = N.standardize_fit(Xtr)
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
        pmu, comps = N.pca_fit(Xtr_s, n_components)
        Ztr = N.pca_transform(Xtr_s, pmu, comps)
        Zte = N.pca_transform(Xte_s, pmu, comps)
        pred[i] = N.SVC(C=1.0, gamma="scale").fit(Ztr, ytr).predict(Zte)[0]
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(
        predictions=pred.tolist(),
        accuracy=(tp + tn) / n,
        sensitivity=tp / (tp + fn) if tp + fn else float("nan"),
        specificity=tn / (tn + fp) if tn + fp else float("nan"),
        confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn),
    )


def main():
    data, labels = simulate_cohort()
    X = build_features(data)
    stats = compare_conditions(X, labels)
    clf = loocv(X, labels)

    sig = stats["reject"]
    d_sig = np.abs(stats["d"][sig])
    summary = {
        "n_participants": int(len(labels)),
        "n_per_group": int((labels == 0).sum()),
        "n_pairs_tested": int(X.shape[1]),
        "n_pairs_significant": int(sig.sum()),
        "effect_size_median": float(np.median(d_sig)) if sig.any() else None,
        "effect_size_min": float(d_sig.min()) if sig.any() else None,
        "effect_size_max": float(d_sig.max()) if sig.any() else None,
        "n_per_group_for_median_effect":
            int(N.n_per_group(float(np.median(d_sig)))) if sig.any() else None,
        "n_per_group_for_smallest_effect":
            int(N.n_per_group(float(d_sig.min()))) if sig.any() else None,
        "classifier": clf,
    }

    np.save("results/coherence_features.npy", X)
    np.save("results/labels.npy", labels)
    np.savetxt("results/pairwise_statistics.csv",
               np.column_stack([stats["diff"], stats["t"], stats["p"],
                                stats["p_adj"], stats["d"], sig.astype(int)]),
               delimiter=",", header="mean_difference,t,p,p_adjusted,cohens_d,significant",
               comments="", fmt="%.10g")
    with open("results/pair_labels.txt", "w") as f:
        f.write("\n".join(pair_labels()))
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
