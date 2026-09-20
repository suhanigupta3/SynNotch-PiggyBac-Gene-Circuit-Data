"""Reference implementation of the Section 4 pipeline using SciPy and
scikit-learn.

This exists so that the self-contained NumPy implementation in numpy_impl.py
can be cross-checked against the standard scientific stack. cross_check.py
runs both and reports the agreement. The NumPy implementation is the one the
manuscript describes as carrying no dependency beyond NumPy.
"""

import numpy as np
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from montage import pair_list
from simulate import FS

BAND = (0.5, 4.0)
NPERSEG, NOVERLAP, N_COMPONENTS = 512, 256, 10


def coherence_features(record):
    b, a = signal.butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype="band")
    filt = signal.filtfilt(b, a, record, axis=-1)
    out = np.empty(len(pair_list()))
    for k, (i, j) in enumerate(pair_list()):
        f, coh = signal.coherence(filt[i], filt[j], fs=FS,
                                  nperseg=NPERSEG, noverlap=NOVERLAP)
        band = (f >= BAND[0]) & (f <= BAND[1])
        out[k] = coh[band].mean()
    return out


def build_features(data):
    return np.asarray([coherence_features(r) for r in data])


def compare_conditions(X, labels):
    sham, treat = X[labels == 0], X[labels == 1]
    t, p = stats.ttest_ind(treat, sham, axis=0)
    m = len(p)
    order = np.argsort(p)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    p_adj = np.empty(m)
    p_adj[order] = adj
    n1, n2 = len(treat), len(sham)
    sp = np.sqrt(((n1 - 1) * treat.var(0, ddof=1) + (n2 - 1) * sham.var(0, ddof=1))
                 / (n1 + n2 - 2))
    d = (treat.mean(0) - sham.mean(0)) / sp
    return dict(t=t, p=p, p_adj=p_adj, reject=p_adj <= 0.05, d=d,
                diff=treat.mean(0) - sham.mean(0))


def loocv(X, y, n_components=N_COMPONENTS):
    n = len(y)
    pred = np.empty(n, dtype=int)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sc = StandardScaler().fit(X[mask])
        pca = PCA(n_components=n_components).fit(sc.transform(X[mask]))
        clf = SVC(C=1.0, kernel="rbf", gamma="scale")
        clf.fit(pca.transform(sc.transform(X[mask])), y[mask])
        pred[i] = clf.predict(pca.transform(sc.transform(X[i:i + 1])))[0]
    return dict(predictions=pred.tolist(), accuracy=float((pred == y).mean()))


def n_per_group(d, power=0.80, alpha=0.05):
    from scipy.stats import nct, t as tdist
    if d == 0:
        return np.inf
    for n in range(2, 500):
        df = 2 * n - 2
        crit = tdist.ppf(1 - alpha / 2, df)
        ncp = d * np.sqrt(n / 2.0)
        if (1 - nct.cdf(crit, df, ncp)) + nct.cdf(-crit, df, ncp) >= power:
            return n
    return None
