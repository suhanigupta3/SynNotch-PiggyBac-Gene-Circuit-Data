"""Self-contained NumPy implementation of the Section 4 analysis.

Depends on NumPy alone. Every component is validated against published
reference values rather than against the library it replaces; run
validate.py to reproduce those checks.

Contents
--------
butter_bandpass   4th-order Butterworth bandpass design (bilinear transform)
filtfilt          zero-phase forward-backward filtering with odd reflection
welch_coherence   magnitude-squared coherence by Welch's method
betainc_reg       regularized incomplete beta function (continued fraction)
t_sf              Student-t survival function
ttest_ind         independent two-sample t-test (equal variance)
benjamini_hochberg  Benjamini-Hochberg step-up FDR control
cohens_d          Cohen's d with pooled standard deviation
pca / svm / loocv  standardization, PCA and an RBF-kernel SVM trained by SMO
"""

import numpy as np

# --------------------------------------------------------------------------
# Filter design
# --------------------------------------------------------------------------

def _butter_analog_poles(order):
    """Poles of the analog Butterworth lowpass prototype (unit cutoff)."""
    k = np.arange(order)
    return np.exp(1j * np.pi * (2 * k + order + 1) / (2 * order))


def butter_bandpass(order, low_hz, high_hz, fs):
    """Digital bandpass Butterworth as second-order-free (b, a) coefficients.

    Matches the standard bilinear-transform design: the analog lowpass
    prototype is frequency-transformed to a bandpass and then mapped to the
    z-plane with frequency prewarping.
    """
    nyq = fs / 2.0
    # Prewarp, using the fs = 2 convention of the classical design.
    wl = 2.0 * 2.0 * np.tan(np.pi * (low_hz / nyq) / 2.0)
    wh = 2.0 * 2.0 * np.tan(np.pi * (high_hz / nyq) / 2.0)
    bw = wh - wl
    wo2 = wl * wh

    p_lp = _butter_analog_poles(order)

    # Lowpass -> bandpass: s -> (s^2 + wo^2) / (bw * s)
    poles = []
    for p in p_lp:
        half = p * bw / 2.0
        disc = np.sqrt(half ** 2 - wo2 + 0j)
        poles.append(half + disc)
        poles.append(half - disc)
    poles = np.asarray(poles)
    zeros = np.zeros(order, dtype=complex)      # order zeros at the origin

    # Analog gain so that the passband is unity at the geometric centre.
    wo = np.sqrt(wo2)
    num = np.prod(1j * wo - zeros)
    den = np.prod(1j * wo - poles)
    k = np.abs(den / num)

    # Bilinear transform with fs = 2.
    fs_bl = 2.0
    zd = (fs_bl * 2 + zeros) / (fs_bl * 2 - zeros)
    pd = (fs_bl * 2 + poles) / (fs_bl * 2 - poles)
    # Zeros at infinity map to z = -1.
    zd = np.concatenate([zd, -np.ones(len(pd) - len(zd))])
    kd = k * np.real(np.prod(fs_bl * 2 - zeros) / np.prod(fs_bl * 2 - poles))

    b = kd * np.real(np.poly(zd))
    a = np.real(np.poly(pd))
    return b, a


def butter_bandpass_zpk(order, low_hz, high_hz, fs):
    """Same design as butter_bandpass, returned as (zeros, poles, gain).

    The factored form is numerically stable at very low normalized
    frequencies, where evaluating the expanded polynomials loses precision to
    cancellation. Use this for magnitude-response checks.
    """
    nyq = fs / 2.0
    wl = 2.0 * 2.0 * np.tan(np.pi * (low_hz / nyq) / 2.0)
    wh = 2.0 * 2.0 * np.tan(np.pi * (high_hz / nyq) / 2.0)
    bw, wo2 = wh - wl, wl * wh
    poles = []
    for p in _butter_analog_poles(order):
        half = p * bw / 2.0
        disc = np.sqrt(half ** 2 - wo2 + 0j)
        poles.append(half + disc)
        poles.append(half - disc)
    poles = np.asarray(poles)
    zeros = np.zeros(order, dtype=complex)
    wo = np.sqrt(wo2)
    k = np.abs(np.prod(1j * wo - poles) / np.prod(1j * wo - zeros))
    fs_bl = 2.0
    zd = (fs_bl * 2 + zeros) / (fs_bl * 2 - zeros)
    pd = (fs_bl * 2 + poles) / (fs_bl * 2 - poles)
    zd = np.concatenate([zd, -np.ones(len(pd) - len(zd))])
    kd = k * np.real(np.prod(fs_bl * 2 - zeros) / np.prod(fs_bl * 2 - poles))
    return zd, pd, kd


def freq_response_zpk(zeros, poles, gain, f_hz, fs):
    """|H(f)| evaluated from the factored form."""
    z = np.exp(2j * np.pi * np.atleast_1d(f_hz) / fs)
    num = np.prod(z[:, None] - zeros[None, :], axis=1)
    den = np.prod(z[:, None] - poles[None, :], axis=1)
    return np.abs(gain * num / den)


def lfilter(b, a, x, zi=None):
    """Direct-form II transposed IIR filtering along the last axis.

    zi, when given, is the initial delay-line state with shape
    (n_signals, order).
    """
    b = np.asarray(b, dtype=float) / a[0]
    a = np.asarray(a, dtype=float) / a[0]
    n = max(len(a), len(b))
    b = np.concatenate([b, np.zeros(n - len(b))])
    a = np.concatenate([a, np.zeros(n - len(a))])
    x = np.atleast_2d(x)
    y = np.zeros_like(x)
    z = np.zeros((x.shape[0], n - 1)) if zi is None else np.array(zi, dtype=float)
    for i in range(x.shape[-1]):
        xi = x[:, i]
        yi = b[0] * xi + z[:, 0]
        for j in range(n - 2):
            z[:, j] = b[j + 1] * xi + z[:, j + 1] - a[j + 1] * yi
        z[:, n - 2] = b[n - 1] * xi - a[n - 1] * yi
        y[:, i] = yi
    return y


def lfilter_zi(b, a):
    """Steady-state delay-line state for a unit step input.

    Starting each filtering pass from a zero state makes the filter ring at
    the record edges. Because the analysis band here begins at 0.5 Hz, that
    transient lands squarely in the band of interest, so the steady-state
    initialisation is not cosmetic.
    """
    b = np.asarray(b, dtype=float) / a[0]
    a = np.asarray(a, dtype=float) / a[0]
    n = max(len(a), len(b))
    b = np.concatenate([b, np.zeros(n - len(b))])
    a = np.concatenate([a, np.zeros(n - len(a))])
    # Transposed companion matrix of the denominator.
    comp = np.zeros((n - 1, n - 1))
    comp[0, :] = -a[1:n]
    comp[1:, :-1] = np.eye(n - 2)
    return np.linalg.solve(np.eye(n - 1) - comp.T, b[1:n] - a[1:n] * b[0])


def filtfilt(b, a, x, padlen=None):
    """Zero-phase filtering: forward then backward, with odd padding and
    steady-state initial conditions on each pass."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    ntaps = max(len(a), len(b))
    if padlen is None:
        padlen = 3 * ntaps
    padlen = min(padlen, x.shape[-1] - 1)

    # Odd reflection about the endpoints.
    front = 2 * x[:, :1] - x[:, padlen:0:-1]
    back = 2 * x[:, -1:] - x[:, -2:-padlen - 2:-1]
    ext = np.concatenate([front, x, back], axis=-1)

    zi = lfilter_zi(b, a)
    y = lfilter(b, a, ext, zi=zi * ext[:, :1])
    y = lfilter(b, a, y[:, ::-1], zi=zi * y[:, -1:])[:, ::-1]
    return y[:, padlen:padlen + x.shape[-1]]


# --------------------------------------------------------------------------
# Spectral estimation
# --------------------------------------------------------------------------

def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _segments(x, nperseg, noverlap):
    step = nperseg - noverlap
    n = (x.shape[-1] - noverlap) // step
    idx = np.arange(nperseg)[None, :] + step * np.arange(n)[:, None]
    return x[..., idx]


def welch_csd(x, y, fs, nperseg=512, noverlap=256, detrend="constant"):
    """Cross- and auto-spectral densities by Welch's method.

    Each segment is mean-detrended before windowing, which is the standard
    default and prevents a non-zero segment mean from leaking power into the
    lowest frequency bins. That matters here because the analysis band starts
    at 0.5 Hz.
    """
    win = _hann(nperseg)
    scale = 1.0 / (fs * np.sum(win ** 2))
    xs = _segments(np.asarray(x, dtype=float), nperseg, noverlap)
    ys = _segments(np.asarray(y, dtype=float), nperseg, noverlap)
    if detrend == "constant":
        xs = xs - xs.mean(axis=-1, keepdims=True)
        ys = ys - ys.mean(axis=-1, keepdims=True)
    xs = xs * win
    ys = ys * win
    X = np.fft.rfft(xs, axis=-1)
    Y = np.fft.rfft(ys, axis=-1)
    pxy = np.mean(X * np.conj(Y), axis=-2) * scale
    pxx = np.mean(np.abs(X) ** 2, axis=-2) * scale
    pyy = np.mean(np.abs(Y) ** 2, axis=-2) * scale
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return freqs, pxx, pyy, pxy


def welch_coherence(x, y, fs, nperseg=512, noverlap=256, detrend="constant"):
    """Magnitude-squared coherence, |Pxy|^2 / (Pxx * Pyy)."""
    freqs, pxx, pyy, pxy = welch_csd(x, y, fs, nperseg, noverlap, detrend)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(pxy) ** 2 / (pxx * pyy)
    return freqs, np.nan_to_num(coh)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _lgamma(z):
    # Lanczos approximation, g = 7, n = 9.
    g = 7
    coef = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    z = float(z)
    if z < 0.5:
        return np.log(np.pi / np.sin(np.pi * z)) - _lgamma(1.0 - z)
    z -= 1.0
    a = coef[0]
    t = z + g + 0.5
    for i in range(1, g + 2):
        a += coef[i] / (z + i)
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(a)


def betainc_reg(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = np.exp(a * np.log(x) + b * np.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - np.exp(
        b * np.log(1.0 - x) + a * np.log(x) - lbeta
    ) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Survival function of Student's t: P(T > t)."""
    t = float(t)
    x = df / (df + t * t)
    p = 0.5 * betainc_reg(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def ttest_ind(a, b):
    """Independent two-sample t-test assuming equal variances."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    df = na + nb - 2
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / df
    denom = np.sqrt(sp2 * (1.0 / na + 1.0 / nb))
    t = (a.mean() - b.mean()) / denom if denom > 0 else 0.0
    p = 2.0 * t_sf(abs(t), df)
    return t, p


def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg step-up. Returns (reject mask, adjusted p-values)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out <= q, out


def cohens_d(a, b):
    """Cohen's d with pooled standard deviation."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def n_per_group(d, power=0.80, alpha=0.05):
    """Per-group n for a two-sided two-sample t-test, normal approximation
    refined by an exact-t power search."""
    if d == 0:
        return np.inf
    z_a, z_b = 1.959963984540054, 0.8416212335729143
    n0 = max(2, int(np.ceil(2 * (z_a + z_b) ** 2 / d ** 2)))
    for n in range(max(2, n0 - 8), n0 + 40):
        df = 2 * n - 2
        ncp = d * np.sqrt(n / 2.0)
        crit = t_ppf(1 - alpha / 2.0, df)
        if _nct_power(crit, df, ncp) >= power:
            return n
    return n0


def t_ppf(p, df, lo=-50.0, hi=50.0, tol=1e-10):
    """Inverse survival by bisection on the CDF."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _nct_power(crit, df, ncp, n_grid=20001):
    """Power of a two-sided t-test, by numerical integration over the
    noncentral t via its normal/chi mixture representation."""
    # T = (Z + ncp) / sqrt(V/df), V ~ chi2_df. Integrate over V.
    v = np.linspace(1e-6, df + 12 * np.sqrt(2 * df), n_grid)
    logpdf = (df / 2.0 - 1.0) * np.log(v) - v / 2.0 - (df / 2.0) * np.log(2.0) - _lgamma(df / 2.0)
    w = np.exp(logpdf)
    w /= np.trapezoid(w, v)
    s = np.sqrt(v / df)
    upper = _norm_sf(crit * s - ncp)
    lower = _norm_sf(-crit * s - ncp)
    return float(np.trapezoid(w * (upper + (1.0 - lower)), v))


def _norm_sf(x):
    return 0.5 * _erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def _erfc(x):
    """Numerical Recipes erfc, ~1.2e-7 relative accuracy."""
    z = np.abs(x)
    t = 2.0 / (2.0 + z)
    ty = 4.0 * t - 2.0
    cof = [-1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
           -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
           4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
           1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
           6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
           9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13,
           3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15]
    d = np.zeros_like(z)
    dd = np.zeros_like(z)
    for j in range(len(cof) - 1, 0, -1):
        tmp = d
        d = ty * d - dd + cof[j]
        dd = tmp
    ans = t * np.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd)
    return np.where(x >= 0.0, ans, 2.0 - ans)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return mu, sd


def pca_fit(X, n_components):
    mu = X.mean(axis=0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:n_components]


def pca_transform(X, mu, comps):
    return (X - mu) @ comps.T


class SVC:
    """RBF-kernel binary SVM trained by sequential minimal optimization."""

    def __init__(self, C=1.0, gamma="scale", tol=1e-3, max_passes=50):
        self.C, self.gamma, self.tol, self.max_passes = C, gamma, tol, max_passes

    def _k(self, A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self._gamma * np.maximum(d2, 0))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y) == 1, 1.0, -1.0)
        n = len(y)
        self._gamma = (1.0 / (X.shape[1] * X.var())) if self.gamma == "scale" else self.gamma
        K = self._k(X, X)
        a = np.zeros(n)
        b = 0.0
        passes = 0
        rng = np.random.default_rng(0)
        while passes < self.max_passes:
            changed = 0
            for i in range(n):
                Ei = (a * y) @ K[:, i] + b - y[i]
                if (y[i] * Ei < -self.tol and a[i] < self.C) or (y[i] * Ei > self.tol and a[i] > 0):
                    j = rng.integers(n - 1)
                    j = j + 1 if j >= i else j
                    Ej = (a * y) @ K[:, j] + b - y[j]
                    ai, aj = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    if L >= H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    a[j] = np.clip(aj - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(a[j] - aj) < 1e-8:
                        continue
                    a[i] = ai + y[i] * y[j] * (aj - a[j])
                    b1 = b - Ei - y[i] * (a[i] - ai) * K[i, i] - y[j] * (a[j] - aj) * K[i, j]
                    b2 = b - Ej - y[i] * (a[i] - ai) * K[i, j] - y[j] * (a[j] - aj) * K[j, j]
                    if 0 < a[i] < self.C:
                        b = b1
                    elif 0 < a[j] < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
        self._X, self._y, self._a, self._b = X, y, a, b
        return self

    def decision_function(self, X):
        return (self._a * self._y) @ self._k(self._X, np.asarray(X, dtype=float)) + self._b

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
