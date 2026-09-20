"""Validation of the self-contained NumPy implementation.

Each component is checked against published reference values rather than
against the library it replaces, as described in Section 4.2.
"""

import numpy as np
import numpy_impl as N
from simulate import FS

TOL = 1e-6
results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + " :: " + detail)


# 1. Butterworth magnitude response: 1/sqrt(2) at both band edges, unity in band.
_z, _p, _k = N.butter_bandpass_zpk(4, 0.5, 4.0, FS)


def _mag(f):
    return float(N.freq_response_zpk(_z, _p, _k, f, FS)[0])


check("butterworth_low_edge", abs(_mag(0.5) - 0.7071067812) < 1e-4,
      "|H(0.5 Hz)| = %.7f, expected 0.7071068" % _mag(0.5))
check("butterworth_high_edge", abs(_mag(4.0) - 0.7071067812) < 1e-4,
      "|H(4.0 Hz)| = %.7f, expected 0.7071068" % _mag(4.0))
check("butterworth_passband", abs(_mag(2.0) - 1.0) < 1e-2,
      "|H(2.0 Hz)| = %.7f, expected ~1" % _mag(2.0))

# 2. Regularized incomplete beta against standard tabulated values.
#    I_0.5(1,1) = 0.5 ; I_0.5(2,3) = 0.6875 ; I_0.25(0.5,0.5) = 1/3
for (x, aa, bb, want) in [(0.5, 1, 1, 0.5), (0.5, 2, 3, 0.6875),
                          (0.25, 0.5, 0.5, 1.0 / 3.0)]:
    got = N.betainc_reg(aa, bb, x)
    check("betainc_I_%g(%g,%g)" % (x, aa, bb), abs(got - want) < TOL,
          "got %.10f, expected %.10f" % (got, want))

# 3. t survival function returns 0.05000 at published one-tailed critical values.
for df, crit in [(1, 6.313752), (5, 2.015048), (10, 1.812461),
                 (30, 1.697261), (46, 1.678660), (120, 1.657651)]:
    got = N.t_sf(crit, df)
    check("t_sf_df%d" % df, abs(got - 0.05) < 5e-5,
          "P(T>%.6f | df=%d) = %.6f, expected 0.050000" % (crit, df, got))

# 4. Benjamini-Hochberg reproduces the worked example of Benjamini & Hochberg
#    (1995), Table 1: 15 p-values, 4 rejected at q = 0.05.
bh1995 = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
          0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
reject, _ = N.benjamini_hochberg(bh1995, 0.05)
check("benjamini_hochberg_1995", int(reject.sum()) == 4,
      "rejected %d of 15 hypotheses, expected 4" % int(reject.sum()))

# 5. Cohen's d against a hand-computable case.
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 4.0, 5.0, 6.0])
check("cohens_d", abs(N.cohens_d(y, x) - 1.5491933) < 1e-6,
      "d = %.7f, expected 1.5491933" % N.cohens_d(y, x))

print()
n_fail = sum(1 for _, ok, _ in results if not ok)
print("%d checks, %d failed" % (len(results), n_fail))
raise SystemExit(1 if n_fail else 0)
