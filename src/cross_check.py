"""Cross-check: the self-contained NumPy implementation against SciPy/scikit-learn.

Both pipelines are run on the same simulated cohort and every stage is
compared. The NumPy implementation is not calibrated against these libraries
(see validate.py for its own validation); this script only demonstrates that
the two agree.
"""

import numpy as np
import analysis as np_pipeline
import reference_impl as ref
from simulate import simulate_cohort

data, labels = simulate_cohort()

Xa = np_pipeline.build_features(data)
Xb = ref.build_features(data)
print("coherence features   max |diff| = %.3e" % np.abs(Xa - Xb).max())

sa = np_pipeline.compare_conditions(Xa, labels)
sb = ref.compare_conditions(Xb, labels)
print("t statistics         max |diff| = %.3e" % np.abs(sa["t"] - sb["t"]).max())
print("p values             max |diff| = %.3e" % np.abs(sa["p"] - sb["p"]).max())
print("adjusted p values    max |diff| = %.3e" % np.abs(sa["p_adj"] - sb["p_adj"]).max())
print("Cohen's d            max |diff| = %.3e" % np.abs(sa["d"] - sb["d"]).max())
print("significant pairs    numpy=%d  scipy=%d  agree=%s"
      % (sa["reject"].sum(), sb["reject"].sum(),
         bool((sa["reject"] == sb["reject"]).all())))

ca = np_pipeline.loocv(Xa, labels)
cb = ref.loocv(Xb, labels)
print("LOOCV accuracy       numpy=%.4f  sklearn=%.4f" % (ca["accuracy"], cb["accuracy"]))

d_sig = np.abs(sa["d"][sa["reject"]])
import numpy_impl as N
print("n per group (median d=%.3f)   numpy=%s  scipy=%s"
      % (np.median(d_sig), N.n_per_group(float(np.median(d_sig))),
         ref.n_per_group(float(np.median(d_sig)))))
print("n per group (min d=%.3f)      numpy=%s  scipy=%s"
      % (d_sig.min(), N.n_per_group(float(d_sig.min())),
         ref.n_per_group(float(d_sig.min()))))
