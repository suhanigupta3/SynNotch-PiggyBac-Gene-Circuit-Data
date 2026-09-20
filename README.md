# SynNotch-PiggyBac gene circuit: computational analysis

Code and outputs supporting the computational analysis in Section 4 of:

> Gupta S and Gupta R (2026). A SynNotch-PiggyBac gene circuit for
> inflammation-responsive IL-10 delivery in Alzheimer's disease: a hypothesis
> and design framework. *Frontiers in Aging Neuroscience* 18:1866161.
> doi: 10.3389/fnagi.2026.1866161

No human or animal data were used. Every record analysed here is **simulated**.
The analysis is an illustrative, fully specified workflow for a candidate
outcome measure. It is not evidence that the proposed circuit produces any
effect. Section 4.4 of the article states the limits of what it can support.

---

## Please read first: provenance of the reported values

The random seeds used for the run reported in the published Section 4.3 were
not preserved. This repository contains a complete, working and independently
validated implementation of the pipeline exactly as Section 4.2 specifies it,
run under a documented seed scheme (`seed = 1000 + i` for sham participants and
`2000 + i` for treatment-simulated participants, `i = 0..23`, in
`src/simulate.py`).

This run does not reproduce the published values digit for digit:

| Quantity | Published Section 4.3 | This repository |
|---|---|---|
| Electrode pairs tested | 171 | 171 |
| Pairs significant after FDR | 58 | 45 |
| Median absolute effect size | 2.34 | 2.80 |
| Smallest significant effect size | 0.74 | 1.26 |
| Largest significant effect size | 4.62 | 5.52 |
| n per group, median effect | 5 | 4 |
| n per group, smallest effect | 31 | 11 |
| LOOCV accuracy / sensitivity / specificity | 100% | 100% |

Every qualitative claim in Section 4.3 holds: a subset of pairs survives FDR
correction, the differences are large by construction, the dominant pattern is
increased frontoparietal coupling, and the classifier separates the two
conditions completely. Panel A of `results/figure2.png` reproduces the
published Figure 2 pattern, with the significant block falling on the
frontoparietal electrode set.

### What was tried to recover the original values

`src/recover.py` tests fourteen seed conventions a person would plausibly have
written: `default_rng(i)` over several offsets, per-group indexing, and the
legacy `RandomState` API. A scheme is accepted only if it reproduces *all four*
published quantities at once. None does. Across all fourteen, the significant
pair count falls in 47–53 and the median effect size in 2.60–3.72. The
published 58 lies outside that range and the published 2.34 below all of it, so
the difference is systematic rather than a matter of which seed was used.

`src/variant.py` tests the one substantive ambiguity in the Section 4.2 text.
The delta-band term is described as reflecting thalamocortical slowing, but the
article does not say whether it is an independent oscillator per electrode or a
single shared generator. Treating it as shared yields 125–132 significant
pairs, well above the published 58. Neither reading reproduces the reported
values; the published figure lies between them.

No further search was run. Choosing a seed or tuning a mixing parameter until
the output landed on 58 would produce a repository whose stated provenance was
false, and the fit would be undetectable from the outside, which is a reason
against it rather than for it.

The remaining underspecified elements, all documented in `src/montage.py` and
`src/simulate.py`, are the membership of the occipitotemporal and
frontoparietal electrode sets and the order in which per-channel parameters are
drawn. The electrode sets were checked against the published Figure 2, whose
significant block falls exactly on the ten frontoparietal electrodes used here.

**The values in the table above have been reported to the journal.** Do not
cite them as the published values, or the reverse, without checking which run
is meant.

---

## Layout

```
src/
  montage.py         19-channel 10-20 montage and electrode groupings
  simulate.py        EEG simulation, Section 4.2 parameters
  numpy_impl.py      self-contained implementation, NumPy only
  analysis.py        coherence, statistics and classification pipeline
  reference_impl.py  same pipeline using SciPy and scikit-learn
  cross_check.py     runs both implementations and compares every stage
  validate.py        validates numpy_impl against published reference values
  figure2.py         generates Figure 2
  recover.py         seed-convention recovery attempt (negative result)
  variant.py         shared vs independent delta generator (negative result)
results/
  summary.json             every value reported in Section 4.3
  pairwise_statistics.csv  per-pair difference, t, p, adjusted p, d, significance
  coherence_features.npy   48 x 171 feature matrix
  labels.npy               0 = sham, 1 = treatment-simulated
  pair_labels.txt          the 171 electrode pair names, in order
  figure2.png / .pdf       Figure 2
```

## Reproducing

```bash
pip install -r requirements.txt

python src/validate.py      # 14 checks against published reference values
python src/analysis.py      # writes results/summary.json
python src/figure2.py       # writes results/figure2.png and .pdf
python src/cross_check.py   # NumPy implementation vs SciPy/scikit-learn
```

`validate.py` and `analysis.py` need NumPy alone. `reference_impl.py`,
`cross_check.py` and `figure2.py` additionally need SciPy, scikit-learn and
Matplotlib.

Runtime is about 30 seconds for `analysis.py` and about 90 seconds for
`cross_check.py` on a laptop.

## The self-contained implementation

Section 4.2 describes an implementation depending on NumPy alone, so that the
analysis can be reproduced without matching a wider software stack.
`src/numpy_impl.py` provides it: Butterworth design by bilinear transform,
zero-phase forward-backward filtering with steady-state initial conditions,
magnitude-squared coherence by Welch's method, the regularized incomplete beta
function, the Student-t survival function, the Benjamini-Hochberg step-up
procedure, Cohen's d, PCA, and an RBF-kernel support vector machine trained by
sequential minimal optimization.

Each component is validated against published reference values rather than
against the library it replaces (`src/validate.py`, 14 checks):

- the Butterworth magnitude response is 0.7071068 at both band edges and
  unity in the passband;
- the regularized incomplete beta reproduces standard tabulated values;
- the t survival function returns 0.050000 at published one-tailed critical
  values for df = 1, 5, 10, 30, 46 and 120;
- the Benjamini-Hochberg routine reproduces the worked example of
  Benjamini and Hochberg (1995), rejecting 4 of 15 hypotheses at q = 0.05.

`cross_check.py` separately confirms that the NumPy implementation and the
SciPy/scikit-learn reference select the identical set of significant pairs and
agree on coherence to within 6e-4.

## Method summary

24 sham and 24 treatment-simulated participants; 30 s of 19-channel EEG at
256 Hz; Gaussian background noise at 18 µV SD and a low-frequency drift term,
generated identically in both conditions, so that all systematic separation
arises from four condition-specific features. Records are bandpass filtered at
0.5–4.0 Hz with a fourth-order Butterworth applied forward and backward,
magnitude-squared coherence is computed for each of the 171 electrode pairs by
Welch's method with 512-sample windows at 50% overlap and averaged across the
band, conditions are compared by independent-samples t-tests with
Benjamini-Hochberg FDR control at q < 0.05, and classification uses the 171
coherence values standardized and reduced to ten principal components inside
each fold, passed to an RBF-kernel SVM under leave-one-out cross-validation.

Full parameter values are in `src/simulate.py`.

## Licence

MIT. See `LICENSE`.
