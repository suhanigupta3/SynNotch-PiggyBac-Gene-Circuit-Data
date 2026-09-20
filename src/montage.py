"""Standard 10-20 19-channel montage and the electrode groupings used by the
simulation described in Section 4.2 of the manuscript."""

# Display and indexing order follows the published Figure 2: the conventional
# 10-20 ordering, not alphabetical. Channel order is presentational only -- the
# set of 171 unordered pairs, and every statistic computed over it, is
# unaffected by it.
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Pz", "Cz",
]
N_CHANNELS = len(CHANNELS)           # 19
N_PAIRS = N_CHANNELS * (N_CHANNELS - 1) // 2   # 171

# Posterior electrodes carrying the alpha component.
POSTERIOR = ["P3", "Pz", "P4", "T5", "T6", "O1", "O2"]

# Occipitotemporal set carrying the shared 1.8 Hz component
# (aberrant local hyper-synchrony in the sham condition).
OCCIPITOTEMPORAL = ["T5", "T6", "O1", "O2"]

# Frontoparietal set carrying the shared 2.5 Hz component
# (long-range coupling, reduced in sham and restored in treatment).
FRONTOPARIETAL = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "P3", "Pz", "P4"]

IDX = {name: i for i, name in enumerate(CHANNELS)}


def indices(names):
    return [IDX[n] for n in names]


def pair_list():
    """Upper-triangle electrode pairs, row-major: 171 (i, j) tuples."""
    return [(i, j) for i in range(N_CHANNELS) for j in range(i + 1, N_CHANNELS)]


def pair_labels():
    return ["%s-%s" % (CHANNELS[i], CHANNELS[j]) for i, j in pair_list()]
