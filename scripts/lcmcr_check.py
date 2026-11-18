"""Fits the LCMCR model (Manrique-Vallier 2016) to the corpus, as an external benchmark.

Latent-class capture-recapture is the natural reference estimator of richness for a multiple-list
design such as this one: the dominant dependence between citing authors is common-cause (shared
florilegia and intermediary sources), which latent classes absorb by construction. What counts as
a "capture list" is not given by the data, however, so the check is run under the three plausible
definitions:

  sources   : one list per citing source (J = 167) -- the unit the two notebooks use;
  authors   : one list per citing author (col_scenario=1, J = 73);
  centuries : one list per transmission century (J = 13) -- the coarsest reading.

Each is fitted under every combination of truncation K in {4, 6, 8} and DP prior Gamma(a, a) with
a in {0.25, 1}, so that the notebooks can report the sensitivity of the answer to both.

Writes results/lcmcr_results.csv (one row per paradigm x K x prior), which the two notebooks read
without needing R. The intermediate incidence matrices (results/lcmcr_input_*.csv) are not
committed. Requires R with the LCMCR package (which itself needs GSL); the notebooks fall back on
the values quoted in their text if the file is absent.

Usage (from the repository root):  python scripts/lcmcr_check.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from lost_texts import load_corpus  # noqa: E402

PATH = os.path.join(ROOT, "data", "corpus_fragments_normalise.xlsx")
RESULTS = os.path.join(ROOT, "results")


def century_lists(corpus):
    """Aggregates the 167 source columns into one column per transmission century."""
    sc = corpus.sc.copy()
    sc[np.isnan(sc)] = np.nanmedian(sc)
    keys = np.round(sc).astype(int)
    uniq = np.unique(keys)
    M = np.zeros((corpus.Sobs, uniq.size))
    for k, u in enumerate(uniq):
        M[:, k] = corpus.N[:, keys == u].sum(1)
    labels = [f"c{u:+d}" for u in uniq]
    return M, labels


def main():
    c2 = load_corpus(PATH, col_scenario=2)
    c1 = load_corpus(PATH, col_scenario=1)
    Mc, lab_c = century_lists(c2)

    designs = {"sources": (c2.N, [f"s{k}" for k in range(c2.J)]),
               "authors": (c1.N, [f"a{k}" for k in range(c1.J)]),
               "centuries": (Mc, lab_c)}

    for name, (M, labels) in designs.items():
        X = (M > 0).astype(int)
        keep = X.sum(1) >= 1                       # LCMCR forbids all-zero rows
        X = X[keep]
        df = pd.DataFrame(X, columns=labels)
        df.to_csv(os.path.join(RESULTS, f"lcmcr_input_{name}.csv"), index=False)
        deg = X.sum(1)
        print(f"{name:10s} J={X.shape[1]:4d}  S_obs={X.shape[0]:3d}  "
              f"Q1={int((deg == 1).sum()):3d}  Q2={int((deg == 2).sum()):3d}  "
              f"max degree={int(deg.max())}")

    r = subprocess.run(["Rscript", os.path.join(HERE, "lcmcr_check.R")],
                       cwd=RESULTS, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"Rscript failed with code {r.returncode}")
    print(pd.read_csv(os.path.join(RESULTS, "lcmcr_results.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
