# Estimating the richness of a lost literature from citations

Code and data for **"Using Citations to Estimate the Richness of Lost Literary Heritage:
the case of Latin Archaic Poetry"** (CHR 2027, under review).

Archaic Latin poetry survives through no manuscript at all: what we have are *fragments*, that
is, citations of these works embedded in later grammarians, commentators and literary authors.
The question this repository addresses is how many archaic works existed — including those that
no surviving source ever cited.

The data are a matrix `N[i, j]` = number of fragments of work *i* cited by source *j*:
**60 observed works × 167 citing sources, 2080 citations**.

## The main result, in one paragraph

The total richness **$S$ is not identifiable** from these data. It depends on extrapolating into
a region of near-zero citability that the corpus cannot constrain, and three independent
diagnostics say so: the estimate follows its prior, it fails to recover a known truth in
simulation, and a single extreme observation moves it by +148 %.

We therefore report the **threshold richness $S(\tau)$** — the number of works, seen or unseen,
whose probability of being cited at least once exceeds $\tau$. It is bounded (by $S_{obs}/\tau$,
with no model at all), well calibrated, and stable across the two parametric families:
**$S(0.5) \approx 51$** under *both* models, where they disagree about $S$ by a factor of more
than ten. Every classical estimator is then placed on the $S(\tau)$ curve at the threshold it
implicitly assumes — Chao2 turns out to be an estimate of $S(0.11)$, Chao–Bunge of $S(0.064)$.

## Contents

| file | what it is |
|---|---|
| `Poisson_lognormal.ipynb` | **Main notebook.** Poisson-lognormal model with cell covariates, marginalised over the unobserved works and sampled by NUTS (numpyro). Non-identifiability of $S$, calibration of $S(\tau)$, latent effects, posterior predictive check, comparison with the classical estimators. |
| `Poisson_gamma.ipynb` | **Companion notebook.** The same matrix and the same estimand under a Gamma–Poisson model, sampled by Gibbs with data augmentation. Ends with the cross-model comparison ($S(\tau)$ curves and posterior predictive checks side by side). |
| `lost_texts.py` | Everything the two notebooks share: the loader, the cell-covariate design, the estimand $S(\tau)$, the simulation and calibration protocol, the predictive statistics, the classical estimators and the figures. Organised in nine numbered sections, listed in its module docstring. |
| `rhat_infinity.py` | Vendored reference implementation of the local convergence diagnostic $\hat R_\infty$ (Moins et al. 2023), used in place of the rank-normalised $\hat R$ because it detects chains that disagree only in the tail. |
| `lcmcr_check.py`, `lcmcr_check.R` | Optional external benchmark: fits latent-class capture–recapture (Manrique-Vallier 2016) under three definitions of a capture list. Requires R + GSL. |
| `corpus_fragments_normalise.xlsx` | The corpus of fragments. One row per citation. |
| `lcmcr_results.csv` | Output of `lcmcr_check.py`, committed so the notebooks run without R. |
| `stau_lognormal.npz`, `ppc_lognormal.npz` | Summaries exported by the lognormal notebook and read by the Gamma one, so that either notebook can be run on its own. |

## Reproducing

```bash
python3 -m venv frag_env
source frag_env/bin/activate
pip install -r requirements.txt

# four host devices, one per NUTS chain
XLA_FLAGS="--xla_force_host_platform_device_count=4" jupyter lab
```

Run `Poisson_lognormal.ipynb` first: it writes `stau_lognormal.npz` and `ppc_lognormal.npz`,
which `Poisson_gamma.ipynb` reads for the cross-model figures. (Both files are committed, so the
Gamma notebook also runs on its own.)

To re-execute non-interactively:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  jupyter nbconvert --to notebook --execute --inplace Poisson_lognormal.ipynb
```

Expect roughly **75 min** for the lognormal notebook (dominated by the 12-replicate calibration
study at $J=167$) and **20 min** for the Gamma one (200,000 Gibbs iterations × 4 chains) on a
recent laptop CPU. No GPU is needed. Every sampler takes a `seed` and draws from an explicit
generator, so a given run reproduces exactly.

The optional LCMCR benchmark is regenerated with:

```bash
sudo apt install r-base libgsl-dev
Rscript -e 'install.packages("LCMCR")'
python lcmcr_check.py          # rewrites lcmcr_results.csv
```

## Two conventions worth knowing about

Both are arguments of `load_corpus` in `lost_texts.py`, and both are *choices* rather than facts
about the tradition:

- **Anonymous works** (*Carmen Saliare*, *Carmen Priami*, …) are kept, with `Author` filled as
  "Anonymous", rather than dropped.
- **`col_scenario=2`** (the default used throughout): a citing source is identified by the pair
  (citing author, citing work), which splits an author into one column per citing work and yields
  167 columns. `col_scenario=1` pools anonymous sources into a single column (73 columns);
  `col_scenario=3` treats each distinct anonymous source separately (107 columns). All
  incidence-based quantities — $Q_1$, $Q_2$, the jackknives, the effort shares — depend on this
  choice.

The century labels of the spreadsheet are in French (`IIe s. av. JC`), as are the source-type
values (`grammairien` / `littéraire` / `commentateur`); `cent()` and `TYPE_KEYS` parse them, and
everything displayed is in English.

## References

- Chao, A. (1987). Estimating the population size for capture–recapture data with unequal
  catchability. *Biometrics* 43, 783–791.
- Chao, A. & Bunge, J. (2002). Estimating the number of species in a stochastic abundance model.
  *Biometrics* 58, 531–539.
- Manrique-Vallier, D. (2016). Bayesian population size estimation using Dirichlet process
  mixtures. *Biometrics* 72, 1246–1254.
- Moins, T., Arbel, J., Dutfoy, A. & Girard, S. (2023). On the use of a local $\hat R$ to improve
  MCMC convergence diagnostic. *Bayesian Analysis* 18(4), 1033–1059.
- Kestemont, M., Karsdorp, F. et al. (2022). Forgotten books: The application of unseen species
  models to the survival of culture. *Science* 375, 765–769.
