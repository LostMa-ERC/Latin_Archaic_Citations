# Estimating the richness of a lost literature from citations

Code and data for **"Using Citations to Estimate the Richness of Lost Literary Heritage:
the case of Latin Archaic Poetry"**, by Émilie Guidi, Théo Moins and Jean-Baptiste Camps
(École nationale des chartes – PSL), accepted at the Computational Humanities Research
conference (CHR) and to appear in the *Anthology of Computers and the Humanities*.

Archaic Latin poetry survives through no manuscript at all: what we have are *fragments*, that
is, citations of these works embedded in later grammarians, commentators and literary authors.
The question this repository addresses is how many archaic works existed — including those that
no surviving source ever cited.

The data are a matrix `N[i, j]` = number of fragments of work *i* cited by source *j*:
**60 observed works × 167 citing sources, 2080 citations**.


## Repository layout

```
.
├── data/
│   └── corpus_fragments_normalise.xlsx   the corpus of fragments, one row per citation
├── notebooks/
│   ├── Poisson_lognormal.ipynb           main notebook (reference model)
│   └── Poisson_gamma.ipynb               companion notebook (Gamma–Poisson cross-check)
├── src/
│   ├── lost_texts.py                     code shared by the two notebooks
│   └── rhat_infinity.py                  local convergence diagnostic R-hat-infinity
├── scripts/
│   ├── lcmcr_check.py                    optional LCMCR benchmark (Python driver)
│   └── lcmcr_check.R                     optional LCMCR benchmark (R fits)
├── results/
│   ├── lcmcr_results.csv                 output of the LCMCR benchmark
│   ├── stau_lognormal.npz                S(tau) curve of the lognormal model
│   └── ppc_lognormal.npz                 predictive draws of the lognormal model
├── figures/                              figures of the paper, written by the notebooks
├── requirements.txt
└── README.md
```

| file | what it is |
|---|---|
| `notebooks/Poisson_lognormal.ipynb` | **Main notebook.** The reference Poisson-lognormal model, marginalised over the unobserved works and sampled by NUTS (numpyro): non-identifiability of $S$, calibration of $S(\tau)$, prior sensitivity, latent effects, posterior predictive check, comparison with the classical estimators. Also fits the exploratory cell-covariate extension of the paper's appendix. |
| `notebooks/Poisson_gamma.ipynb` | **Companion notebook.** The same matrix and the same estimand under a Gamma–Poisson model, sampled by Gibbs with data augmentation. Ends with the cross-model comparison ($S(\tau)$ curves and posterior predictive checks side by side). |
| `src/lost_texts.py` | Everything the two notebooks share: the loader, the cell-covariate design, the estimand $S(\tau)$, the simulation and calibration protocol, the predictive statistics, the classical estimators and the figures. Organised in nine numbered sections, listed in its module docstring. |
| `src/rhat_infinity.py` | Vendored reference implementation of the local convergence diagnostic $\hat R_\infty$ (Moins et al. 2025), used in place of the rank-normalised $\hat R$ because it detects chains that disagree only in the tail. |
| `scripts/lcmcr_check.py`, `scripts/lcmcr_check.R` | Optional external benchmark: fits latent-class capture–recapture (Manrique-Vallier 2016) under three definitions of a capture list. Requires R + GSL. |
| `results/lcmcr_results.csv` | Output of `lcmcr_check.py`, committed so that the notebooks run without R. |
| `results/stau_lognormal.npz`, `results/ppc_lognormal.npz` | Summaries exported by the lognormal notebook and read by the Gamma one, so that either notebook can be run on its own. |

## Reproducing

```bash
python3 -m venv frag_env
source frag_env/bin/activate
pip install -r requirements.txt

# four host devices, one per NUTS chain
XLA_FLAGS="--xla_force_host_platform_device_count=4" jupyter lab
```

The notebooks use paths relative to `notebooks/`, which is the working directory both of
`jupyter lab` and of `jupyter nbconvert`. Run `Poisson_lognormal.ipynb` first: it writes
`results/stau_lognormal.npz` and `results/ppc_lognormal.npz`, which `Poisson_gamma.ipynb` reads
for the cross-model comparison. Both files are committed, so the Gamma notebook also runs on its
own.

To re-execute non-interactively:

```bash
cd notebooks
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  jupyter nbconvert --to notebook --execute --inplace Poisson_lognormal.ipynb
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  jupyter nbconvert --to notebook --execute --inplace Poisson_gamma.ipynb
```


The optional LCMCR benchmark is regenerated with:

```bash
sudo apt install r-base libgsl-dev
Rscript -e 'install.packages("LCMCR")'
python scripts/lcmcr_check.py          # rewrites results/lcmcr_results.csv
```

## The data

`data/corpus_fragments_normalise.xlsx` has one row per citation (2080 rows). The columns used by
the code are:

| column | content |
|---|---|
| `Author`, `Work` | the cited (lost) work; they define the rows of `N`. A missing `Author` (anonymous works such as *Carmen Saliare*) is read as "Anonymous". |
| `Century` | century of composition of the work, in French notation (`IIe s. av. JC` = 2nd c. BCE, `IV-IIIe` = a range). |
| `Author_fragment`, `Work_fragment` | the citing source; they define the columns of `N` (see `col_scenario` below). |
| `Author_fragment_type` | category of the citing source: `grammairien` (grammarian), `littéraire` (literary author), `commentateur` (commentator). |
| `Transmission_century` | century of the citing source, same notation as `Century`. |

The other columns (`Fragment_number`, `Place_fragment`, `Fragment_length (word)`, `dubia`,
`Type vers`, `Voix`) are descriptive and are not used by the models. Two rows carry an editorial
note in an unnamed last column.

The century labels and the source-type values are in French, as encoded by the philologists who
built the corpus; `cent()` and `TYPE_KEYS` in `src/lost_texts.py` parse them, and everything
displayed is in English.

<!-- ### Two conventions worth knowing about

Both are arguments of `load_corpus` in `src/lost_texts.py`, and both are *choices* rather than
facts about the tradition:

- **Anonymous works** (*Carmen Saliare*, *Carmen Priami*, …) are kept, with `Author` filled as
  "Anonymous", rather than dropped.
- **`col_scenario=2`** (the default, used throughout the paper and the notebooks): a citing source
  is identified by the pair (citing author, citing work), which splits an author into one column
  per citing work and yields 167 columns. `col_scenario=1` identifies a source by its author alone
  and pools anonymous sources into a single column (73 columns); `col_scenario=3` treats each
  distinct anonymous source separately (107 columns). All incidence-based quantities — $Q_1$,
  $Q_2$, the jackknives, the effort shares — depend on this choice.

### Known inconsistencies in the data

The spreadsheet is published exactly as it was used for the paper, so that every number can be
reproduced. The following inconsistencies are known and were deliberately left uncorrected:

- **Three citing sources are each split into two columns by a spelling variant:** `Festus` /
  `␣Festus` (leading space, one citation), `Tusculanae disputationes` / `Tusculanae Disputationes`,
  and `scholia Lucani` / `*Scholia Lucani*`. Merging them would give $J=164$ columns instead of 167
  under `col_scenario=2` (72 instead of 73 under `col_scenario=1`), and a maximum of 100 instead of
  101 distinct sources for a single work. $S_{obs}$, $f_1$, $f_2$, $Q_1$, $Q_2$, Chao2, the
  jackknives and Chao–Bunge are unchanged.
- **Spelling variants of cited authors and works** that do not affect the matrix, because they
  never concern the same work: `Cn.Naevius` / `Cn. Naevius`, `Epigramma` / `epigramma`, a trailing
  space in `Porcius Licinus `, and `A. Atilius Catalinus` for *Calatinus*.
- **Missing centuries:** three works have no `Century` (*Carmen Saliare* and two of the three
  epigrams of Pompilius); they are imputed by the corpus mean. This matters only for the
  cell-covariate extension.
- The `dubia` column mixes the editorial status of the fragment with two stray metre values
  (`hexamètre dactylique`), one typo (`iincertae sedis`) and one note in French. It is not used by
  the models.
- Two citing sources are later than the 9th–10th century: a Vatican glossary (12th c.) and the
  *scholia* on Petronius (15th c.), one citation each.
 -->
## References

- Chao, A. (1987). Estimating the population size for capture–recapture data with unequal
  catchability. *Biometrics* 43, 783–791.
- Chao, A. & Bunge, J. (2002). Estimating the number of species in a stochastic abundance model.
  *Biometrics* 58, 531–539.
- Kestemont, M., Karsdorp, F. et al. (2022). Forgotten books: The application of unseen species
  models to the survival of culture. *Science* 375, 765–769.
- Manrique-Vallier, D. (2016). Bayesian population size estimation using Dirichlet process
  mixtures. *Biometrics* 72, 1246–1254.
- Moins, T., Arbel, J., Dutfoy, A. & Girard, S. (2025). On the use of a local $\hat R$ to improve
  MCMC convergence diagnostic. *Bayesian Analysis* 20(1), 131–156.
