"""Shared code for the two companion notebooks on the estimation of the number of lost
Latin texts:

  * ``notebooks/Poisson_lognormal.ipynb`` -- Poisson-lognormal model (separable reference model,
    plus an exploratory cell-covariate extension), marginalised
    and sampled by NUTS (numpyro);
  * ``notebooks/Poisson_gamma.ipynb``     -- Gamma-Poisson model, sampled by Gibbs with data augmentation.

Both notebooks analyse **exactly the same matrix** N[i, j] = number of fragments of work i cited
by source j, and both target the same estimand S(tau) = the number of works whose detection
probability exceeds tau. Everything that does not depend on the parametric family therefore lives
here once: the loader, the cell-covariate design, S(tau), the calibration protocol, the posterior
predictive statistics, the non-parametric estimators and the shared figures.

Where a function must know which family it is serving, it takes a ``family`` argument (or reads it
from the ``aux`` dictionary returned by the fitting routine), with the two admissible values:

    ``"pln"`` -- Poisson-lognormal      ``"gp"`` -- Gamma-Poisson

Every stochastic routine draws from an explicit ``numpy`` generator seeded by its ``seed``
argument (never from the global ``numpy.random`` state), so that a given call reproduces its
figures and tables exactly.

Only ``numpy``/``pandas``/``matplotlib``/``scipy`` are imported at module level; ``jax`` and
``numpyro`` are imported lazily inside :func:`fit_pln`, so that the Gamma-Poisson notebook can use
this module without them.

Contents
--------
    0. Visual house style
    1. The corpus: loading, chronology, cell-covariate design
    2. Poisson-lognormal model (marginalised, sampled by NUTS)
    3. Gamma-Poisson model (Gibbs sampler with data augmentation), and R-hat-infinity
    4. The shared estimand: threshold richness S(tau)
    5. Simulation of a complete corpus, recovery and calibration
    6. Classical richness estimators, and what threshold they implicitly assume
    7. Posterior predictive checking
    8. Latent effects: who transmits, which works are citable
"""

from __future__ import annotations

import colorsys
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import gammaln
from scipy.stats import gaussian_kde

__all__ = [
    # visual house style
    "setup_style", "LABEL_FS", "TICK_FS", "LEGEND_FS", "TITLE_FS",
    "PANEL_TITLE_FS", "PANEL_LABEL_FS", "PANEL_TICK_FS",
    "MULTI_TITLE_FS", "MULTI_LABEL_FS", "MULTI_TICK_FS", "MULTI_LEGEND_FS",
    # corpus
    "TYPE_KEYS", "TYPES_EN", "ROMAN", "cent", "sum_to_zero_basis", "H3",
    "Corpus", "load_corpus", "cell_design", "independence_deviance",
    # Poisson-lognormal
    "setup_numpyro", "build_strata", "fit_pln", "rhat_ess",
    # Gamma-Poisson
    "slice_shape", "gamma_and_log", "gibbs", "merge_chains", "print_rhat_table",
    # convergence diagnostic (R-hat-infinity, Moins et al. 2025)
    "rhat_inf", "rhat_inf_threshold", "R_INF_TABLE", "R_INF_TABLE_M",
    # estimand
    "TAU_GRID", "TAU_FINE", "S_tau", "print_S_tau_table", "plot_S_tau_curve", "plot_S_tau_comparison",
    "export_S_tau_curve", "load_S_tau_curve",
    # simulation / calibration
    "simulate_corpus", "true_S_tau", "check", "calibration_summary", "print_calibration",
    # non-parametric / external estimators
    "freq_counts", "chao2", "chao_bunge", "nonparam_estimators", "print_nonparam",
    "load_lcmcr", "implied_tau", "implied_tau_table", "plot_S_tau_estimators",
    # posterior predictive
    "STATS", "STAT_LABELS", "STAT_GROUPS", "corpus_stats", "pbayes", "ppc",
    "print_ppc_table", "plot_ppc_hists", "plot_ppc_pvalues", "load_lognormal_ppc",
    # latent effects
    "gini", "effort_shares", "print_effort_table", "plot_effort_bars",
    "print_citability_summary", "plot_shrinkage", "plot_cell_coefficients",
    "pln_latent", "print_group_richness", "plot_group_richness",
]


# =============================================================================
# 0. Visual house style
#
# A single place to fix the look of every figure in both notebooks: a ggplot theme, large
# fonts for single-panel figures (label 24 / ticks 20 / legend 22 / bold title 24), and a
# qualitative palette with a KDE overlay for posterior and predictive histograms. The style
# follows the companion unseen-species study of Guidi, Moins & Camps (CHR 2025), so that
# figures from the two papers can be read side by side; it is implemented on top of plain
# matplotlib, without a seaborn dependency.
# =============================================================================

LABEL_FS = 24
TICK_FS = 20
LEGEND_FS = 22
TITLE_FS = 24
PANEL_TITLE_FS = 13     # multi-panel PPC grids (5-8 panels): per-panel title
PANEL_LABEL_FS = 12     # multi-panel PPC grids: per-panel x/y label
PANEL_TICK_FS = 11      # multi-panel PPC grids: per-panel ticks

MULTI_TITLE_FS = 18     # 2-3-panel figures (shrinkage, cell coefficients...): per-panel title
MULTI_LABEL_FS = 16     # 2-3-panel figures: per-panel x/y label
MULTI_TICK_FS = 13      # 2-3-panel figures: per-panel ticks
MULTI_LEGEND_FS = 13    # 2-3-panel figures: per-panel legend


def _husl_like(n, lightness=0.60, saturation=0.65):
    """A qualitative palette of n evenly-spaced hues: plain HLS at fixed lightness and
       saturation, which approximates a perceptually uniform categorical palette closely
       enough for these figures without pulling in an extra dependency."""
    return [colorsys.hls_to_rgb(i / n, lightness, saturation) for i in range(n)]


def setup_style():
    """Applies the visual house style: ggplot background and the font sizes fixed above for
       axis labels, tick labels, legends and titles. Call this once, right after the imports,
       before producing any figure."""
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": TITLE_FS,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "lines.linewidth": 2.5,
    })


def _style_axis(ax, xlabel=None, ylabel=None, title=None, legend=False, legend_loc="best",
                spines_off=("top", "right"), grid=True,
                label_fs=LABEL_FS, tick_fs=TICK_FS, title_fs=TITLE_FS, legend_fs=LEGEND_FS):
    """The single-panel styling recipe: white background, a light dashed grid below the data,
       hidden top/right spines, and the house font sizes. Used by every axis of the two
       notebooks -- pass the smaller MULTI_* constants for a 2-3-panel figure, or PANEL_* for a
       dense PPC-style grid (both defined above), instead of the default single-figure sizes."""
    ax.set_facecolor("white")
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8, color="gray")
        ax.set_axisbelow(True)
    for sp in spines_off:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    for line in ax.get_lines():
        if line.get_linewidth() <= 1.5:
            line.set_linewidth(2.5)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fs)
    if title is not None:
        ax.set_title(title, fontsize=title_fs, fontweight="bold")
    if legend:
        ax.legend(fontsize=legend_fs, loc=legend_loc)
    return ax


# =============================================================================
# 1. The corpus: loading, chronology, cell-covariate design
# =============================================================================

# The source-type labels below are the raw values of the `Author_fragment_type` field of the
# spreadsheet (in French, as encoded by the philologists who built the corpus); TYPES_EN gives
# the English labels used in every printout and figure of the notebooks.
TYPE_KEYS = ["grammairien", "littéraire", "commentateur"]   # data values (complete partition)
TYPES_EN = ["grammarian", "literary", "commentator"]        # display labels


def sum_to_zero_basis(J):
    """Orthonormal (J x J-1) basis of the sum-to-zero subspace: x = s*(H @ v), v ~ N(0,1)^{J-1}
       enforces sum(x) = 0 without distorting the distribution (constant Jacobian). Used for eta
       (J = number of sources) and for H3 = sum_to_zero_basis(3), the symmetric contrast between
       the three source categories."""
    H = np.zeros((J, J - 1))
    for k in range(J - 1):
        H[:k + 1, k] = 1.0
        H[k + 1, k] = -(k + 1.0)
        H[:, k] /= np.sqrt((k + 1.0) * (k + 2.0))
    return H


H3 = sum_to_zero_basis(3)   # (3,2): the three source categories form a complete partition of the
                            # sources, so only the DIFFERENCES between categories are identifiable
                            # (their common level is absorbed by the row effect xi_i). H3 encodes
                            # that sum-to-zero contrast with 2 degrees of freedom, symmetric in the
                            # three categories.

# --- conversion of century labels into numeric values (negative = BCE) ---
ROMAN = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8,
         'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15}


def cent(s):
    """Parses the century labels of the spreadsheet, which are written in French:
       'IIe s. av. JC' -> -2.0 ; 'IVe s. ap. JC' -> 4.0 ; a range such as 'IV-IIIe' -> midpoint.

       The century token must be isolated BEFORE the era marker ('av.' = BCE, 'ap.' = CE): the
       'v' of 'av.' is itself a valid Roman numeral, so scanning the whole string would read
       'Ie s. av. JC' as the range [I, V] and return -3 instead of -1."""
    if not isinstance(s, str):
        return np.nan
    low = s.lower()
    ap = 'ap' in low
    head = re.split(r'a[vp]', low, maxsplit=1)[0]          # text preceding the era marker
    m = re.findall(r'(xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|iv|v|iii|ii|i)', head)
    if not m:
        return np.nan
    v = np.mean([ROMAN[x.upper()] for x in m])
    return v if ap else -v


def cell_design(sc, wc, cat):
    """Two cell covariates (missing centuries imputed by the corpus mean):
         D[i,j]    = log(1 + max(0, century(j) - century(i)))  chronological distance.
                     (0 for a pair in which the source precedes the work; no such pair occurs in
                     this corpus, so no explicit structural zero is imposed.)
         Tc[i,j,:] = H3[cat[j]] * (century(i) - mean)  source category (symmetric two-dimensional
                     sum-to-zero contrast, see H3) x century of the work. Multiplied by the
                     coefficient vector u_type (see fit_pln), it yields one coefficient per
                     category, c_type = H3 @ u_type, summing to zero -- only the differences
                     between categories are identifiable, not their common level.
       Also returns wc_ctr (centred work centuries) and wc_bar (the mean), reused for the
       unobserved population."""
    wcf = np.where(np.isnan(wc), np.nanmean(wc), wc)
    scf = np.where(np.isnan(sc), np.nanmean(sc), sc)
    wc_bar = wcf.mean()
    wc_ctr = wcf - wc_bar
    D = np.log1p(np.clip(scf[None, :] - wcf[:, None], 0, None))
    Tc = wc_ctr[:, None, None] * H3[cat][None, :, :]
    return D, Tc, scf, wcf, wc_bar


@dataclass
class Corpus:
    """The observed citation matrix together with everything derived from it.

    Attributes
    ----------
    N       : (Sobs, J) float array of citation counts.
    rows    : work labels ("Author | Work"), cols: citing-source labels.
    wc, sc  : numeric century of each work / source (NaN where missing).
    cat     : source category, coded 0/1/2 in TYPE_KEYS.
    wcf,scf : the same centuries with missing values imputed by the corpus mean.
    wc_bar  : mean work century (the centring used by the category covariate).
    D, Tc   : the two cell covariates (see :func:`cell_design`).
    """
    N: np.ndarray
    rows: list
    cols: list
    wc: np.ndarray
    sc: np.ndarray
    cat: np.ndarray
    wcf: np.ndarray = field(repr=False, default=None)
    scf: np.ndarray = field(repr=False, default=None)
    wc_bar: float = 0.0
    D: np.ndarray = field(repr=False, default=None)
    Tc: np.ndarray = field(repr=False, default=None)

    @property
    def Sobs(self):
        return self.N.shape[0]

    @property
    def J(self):
        return self.N.shape[1]

    @property
    def f(self):
        """Row totals: number of citations per work."""
        return self.N.sum(1)

    def describe(self, col_scenario=None):
        """Prints the one-paragraph description of the matrix used at the top of both notebooks."""
        f = self.f
        tag = "" if col_scenario is None else f"[col_scenario={col_scenario}] "
        print(f"{tag}Matrix: {self.Sobs} observed works x {self.J} sources | "
              f"{int(self.N.sum())} citations | singletons f1={int((f == 1).sum())} | "
              f"doubletons f2={int((f == 2).sum())} | largest row={int(f.max())}")

    def describe_covariates(self):
        """Prints the chronology / cell-covariate diagnostic (Poisson-lognormal notebook)."""
        print("Sources per category: "
              + ", ".join(f"{t}={int((self.cat == k).sum())}" for k, t in enumerate(TYPES_EN))
              + f" (total {self.J})")
        print(f"Work centuries: {np.nanmin(self.wc):.2f} to {np.nanmax(self.wc):.2f}, "
              f"mean {self.wc_bar:.2f} "
              f"({int(np.isnan(self.wc).sum())} works with a missing century, imputed by the mean)")
        early = int((self.scf[None, :] < self.wcf[:, None]).sum())
        print(f"Chronological distance D: {self.D.min():.2f} to {self.D.max():.2f} | "
              f"source-earlier-than-work pairs (D=0 imposed): {early}/{self.Sobs * self.J}")


def load_corpus(path, col_scenario=2, row_keys=("Author", "Work"),
                col_key="Author_fragment", drop_dubia=False, binarize=False):
    """Builds N[i,j] (citation counts) together with the work/source attributes used by the cell
       covariates:
         wc[i]   numeric century of composition of work i
         sc[j]   numeric century of activity of source j (mean if several)
         cat[j]  category of source j, coded 0/1/2 in TYPE_KEYS (field Author_fragment_type),
                 by majority vote over its fragments (fallback: the corpus-wide majority category
                 -- Author_fragment_type has only 4 missing values out of 2080).
       A missing Author (anonymous or collective works, e.g. Carmen Saliare) is filled with
       "Anonymous" instead of being dropped, so that these works are not lost from the matrix.

       col_scenario (default 2, the convention used throughout the paper and both notebooks)
       controls the handling of missing Author_fragment (anonymous citing source,
       e.g. scholia or glossaries with no identified author -- 93 rows out of 2080):
         1 : all of them pooled into a single "Anonymous" column            (73 columns)
         2 : composite key Author_fragment | Work_fragment for ALL sources,
             "Anonymous" as fallback for a missing Author_fragment. This also splits
             known authors into one column per citing work                 (167 columns)
         3 : Author_fragment as is; if missing, fallback = Work_fragment
             (each distinct anonymous source = a separate "detector")       (107 columns)

       row_keys=("Author",) to reason at the author level; binarize=True gives incidence
       (cited / not cited) instead of counts; drop_dubia=True removes the fragments flagged
       as dubious.

       This single loader is used by BOTH notebooks, which therefore analyse exactly the same
       matrix N by construction."""
    df = pd.read_excel(path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    if drop_dubia:
        df = df[df["dubia"].isna()]
    df["Author"] = df["Author"].fillna("Anonymous")

    if col_scenario == 1:
        df[col_key] = df[col_key].fillna("Anonymous")
    elif col_scenario == 2:
        af = df[col_key].fillna("Anonymous").astype(str)
        wf = df["Work_fragment"]
        df[col_key] = np.where(wf.isna(), af, af + " | " + wf.astype(str))
    elif col_scenario == 3:
        df[col_key] = df[col_key].fillna(df["Work_fragment"])
    else:
        raise ValueError("col_scenario must be 1, 2 or 3")

    df = df.dropna(subset=list(row_keys) + [col_key])
    df["__row__"] = df[list(row_keys)].astype(str).agg(" | ".join, axis=1)
    df["__col__"] = df[col_key].astype(str)
    df["wc"] = df["Century"].map(cent)
    df["sc"] = df["Transmission_century"].map(cent)

    tab = pd.crosstab(df["__row__"], df["__col__"])
    rows, cols = tab.index.to_list(), tab.columns.to_list()
    N = tab.values.astype(float)
    if binarize:
        N = (N > 0).astype(float)

    wc = np.asarray(df.groupby("__row__")["wc"].first().reindex(rows).values, float)
    sc = np.asarray(df.groupby("__col__")["sc"].mean().reindex(cols).values, float)
    mode_type = df.groupby("__col__")["Author_fragment_type"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
    fallback = df["Author_fragment_type"].mode().iat[0]
    type_j = mode_type.reindex(cols).fillna(fallback)
    cat = type_j.map({t: k for k, t in enumerate(TYPE_KEYS)}).values.astype(int)

    D, Tc, scf, wcf, wc_bar = cell_design(sc, wc, cat)
    return Corpus(N=N, rows=rows, cols=cols, wc=wc, sc=sc, cat=cat,
                  wcf=wcf, scf=scf, wc_bar=wc_bar, D=D, Tc=Tc)


def independence_deviance(N, verbose=True):
    """Preliminary diagnostic: how far does the INDEPENDENCE model (purely separable,
       lambda_ij = exp(xi_i + eta_j)) depart from the data? Its MLE reproduces the margins exactly:
       lambda_ij = R_i C_j / T. We compare its Poisson deviance to its degrees of freedom,
       (Sobs-1)(J-1). Beware: on a table this sparse the deviance is NOT chi-square distributed,
       so the ratio settles nothing (see the discussion in the notebook)."""
    Sobs, J = N.shape
    Ri, Cj, T = N.sum(1), N.sum(0), N.sum()
    lam_ind = np.outer(Ri, Cj) / T
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(N > 0, N * np.log(N / lam_ind), 0.0)
    deviance = 2 * np.sum(term - (N - lam_ind))
    dfree = (Sobs - 1) * (J - 1)
    if verbose:
        print(f"Poisson deviance (independence model) : {deviance:.0f}")
        print(f"Degrees of freedom ({Sobs}-1)({J}-1)       : {dfree}")
        print(f"Deviance / df ratio                    : {deviance / dfree:.2f}")
        print(f"Empty cells                            : {int((N == 0).sum())}/{Sobs * J} "
              f"({100 * (N == 0).mean():.0f}%)  |  non-empty cells: {int((N > 0).sum())}")
    return deviance, dfree


# =============================================================================
# 2. Poisson-lognormal model (marginalised, sampled by NUTS)
# =============================================================================

# Gauss-Hermite nodes/weights for the smooth expectation E_{N(0,1)}[f] = sum _GW f(_GX),
# used inside the model to marginalise xi out of pi0.
_GX, _GW_H = hermegauss(24)
_GW_H = _GW_H / np.sqrt(2 * np.pi)

# A fine equispaced grid is used instead for the S(tau) integral, whose integrand contains an
# indicator function and is therefore not smooth enough for Gauss-Hermite quadrature.
_ZG = np.linspace(-8, 8, 400)
_WG = np.exp(-_ZG ** 2 / 2) / np.sqrt(2 * np.pi)
_WG /= _WG.sum()


def setup_numpyro(device_count=4):
    """Must be called before any jax computation (it sets the number of host devices used to run
       the NUTS chains in parallel)."""
    import numpyro
    numpyro.set_host_device_count(device_count)


def build_strata(wcf, scf, cat, wc_bar, cell):
    """Groups the works by CENTURY (the only work-level attribute in the model) and prepares, for
       an UNOBSERVED work of each century, the source-side quantities (distance, source category x
       century interaction). cell=False: a single stratum and no cell covariate.
       g_of[i] = stratum (century) index of observed work i -- required by S_tau(by_group=True) to
       count the ALREADY OBSERVED part of each stratum in addition to its unobserved part."""
    if cell:
        key = np.round(wcf, 3)
        uniq = sorted(set(key.tolist()))
        g_of = np.array([uniq.index(k) for k in key.tolist()])
        n_g = np.array([(g_of == g).sum() for g in range(len(uniq))], float)
        wc_g = np.array(uniq)
        Dg = np.log1p(np.clip(scf[None, :] - wc_g[:, None], 0, None))
        Tg = (wc_g[:, None] - wc_bar)[:, :, None] * H3[cat][None, :, :]
    else:
        g_of = np.zeros(len(wcf), dtype=int)
        n_g = np.array([len(wcf)], float)
        wc_g = np.array([0.0])
        Dg = np.zeros((1, len(scf)))
        Tg = np.zeros((1, len(scf), H3.shape[1]))
    return n_g, wc_g, Dg, Tg, g_of


def fit_pln(Nv, wcf, scf, cat, wc_bar, cell=True, seed=0, warmup=1500, samples=1500,
            chains=4, a0=1.0, b0=1e-3, target_accept=0.95, sx_prior=3.0):
    """Fits the marginalised Poisson-lognormal model by NUTS. cell=True activates the cell
       covariates (chronological distance + source category x century); cell=False gives the
       separable reference model. The source category (grammarian / literary / commentator, a
       complete partition of the sources) is encoded by a sum-to-zero contrast with two degrees of
       freedom, u_type ~ N(0,1)^2 (basis H3): only the DIFFERENCES between categories are
       identifiable, their common level being absorbed by the row effect xi_i.
       A posteriori, c_type = H3 @ u_type returns one coefficient per category, summing to zero and
       symmetrically interpretable ('this category over- or under-cites recent works relative to
       the average of the three'). Returns (samples, samples_by_chain, n_divergences, aux)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    Sobs, Jl = Nv.shape
    Dm = np.log1p(np.clip(scf[None, :] - wcf[:, None], 0, None)) if cell else np.zeros_like(Nv)
    Tm = ((wcf[:, None] - wc_bar)[:, :, None] * H3[cat][None, :, :] if cell
          else np.zeros(Nv.shape + (H3.shape[1],)))
    n_g, wc_g, Dg, Tg, g_of = build_strata(wcf, scf, cat, wc_bar, cell)
    a0g = a0 / len(n_g)
    H = sum_to_zero_basis(Jl)
    dat = {k: jnp.asarray(x) for k, x in
           dict(Nv=Nv, Dm=Dm, Tm=Tm, Dg=Dg, Tg=Tg, n_g=n_g, GX=_GX, GW=_GW_H, H=H).items()}

    def model():
        mu_x = numpyro.sample("mu_x", dist.Normal(-4.0, 5.0))
        s_x = numpyro.sample("s_x", dist.HalfNormal(sx_prior))
        s_y = numpyro.sample("s_y", dist.HalfNormal(3.0))
        c_dist = numpyro.sample("c_dist", dist.Normal(0., 1.)) if cell else 0.0
        u_type = (numpyro.sample("u_type", dist.Normal(0., 1.).expand([H3.shape[1]])) if cell
                  else jnp.zeros(H3.shape[1]))
        numpyro.deterministic("c_type", jnp.asarray(H3) @ u_type)
        z = numpyro.sample("z", dist.Normal(0, 1).expand([Sobs]))
        v = numpyro.sample("v", dist.Normal(0, 1).expand([Jl - 1]))
        xi = numpyro.deterministic("xi", mu_x + s_x * z)
        eta = numpyro.deterministic("eta", s_y * (dat["H"] @ v))
        type_eff = jnp.einsum("ijk,k->ij", dat["Tm"], u_type)
        theta = xi[:, None] + eta[None, :] + c_dist * dat["Dm"] + type_eff
        numpyro.factor("obs", jnp.sum(dat["Nv"] * theta - jnp.exp(theta)))
        type_eff_g = jnp.einsum("gjk,k->gj", dat["Tg"], u_type)
        Yb = jnp.sum(jnp.exp(eta[None, :] + c_dist * dat["Dg"] + type_eff_g), axis=1)
        E = jnp.exp(mu_x + s_x * dat["GX"])[None, :] * Yb[:, None]
        pi0 = jnp.clip(jnp.sum(dat["GW"][None, :] * jnp.exp(-E), axis=1), 1e-12, 1 - 1e-12)
        numpyro.deterministic("pi0", pi0)
        numpyro.deterministic("Yb", Yb)
        numpyro.factor("pop", jnp.sum(-(a0g + dat["n_g"]) * jnp.log(b0 + 1 - pi0)))

    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept),
                num_warmup=warmup, num_samples=samples, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    s = mcmc.get_samples()
    sc_ = mcmc.get_samples(group_by_chain=True)
    ndiv = int(mcmc.get_extra_fields()["diverging"].sum())
    aux = dict(family="pln", Dm=Dm, Tm=Tm, Dg=Dg, Tg=Tg, n_g=n_g, wc_g=wc_g, g_of=g_of,
               cell=cell, a0g=a0g, b0=b0)
    return {k: np.asarray(v) for k, v in s.items()}, sc_, ndiv, aux


def rhat_ess(sc_by_chain, keys, max_nb_points=500):
    """Maximum R-hat-infinity (see :func:`rhat_inf`, section 3) and minimum bulk ESS over a list
       of scalar parameters. Vector-valued entries are scanned component by component, the
       maximum being taken over all components. The threshold to compare the returned
       R-hat-infinity with is given by :func:`rhat_inf_threshold`."""
    from numpyro.diagnostics import summary as nsummary
    st = nsummary({k: sc_by_chain[k] for k in keys}, group_by_chain=True)
    rmax = 0.0
    for k in keys:
        a = np.asarray(sc_by_chain[k])                       # (n_chains, n_draws, ...)
        a = a.reshape(a.shape[0], a.shape[1], -1)
        rmax = max([rmax] + [rhat_inf(a[:, :, j], max_nb_points) for j in range(a.shape[2])])
    return rmax, min(np.nanmin(st[k]["n_eff"]) for k in keys)


# =============================================================================
# 3. Gamma-Poisson model (Gibbs sampler with data augmentation)
# =============================================================================

def slice_shape(logf, x0, rng, w=1.0, m=50):
    """One-dimensional slice sampler, in log space, for a positive shape parameter (log-concave
       target). It uses the supplied generator `rng` (and not the global numpy.random state) so
       that `seed=` makes every execution fully reproducible -- including for alpha and gamma,
       which are drawn by this sampler and directly determine S(tau) and the calibration and
       sensitivity diagnostics of the notebook."""
    psi0 = np.log(x0)
    y = logf(psi0) - rng.exponential()
    u = rng.random()
    L, R = psi0 - w * u, psi0 + w * (1 - u)
    j = int(m * rng.random())
    k = m - 1 - j
    while j > 0 and logf(L) > y:
        L -= w
        j -= 1
    while k > 0 and logf(R) > y:
        R += w
        k -= 1
    for _ in range(200):
        psi = L + rng.random() * (R - L)
        if logf(psi) > y:
            return np.exp(psi)
        if psi < psi0:
            L = psi
        else:
            R = psi
    return np.exp(psi)


def gamma_and_log(rng, shape, rate, size):
    """Draws Gamma(shape, rate) together with its log, remaining stable even for a tiny shape
       (boosting X = Gamma(a+1)·U^{1/a}: avoids the underflow of log(X) when a -> 0)."""
    if size == 0:
        return np.zeros(0), np.zeros(0)
    if shape >= 1:
        g = rng.gamma(shape, 1 / rate, size=size)
        return g, np.log(g)
    gb = rng.gamma(shape + 1, 1 / rate, size=size)
    logg = np.log(gb) + np.log(rng.random(size)) / shape
    return np.exp(logg), logg


def gibbs(N, n_iter=200000, burn=100000, thin=10, seed=0,
          a_alpha=1., b_alpha=1., a_gamma=1., b_gamma=1.,   # shape priors  ~ Gamma
          a_beta=1., b_beta=1., a_delta=1., b_delta=1.,     # rate priors   ~ Gamma
          a0=1e-3, b0=1e-3,                                 # prior on Lambda ~ Gamma (near-flat)
          px_step=0.5, store_latent=False):
    """Gibbs sampler for the Gamma-Poisson model. The default chain length (200,000 iterations,
       half of them discarded as burn-in, thinned by 10) is dictated by the mixing of alpha and
       Lambda on this corpus: at 20,000 iterations the diagnostic for Lambda and pi0 stays far
       above any reasonable threshold, whereas these settings bring every R-hat-infinity (see
       rhat_inf) below the 5% threshold, for a run time of about 30 s per chain.

       store_latent=True additionally stores, at each retained iteration, the complete vector of
       source efforts Y_j rather than only their sum Y_bullet. This is what the interpretation of
       the latent effects and the posterior predictive check need; it is off by default because
       it multiplies the memory footprint of a long chain by J.

       Returns (post, aux) where post is a dictionary of posterior draws and aux carries the shape
       of the problem and the family tag read by S_tau() and ppc()."""
    rng = np.random.default_rng(seed)
    Sobs, J = N.shape
    Ni, Nj = N.sum(1), N.sum(0)
    alpha = beta = gamma = delta = 1.0
    X = rng.gamma(1, 1, Sobs)
    Y = rng.gamma(1, 1, J)
    Lam = float(Sobs)
    keep = {k: [] for k in ["alpha", "beta", "gamma", "delta", "Lam", "S0", "pi0", "S0_rb"]}
    kept_X, kept_Xu, kept_Yp, kept_Y = [], [], [], []

    for it in range(n_iter):
        Yp = Y.sum()
        # (a) observed row effects
        X = rng.gamma(alpha + Ni, 1 / (beta + Yp))
        # (b) augmentation: undetected units
        pi0 = (beta / (beta + Yp)) ** alpha
        S0 = int(rng.poisson(Lam * pi0))
        Xu, logXu = gamma_and_log(rng, alpha, beta + Yp, S0)
        sumXu, sumlogXu = Xu.sum(), logXu.sum()
        S = Sobs + S0
        Xp = X.sum() + sumXu
        # (c) column effects (all S units contribute through exp(-X_i Y_j))
        Y = rng.gamma(gamma + Nj, 1 / (delta + Xp))
        Yp = Y.sum()
        # (d) rates (conjugate)
        beta = rng.gamma(a_beta + S * alpha, 1 / (b_beta + Xp))
        delta = rng.gamma(a_delta + J * gamma, 1 / (b_delta + Yp))
        # (e) shapes (log-concave) by slice sampling
        tX = np.log(X).sum() + sumlogXu
        alpha = slice_shape(lambda p: (a_alpha - 1) * p - b_alpha * np.exp(p)
                            + np.exp(p) * (S * np.log(beta) + tX) - S * gammaln(np.exp(p)) + p,
                            alpha, rng)
        tY = np.log(Y).sum()
        gamma = slice_shape(lambda p: (a_gamma - 1) * p - b_gamma * np.exp(p)
                            + np.exp(p) * (J * np.log(delta) + tY) - J * gammaln(np.exp(p)) + p,
                            gamma, rng)
        # (f) intensity of the super-population
        Lam = rng.gamma(a0 + S, 1 / (b0 + 1))
        # (g) expansion move: X_i->cX_i, Y_j->Y_j/c, beta->beta/c, delta->c*delta
        #     the acceptance ratio reduces to the ratio of the priors on (beta,delta) alone
        c = np.exp(rng.normal(0, px_step))
        la = ((a_beta - 1) * np.log(beta / c) - b_beta * beta / c) \
            - ((a_beta - 1) * np.log(beta) - b_beta * beta) \
            + ((a_delta - 1) * np.log(c * delta) - b_delta * c * delta) \
            - ((a_delta - 1) * np.log(delta) - b_delta * delta)
        if np.log(rng.random()) < la:
            X *= c
            Y /= c
            beta /= c
            delta *= c
            if S0 > 0:
                Xu *= c       # scale consistency: Xu must follow X, otherwise X_i*Yp and
                              # Xu*Yp would no longer be on the same scale after this move
        # Rao-Blackwellised S0 (lower variance) for the posterior of S
        Yp = Y.sum()
        pi0 = (beta / (beta + Yp)) ** alpha
        p_nb = (b0 + 1 - pi0) / (b0 + 1)
        S0_rb = int(rng.negative_binomial(a0 + Sobs, min(max(p_nb, 1e-12), 1.0)))

        if it >= burn and (it - burn) % thin == 0:
            for kk, v in [("alpha", alpha), ("beta", beta), ("gamma", gamma), ("delta", delta),
                          ("Lam", Lam), ("S0", S0), ("pi0", pi0), ("S0_rb", S0_rb)]:
                keep[kk].append(v)
            kept_X.append(X.copy())
            kept_Xu.append(Xu.copy())
            kept_Yp.append(Yp)
            if store_latent:
                kept_Y.append(Y.copy())

    out = {k: np.array(v) for k, v in keep.items()}
    out["X"] = np.array(kept_X)
    out["Xu"] = kept_Xu                      # list: its length varies from draw to draw
    out["Yp"] = np.array(kept_Yp)
    if store_latent:
        out["Y"] = np.array(kept_Y)
    aux = dict(family="gp", Sobs=Sobs, J=J, a0=a0, b0=b0)
    return out, aux


# -----------------------------------------------------------------------------
# Convergence diagnostic: R-hat-infinity
#
# Moins, Arbel, Dutfoy & Girard (2025), "On the use of a local R-hat to improve
# MCMC convergence diagnostic", Bayesian Analysis 20(1), 131-156. The reference
# implementation is vendored in rhat_infinity.py and used as is.
#
# The traditional Gelman-Rubin R-hat compares a between-chain to a within-chain
# VARIANCE, and is therefore blind to any disagreement between chains that does
# not show up in the first two moments -- in particular to chains that explore
# different tails while sharing a mean and a variance, which is exactly the
# failure mode to fear here (heavy-tailed posteriors for Lambda, S and pi0).
# R-hat-infinity replaces that variance comparison by
#
#     R-hat-infinity = sup_x R-hat(x),   R-hat(x) = R-hat applied to 1{theta <= x},
#
# i.e. it compares the whole empirical cumulative distribution functions of the
# chains, quantile by quantile, and returns the worst quantile. It equals 1 only
# if the chains agree everywhere, and the quantile attaining the supremum tells
# WHERE they disagree. The supremum is estimated on a grid of at most
# `max_nb_points` draws taken from the chains themselves.
# -----------------------------------------------------------------------------

# Table 2 of the paper: empirical quantiles R_inf,lim of the R-hat-infinity
# distribution under the null hypothesis that all chains follow the same
# distribution (target ESS 400, 2000 replications). Rows: number of chains m --
# to be DOUBLED when the split version is used, as it is here; columns: level
# alpha. The paper documents (its Figure 2) that the threshold is linear in m at
# fixed alpha, which is what the interpolation below exploits.
R_INF_TABLE_M = np.array([2, 3, 4, 8, 10, 20])
R_INF_TABLE = {
    0.005: [1.018, 1.023, 1.027, 1.038, 1.043, 1.080],
    0.01:  [1.016, 1.022, 1.025, 1.037, 1.041, 1.076],
    0.05:  [1.012, 1.016, 1.020, 1.031, 1.036, 1.062],
    0.1:   [1.010, 1.014, 1.018, 1.028, 1.033, 1.056],
}


def rhat_inf_threshold(m, alpha=0.05, split=True):
    """Threshold for R-hat-infinity with `m` chains at level `alpha`, interpolated in Table 2
       of Moins et al. (2025). With `split=True` the effective number of chains is 2m, so that
       the usual four split chains give 1.031 at the 5% level (the 1.03 recommended in the
       paper) rather than the 1.01 rule of thumb of the traditional R-hat."""
    if split:
        m *= 2
    return float(np.interp(m, R_INF_TABLE_M, R_INF_TABLE[alpha]))


def rhat_inf(mat, max_nb_points=500):
    """R-hat-infinity of one scalar parameter. mat: (n_chains, n_draws), as for the traditional
       R-hat; the reference implementation expects the transpose. Indicator variables degenerate
       at the two ends of the grid (zero within-chain variance), which produces nan values that
       rhat_infinity() already discards -- errstate only silences the warning."""
    from rhat_infinity import rhat_infinity
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(rhat_infinity(np.asarray(mat, float).T, max_nb_points=max_nb_points))


def merge_chains(chains):
    """Concatenates the draws of several `gibbs` runs. `Xu` is a list whose length varies from
       draw to draw and is therefore concatenated as a list, `X` (and `Y`, if stored) along the
       draw axis, everything else as a flat vector."""
    post = {}
    for k in chains[0]:
        if k == "Xu":
            post[k] = sum([c[k] for c in chains], [])
        elif k in ("X", "Y"):
            post[k] = np.concatenate([c[k] for c in chains], axis=0)
        else:
            post[k] = np.concatenate([c[k] for c in chains])
    return post


def print_rhat_table(post, chains, keys, alpha=0.05):
    """Median, 95% CI and R-hat-infinity for each named scalar parameter, with the threshold
       associated with the number of (split) chains at level `alpha`."""
    thr = rhat_inf_threshold(len(chains), alpha)
    for k in keys:
        lo, md, hi = np.quantile(post[k], [.025, .5, .975])
        rh = rhat_inf(np.array([c[k] for c in chains]))
        flag = "" if rh < thr else "   <-- ABOVE THRESHOLD"
        print(f"{k:6s}  median {md:9.3f}   95% CI [{lo:8.3f}, {hi:9.3f}]   "
              f"R-hat-inf {rh:.3f}{flag}")
    print(f"\nR-hat-infinity (Moins et al. 2025), split version on {len(chains)} chains: "
          f"threshold at the {alpha:.0%} level = {thr:.3f}")


# =============================================================================
# 4. The shared estimand: threshold richness S(tau)
# =============================================================================

TAU_GRID = np.linspace(0.03, 0.9, 40)   # canonical x-grid for the S(tau) curve, shared by both
                                        # notebooks so that the cross-model comparison figure
                                        # (plot_S_tau_comparison) overlays two curves computed
                                        # on exactly the same abscissa.

TAU_FINE = np.geomspace(0.002, 0.9, 70)  # log-spaced grid used only to INVERT the curve
                                         # (implied_tau): the classical estimators can land far
                                         # below tau=0.03, especially under the Gamma-Poisson
                                         # model whose curve is the flatter of the two.


def S_tau(post, aux, taus, seed=0, by_group=False):
    """Threshold richness: the number of works (observed and unobserved) whose detection
       probability 1-exp(-mu_i) exceeds tau, for each posterior draw. Unlike S, this is a bounded
       functional of the measure that the data actually identify (see the discussion in the
       notebooks). Returns a dictionary {tau: array of draws}.

       The reconstruction depends on the family, read from aux["family"]:

       * "gp"  (Gamma-Poisson): the sampler augments the unobserved works explicitly, so the count
               is a direct comparison of mu = X * Y_bullet against the threshold, over the
               observed X and the imputed Xu of that same draw.
       * "pln" (Poisson-lognormal): S_0 and its citabilities are marginalised out, so the
               unobserved part is reconstructed per century from Lambda_g|data ~
               Gamma(a0g+n_g, b0+1-pi0_g), integrating over a fine grid in xi the intensity of the
               works that are undetected but have p_det >= tau. The observed part uses the exact
               mu_i, cell covariates included; it returns (out, pdet).

       by_group=True (Poisson-lognormal only): does NOT sum the unobserved part over strata -- it
       keeps S_g(tau) per century g (observed part counted through g_of, plus the unobserved part
       of stratum g). Unlike 1-pi0(g), which is not identified (see discussion), S_g(tau) is
       calibrated exactly as S(tau) is (same argument, applied stratum by stratum): it is the
       legitimate chronological decomposition of threshold richness. It then returns
       (out, pdet, out_g, obs_g), where out_g[t] = S_g(tau) and obs_g[t] = number of OBSERVED works
       with p_det >= tau in each stratum (the numerator of the threshold capture fraction
       obs_g/S_g); both have shape (draws, G)."""
    taus = list(taus)
    family = aux.get("family", "pln")

    if family == "gp":
        if by_group:
            raise ValueError("by_group is only defined for the Poisson-lognormal model "
                             "(the Gamma-Poisson model has no century strata)")
        n = len(post["Yp"])
        out = {t: np.empty(n) for t in taus}
        for d in range(n):
            mu_obs = post["X"][d] * post["Yp"][d]
            mu0 = post["Xu"][d] * post["Yp"][d]
            for t in taus:
                mu_thr = -np.log(1 - t)
                out[t][d] = (mu_obs >= mu_thr).sum() + (mu0 >= mu_thr).sum()
        return out

    if family != "pln":
        raise ValueError(f"unknown family {family!r}")

    rng = np.random.default_rng(seed)
    mu_x, s_x = post["mu_x"], post["s_x"]
    xi, eta = post["xi"], post["eta"]
    pi0, Yb = post["pi0"], post["Yb"]
    cd = post["c_dist"] if aux["cell"] else np.zeros(len(mu_x))
    cu = post["u_type"] if aux["cell"] else np.zeros((len(mu_x), aux["Tm"].shape[-1]))
    Dm, Tm = aux["Dm"], aux["Tm"]
    n_g, a0g, b0, g_of = aux["n_g"], aux["a0g"], aux["b0"], aux["g_of"]
    G = len(n_g)
    nd = len(mu_x)
    out = {t: np.empty(nd) for t in taus}
    pdet = np.empty((nd, xi.shape[1]))
    out_g = {t: np.empty((nd, G)) for t in taus} if by_group else None
    obs_gd = {t: np.empty((nd, G)) for t in taus} if by_group else None
    for d in range(nd):                                   # d indexes the posterior draw
        theta = xi[d][:, None] + eta[d][None, :] + cd[d] * Dm + Tm @ cu[d]
        mu_i = np.sum(np.exp(theta), axis=1)
        p_i = 1 - np.exp(-mu_i)
        pdet[d] = p_i
        Lam_g = rng.gamma(a0g + n_g, 1.0 / (b0 + 1 - pi0[d]))
        Eg = np.exp(mu_x[d] + s_x[d] * _ZG)[None, :] * Yb[d][:, None]     # (G, 400)
        pdet_g, nondet_g = 1 - np.exp(-Eg), np.exp(-Eg)
        for t in taus:
            obs_g = np.array([np.sum(p_i[g_of == g] >= t) for g in range(G)], float)
            unseen_g = Lam_g * np.sum(_WG[None, :] * (pdet_g >= t) * nondet_g, axis=1)
            Sg = obs_g + rng.poisson(np.clip(unseen_g, 0.0, None))
            out[t][d] = Sg.sum()
            if by_group:
                out_g[t][d] = Sg
                obs_gd[t][d] = obs_g
    return (out, pdet, out_g, obs_gd) if by_group else (out, pdet)


def print_S_tau_table(draws, taus, Sobs):
    """The S(tau) table with the model-free hard bound S_obs/tau, common to both notebooks."""
    print(f"{'tau':>6s}{'mu_tau':>9s}{'S(tau) median':>15s}{'95% CI':>18s}{'hard bound S_obs/tau':>22s}")
    for t in taus:
        lo, md, hi = np.quantile(draws[t], [.025, .5, .975])
        print(f"{t:6.2f}{-np.log(1 - t):9.3f}{md:15.1f}   [{lo:5.1f},{hi:6.1f}]{Sobs / t:22.0f}")


def _S_tau_quantiles(tau_grid, draws):
    tau_grid = np.asarray(list(tau_grid), float)
    med = np.array([np.median(draws[t]) for t in tau_grid])
    lo = np.array([np.quantile(draws[t], .025) for t in tau_grid])
    hi = np.array([np.quantile(draws[t], .975) for t in tau_grid])
    return tau_grid, med, lo, hi


def plot_S_tau_curve(tau_grid, draws, Sobs, ylim=None, title="Threshold richness S(τ)", ax=None):
    """S(tau) against tau, with its 95% band, the hard bound S_obs/tau and the observed count."""
    tau_grid, med, lo, hi = _S_tau_quantiles(tau_grid, draws)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    ax.plot(tau_grid, med, color="navy", label="median S(τ)")
    ax.fill_between(tau_grid, lo, hi, color="navy", alpha=.15, label="95% CI")
    ax.plot(tau_grid, Sobs / tau_grid, "r--", label="hard bound S_obs/τ")
    ax.axhline(Sobs, color="k", ls=":", label=f"S_obs={Sobs}")
    if ylim is not None:
        ax.set_ylim(*((0, ylim) if np.isscalar(ylim) else ylim))
    _style_axis(ax, xlabel="τ (detection-probability threshold)", ylabel="S(τ)",
               title=title, legend=True)
    return ax


def export_S_tau_curve(path, tau_grid, draws):
    """Saves the (tau_grid, median, 95% CI) summary of an S(tau) curve to a small .npz file,
       so that a companion notebook can overlay it on its own curve without re-running the
       sampler -- the same reproducibility convention already used for the cross-tabulated
       PPC (see ppc_lognormal.npz / load_lognormal_ppc)."""
    tau_grid, med, lo, hi = _S_tau_quantiles(tau_grid, draws)
    np.savez(path, tau=tau_grid, med=med, lo=lo, hi=hi)


def load_S_tau_curve(path, fallback=None):
    """Loads an S(tau) curve summary exported by :func:`export_S_tau_curve`. Falls back on a
       dictionary of published (tau, med, lo, hi) arrays when the file is absent, which then
       serves only as a reference point. Returns (tau_grid, med, lo, hi, provenance)."""
    if os.path.exists(path):
        d = np.load(path)
        return d["tau"], d["med"], d["lo"], d["hi"], f"recomputed from {path}"
    if fallback is not None:
        return (np.asarray(fallback["tau"]), np.asarray(fallback["med"]),
                np.asarray(fallback["lo"]), np.asarray(fallback["hi"]),
                "published reference values (file absent)")
    raise FileNotFoundError(path)


def plot_S_tau_comparison(tau_grid_a, draws_a, tau_grid_b, draws_b, Sobs,
                          label_a="Poisson-lognormal", label_b="Gamma-Poisson",
                          color_a="navy", color_b="firebrick", ylim=None,
                          title="Threshold richness S(τ): lognormal vs Gamma-Poisson", ax=None):
    """Overlays the S(tau) curves of the two parametric families on the SAME figure -- the
       comparison that matters most in this work, since S(tau) at moderate tau is where the
       two families agree despite disagreeing by a large factor on S itself. Either curve may
       be passed as raw posterior draws {tau: array} (via tau_grid_*, draws_*) or, if a
       notebook does not have the other model's posterior available, via the (tau, med, lo,
       hi) summary returned by :func:`load_S_tau_curve` -- pass draws_*=None and a 4-tuple as
       tau_grid_* in that case."""
    def _curve(tau_grid, draws):
        if draws is None:
            tau, med, lo, hi = tau_grid
            return np.asarray(tau, float), med, lo, hi
        return _S_tau_quantiles(tau_grid, draws)

    ta, ma, la, ha = _curve(tau_grid_a, draws_a)
    tb, mb, lb, hb = _curve(tau_grid_b, draws_b)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    ax.plot(ta, ma, color=color_a, label=f"median S(τ) — {label_a}")
    ax.fill_between(ta, la, ha, color=color_a, alpha=.15)
    ax.plot(tb, mb, color=color_b, label=f"median S(τ) — {label_b}")
    ax.fill_between(tb, lb, hb, color=color_b, alpha=.15)
    tau_grid = ta if len(ta) >= len(tb) else tb
    ax.plot(tau_grid, Sobs / tau_grid, "k--", label="hard bound S_obs/τ")
    ax.axhline(Sobs, color="k", ls=":", label=f"S_obs={Sobs}")
    if ylim is not None:
        ax.set_ylim(*((0, ylim) if np.isscalar(ylim) else ylim))
    _style_axis(ax, xlabel="τ (detection-probability threshold)", ylabel="S(τ)",
               title=title, legend=True)
    return ax


# =============================================================================
# 5. Simulation of a complete corpus, recovery and calibration
# =============================================================================

def simulate_corpus(family, J, Lam, seed=0, **params):
    """Generates a complete corpus under the model, then truncates it to the detected rows
       (a work is observed iff it is cited at least once). Returns a dictionary with the observed
       matrix `N`, the ground truth `truth` (including `mu_all`, the detection intensities BEFORE
       truncation, needed to evaluate the true value of S(tau)) and, for the Poisson-lognormal
       family, the work/source attributes required to refit the model.

       family="gp"  : params = alpha, beta (citability Gamma), gamma, delta (effort Gamma).
       family="pln" : params = mu_x, s_x, s_y and, optionally, the cell-covariate coefficients
                      c_dist and c_type = (c_gram, c_lit, c_comm) -- the TRUE coefficients per
                      category (they need not sum to zero on input; only their centred part, the
                      difference between categories, can be recovered from the fitted model)."""
    rng = np.random.default_rng(seed)
    S = rng.poisson(Lam)

    if family == "gp":
        alpha, beta = params["alpha"], params["beta"]
        gamma, delta = params["gamma"], params["delta"]
        X = rng.gamma(alpha, 1 / beta, size=S)
        Y = rng.gamma(gamma, 1 / delta, size=J)
        N = rng.poisson(np.outer(X, Y))                   # S x J matrix
        detected = N.sum(1) >= 1                          # observed iff cited at least once
        truth = dict(S=int(S), Sobs=int(detected.sum()), S0=int((~detected).sum()),
                     alpha=alpha, gamma=gamma, betadelta=beta * delta,
                     pi0=(beta / (beta + Y.sum())) ** alpha)
        truth["mu_all"] = X * Y.sum()      # true detection intensities (before truncation)
        return dict(N=N[detected], truth=truth)

    if family != "pln":
        raise ValueError(f"unknown family {family!r}")

    mu_x, s_x, s_y = params["mu_x"], params["s_x"], params["s_y"]
    c_dist = params.get("c_dist", 0.0)
    c_type = np.asarray(params.get("c_type", (0., 0., 0.)), float)
    wc_s = rng.choice([-3.0, -2.5, -2.0], size=S)                 # work centuries (archaic)
    sc_s = rng.choice([-1.0, 1.0, 2.0, 4.0, 6.0], size=J)         # source centuries
    cat_s = rng.integers(0, 3, size=J)                            # source category, uniform
    wc_bar_s = wc_s.mean()
    xi = mu_x + rng.normal(0, s_x, S)
    eta = rng.normal(0, s_y, J)
    eta -= eta.mean()
    Dm = np.log1p(np.clip(sc_s[None, :] - wc_s[:, None], 0, None))
    Tm = (wc_s[:, None] - wc_bar_s) * c_type[cat_s][None, :]
    lam = np.exp(xi[:, None] + eta[None, :] + c_dist * Dm + Tm)
    Nfull = rng.poisson(lam)
    det = Nfull.sum(1) >= 1
    truth = dict(S=int(S), Sobs=int(det.sum()), mu_x=mu_x, s_x=s_x, s_y=s_y,
                 c_dist=c_dist, c_type=c_type, mu_all=lam.sum(1))
    return dict(N=Nfull[det].astype(float), wc=wc_s[det], sc=sc_s, cat=cat_s,
                wc_bar=wc_bar_s, truth=truth)


def true_S_tau(truth, tau):
    """The true value of S(tau) in a simulated corpus: the number of works, detected or not, whose
       detection probability exceeds tau (computed from mu_all, the intensities before
       truncation)."""
    return int((1 - np.exp(-truth["mu_all"]) >= tau).sum())


def check(name, samp, tv, width=22):
    """Prints one line of a recovery table: truth, posterior median, 95% CI and a pass flag."""
    lo, md, hi = np.quantile(samp, [.025, .5, .975])
    flag = "OK" if lo <= tv <= hi else ">>> OUTSIDE CI <<<"
    print(f"  {name:{width}s} truth {tv:8.3f} | median {md:8.3f}  95%CI[{lo:7.3f},{hi:8.3f}]  {flag}")


def calibration_summary(records, error="bias"):
    """Turns the replicates of a calibration study into coverage / error / width.

       records : dict {estimand key -> list of (draws, true value)}, one entry per replicate.
       error   : "bias" -> signed median relative bias (median(draws)-truth)/truth
                 "abs"  -> median relative absolute error |median(draws)-truth|/truth

       Returns {key: dict(coverage, error, width)} with the three quantities as fractions. The
       width is the 95% interval width as a fraction of the truth: coverage obtained through an
       enormous interval is not an estimate, which is why it is always reported alongside."""
    summary = {}
    for key, reps in records.items():
        cov, err, wid = 0, [], []
        for draws, truth in reps:
            lo, hi = np.quantile(draws, [.025, .975])
            cov += (lo <= truth <= hi)
            denom = max(abs(truth), 1)
            d = (np.median(draws) - truth) / denom
            err.append(abs(d) if error == "abs" else d)
            wid.append((hi - lo) / denom)
        summary[key] = dict(coverage=cov / len(reps),
                            error=float(np.median(err)), width=float(np.median(wid)))
    return summary


def print_calibration(summary, labels=None, error_label="median bias"):
    """Prints the calibration table (coverage, error, interval width as a % of the truth)."""
    labels = labels or {k: k for k in summary}
    print(f"{'estimand':12s}{'95% CI cov.':>13s}{error_label:>19s}{'95% CI width (% of truth)':>27s}")
    for k, s in summary.items():
        sign = "+" if error_label.endswith("bias") else ""
        print(f"{labels.get(k, k):12s}{100 * s['coverage']:11.0f}%"
              f"{100 * s['error']:{sign}18.1f}%{100 * s['width']:26.0f}%")


# =============================================================================
# 6. Classical richness estimators, and what threshold they implicitly assume
# =============================================================================
#
# Which estimators belong here, and why. The sampling design of this corpus is a
# multiple-list (capture-recapture) design: the J citing sources are the capture occasions, a
# work is "captured" by source j if source j cites it. The natural data are therefore the
# INCIDENCE frequency counts Q_k = number of works cited by exactly k distinct sources.
#
# Abundance-based estimators (Chao1 and its relatives) instead treat the 2080 individual
# citations as the sample, i.e. as independent draws from a multinomial over works. That
# assumption is plainly false here: once a grammarian has excerpted a work he quotes it dozens
# of times (Nonius quotes Lucilius 1122 times), so citations arrive in bursts and f1, f2 are
# counts of clustered, not independent, events. Chao1 is therefore not reported; its place is
# taken by Chao2 -- the same estimator recast on the incidence counts, where the independent
# replicates really are the sources.
#
# The one abundance-based estimator kept is Chao & Bunge (2002), because it is the estimator
# derived under exactly the gamma-mixed Poisson law that the companion notebook fits: it is not
# a generic competitor but the non-parametric counterpart of the Gamma-Poisson model itself.
# It is reported on both spectra (abundance and incidence) with the cut-off diagnostic the
# paper prescribes.

def freq_counts(counts):
    """Frequency counts f_k (k = 1, 2, ...) of a vector of per-work counts, zeros dropped."""
    counts = np.asarray(counts)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.zeros(0, int)
    return np.array([int((counts == k).sum()) for k in range(1, int(counts.max()) + 1)])


def chao2(Q1, Q2, Sobs, m, bias_corrected=False):
    """Chao2, the incidence-based lower bound on richness (Chao 1987), with its log-normal 95 %
       confidence interval. m = number of sampling units (here: citing sources), Q1/Q2 = number
       of works seen in exactly one / exactly two of them. This is the estimator that Chao1
       becomes once the sampling unit is taken to be the SOURCE rather than the individual
       citation -- the only reading of the design that is defensible here.

       bias_corrected=True returns the small-sample form Sobs + A*Q1*(Q1-1)/(2*(Q2+1)), which is
       the one to quote when Q2 is small. Returns (estimate, lo, hi)."""
    A = (m - 1) / m
    if bias_corrected or Q2 == 0:
        est = Sobs + A * Q1 * (Q1 - 1) / (2 * (Q2 + 1))
        if Q2 == 0:
            return est, est, est
    else:
        est = Sobs + A * Q1 ** 2 / (2 * Q2)
    r = Q1 / Q2
    var = Q2 * (A / 2 * r ** 2 + A ** 2 * r ** 3 + A ** 2 / 4 * r ** 4)
    D = est - Sobs
    if D <= 0:
        return est, est, est
    C = np.exp(1.96 * np.sqrt(np.log(1 + var / D ** 2)))
    return est, Sobs + D / C, Sobs + D * C


def chao_bunge(counts, cutoff=10):
    """Chao & Bunge (2002, Biometrics 58, 531-539), the duplication-fraction estimator derived
       under a gamma-mixed Poisson abundance model -- i.e. under exactly the law that the
       Gamma-Poisson notebook fits, which is why it is the one abundance-based estimator worth
       keeping here.

       Writing f_k for the frequency counts and n = sum k*f_k, their equation (2) is

           N* = sum_{k>=2} f_k / theta_hat,   theta_hat = 1 - f1 * sum k^2 f_k / n^2,

       where theta is the expected duplicated fraction. The estimator is undefined when
       theta_hat <= 0, which happens as soon as a few enormous counts dominate sum k^2 f_k --
       the paper says so explicitly ("a negative value of theta implies that the overlap measure
       is not high enough to produce a reliable estimate"), and it is exactly what this corpus
       does. Their remedy (equation 7) is to restrict the estimating equation to the RARE
       classes, counts <= cutoff, and to add back the abundant works as observed:

           N = D_a + sum_{k=2}^{cutoff} f_k / theta_hat(cutoff),

       with cutoff = 10 the value fixed in advance by Chao, Ma & Yang (1993) and used throughout
       the 2002 paper. Pass cutoff=None for the uncut estimator (2).

       The variance follows the paper's asymptotic multinomial argument and the interval is the
       log transform of Chao (1987), so the lower bound never falls below the observed count.
       Returns (estimate, lo, hi, theta_hat, se)."""
    counts = np.asarray(counts)
    counts = counts[counts > 0]
    D = counts.size
    if cutoff is None:
        Da, rare = 0, counts
    else:
        rare = counts[counts <= cutoff]
        Da = D - rare.size
    f = freq_counts(rare).astype(float)
    if f.size == 0 or f[0] == 0:
        return float(D), float(D), float(D), 1.0, 0.0
    k = np.arange(1, f.size + 1, dtype=float)
    n, Q = (k * f).sum(), (k ** 2 * f).sum()
    theta = 1.0 - f[0] * Q / n ** 2
    if theta <= 0:
        return np.nan, np.nan, np.nan, theta, np.nan
    A = f[1:].sum()
    est = Da + A / theta
    # delta method on the frequency counts, treated as multinomial with total est
    is1 = (k == 1).astype(float)
    dtheta = -(is1 * Q / n ** 2 + f[0] * k ** 2 / n ** 2 - 2 * k * f[0] * Q / n ** 3)
    dN = (1.0 - is1) / theta - A / theta ** 2 * dtheta
    cov = np.diag(f) - np.outer(f, f) / est
    var = float(dN @ cov @ dN)
    d = est - D
    if var <= 0 or d <= 0:
        return est, est, est, theta, np.sqrt(max(var, 0.0))
    C = np.exp(1.96 * np.sqrt(np.log(1 + var / d ** 2)))
    return est, D + d / C, D + d * C, theta, np.sqrt(var)


def nonparam_estimators(N, cutoff=10):
    """The classical richness estimators applicable to this design, all on the SAME matrix N.

       Incidence-based (sources = capture occasions; the defensible reading of the design):
         Chao2, Chao2-bc  -- lower bounds on richness under arbitrary heterogeneity;
         Jack1, Jack2     -- the first- and second-order incidence jackknives. They survive the
                             objection that removes Chao1 (they read only Q1, Q2 and m), but note
                             that Burnham & Overton's (m-1)/m correction assumes the m occasions
                             are exchangeable: at m=167 that factor is 0.994, so they degenerate
                             to Sobs+Q1 and Sobs+2*Q1-Q2 with the design dropping out entirely,
                             while the sources are in fact wildly unequal (Gini of effort 0.83).
                             Reported as the field's standard reference, not as a third estimate.
       Abundance-based, kept only because it is the estimator matched to the Gamma-Poisson
       family (see :func:`chao_bunge`):
         Chao-Bunge (abundance) and, for comparison, Chao-Bunge on the incidence spectrum.

       Chao1 is deliberately absent: it reads the individual citations as independent draws,
       whereas citations arrive in bursts within a source (see the section comment above).
       Returns (estimates, frequencies) where each estimate is a (value, lo, hi) triple."""
    Sobs, m = N.shape
    ab = N.sum(1)                                  # citations per work   -> abundance spectrum
    inc = (N > 0).sum(1)                           # sources per work     -> incidence spectrum
    f1, f2 = int((ab == 1).sum()), int((ab == 2).sum())
    Q1, Q2 = int((inc == 1).sum()), int((inc == 2).sum())
    out = {"Chao2": chao2(Q1, Q2, Sobs, m),
           "Chao2-bc": chao2(Q1, Q2, Sobs, m, bias_corrected=True)}
    j1 = Sobs + Q1 * (m - 1) / m
    j2 = Sobs + Q1 * (2 * m - 3) / m - Q2 * (m - 2) ** 2 / (m * (m - 1))
    out["Jack1"], out["Jack2"] = (j1, j1, j1), (j2, j2, j2)
    cb_a = chao_bunge(ab, cutoff=cutoff)
    cb_i = chao_bunge(inc, cutoff=cutoff)
    out["Chao-Bunge (ab.)"] = cb_a[:3]
    out["Chao-Bunge (inc.)"] = cb_i[:3]
    fr = dict(f1=f1, f2=f2, Q1=Q1, Q2=Q2, m=m, Sobs=Sobs, cutoff=cutoff,
              theta_ab=cb_a[3], theta_inc=cb_i[3],
              theta_ab_uncut=chao_bunge(ab, cutoff=None)[3],
              theta_inc_uncut=chao_bunge(inc, cutoff=None)[3])
    return out, fr


def print_nonparam(est, fr, N=None, cutoffs=(4, 6, 8, 10, 15, 20, 25)):
    """Prints the estimator table and, when the matrix N is passed, the Chao-Bunge cut-off
       stability diagnostic that the 2002 paper asks for: the estimate is to be trusted only
       over a range of cut-offs on which it is stable."""
    print(f"S_obs={fr['Sobs']}  m={fr['m']} sources | abundance f1={fr['f1']} f2={fr['f2']} | "
          f"incidence Q1={fr['Q1']} Q2={fr['Q2']}")
    print(f"\n{'estimator':22s}{'estimate':>10s}{'95% CI':>20s}")
    for k, (v, lo, hi) in est.items():
        ci = "--" if not np.isfinite(lo) or abs(hi - lo) < 1e-9 else f"[{lo:.0f}, {hi:.0f}]"
        val = "undefined" if not np.isfinite(v) else f"{v:.0f}"
        print(f"{k:22s}{val:>10s}{ci:>20s}")
    print(f"\nChao-Bunge duplicated fraction theta, uncut (estimator 2 of the paper): abundance "
          f"{fr['theta_ab_uncut']:+.2f}, incidence {fr['theta_inc_uncut']:+.2f}"
          "\n  -- theta <= 0 means the estimator is undefined and the cut-off form (7) is needed.")
    if N is None:
        return
    ab, inc = N.sum(1), (N > 0).sum(1)
    print(f"\n{'cut-off':>8s}{'N (abundance)':>16s}{'theta':>8s}{'N (incidence)':>16s}{'theta':>8s}")
    for c in cutoffs:
        va, _, _, ta, _ = chao_bunge(ab, cutoff=c)
        vi, _, _, ti, _ = chao_bunge(inc, cutoff=c)
        fa = "undefined" if not np.isfinite(va) else f"{va:.0f}"
        fi = "undefined" if not np.isfinite(vi) else f"{vi:.0f}"
        print(f"{c:8d}{fa:>16s}{ta:8.2f}{fi:>16s}{ti:8.2f}")


def load_lcmcr(path="lcmcr_results.csv", paradigm=None):
    """Reads the LCMCR posterior summaries produced by lcmcr_check.py / lcmcr_check.R (fits of
       Manrique-Vallier's 2016 latent-class capture-recapture model on the current corpus under
       three definitions of a capture list). Returns a DataFrame, or None when the file is
       absent -- the notebooks then fall back on the published values, exactly as they do for
       the cross-model PPC."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if paradigm is None else df[df["paradigm"] == paradigm]


# --- the bridge: what threshold does a point estimate of S implicitly assume? ----------------

def implied_tau(value, tau_grid, draws):
    """Inverts the S(tau) curve at a given richness value: returns the threshold tau at which
       the model's threshold richness equals `value`.

       This is the device that makes the classical estimators commensurable with S(tau).
       Each of them returns a single number for the non-identifiable S; but every such number
       IS the model's answer to "how many works had a detection probability of at least tau?"
       for exactly one tau. Reporting that tau turns an unfalsifiable claim about the total
       into a statement one can argue with: an estimator that lands at tau = 0.3 is counting
       only works that had at least a one-in-three chance of being cited, whatever its author
       believed it was counting.

       The inversion is done draw by draw (S(tau) is decreasing in tau within each posterior
       draw), so the answer comes with its own uncertainty. Returns (median, lo, hi); a bound
       is NaN when `value` falls outside the range spanned by the curve on tau_grid."""
    tau_grid = np.asarray(list(tau_grid), float)
    order = np.argsort(tau_grid)
    tg = tau_grid[order]
    S = np.column_stack([np.asarray(draws[t], float) for t in tau_grid[order]])   # (ndraw, ntau)
    out = np.full(S.shape[0], np.nan)
    for d in range(S.shape[0]):
        y = np.maximum.accumulate(S[d][::-1])[::-1]        # enforce monotonicity (Poisson noise)
        if value > y[0] or value < y[-1]:
            continue
        out[d] = np.interp(-value, -y, np.log(tg))          # y decreasing -> negate to sort
    out = np.exp(out)
    if np.all(np.isnan(out)):
        return np.nan, np.nan, np.nan
    return tuple(np.nanquantile(out, [.5, .025, .975]))


def implied_tau_table(est, tau_grid, draws, Sobs=None, extra=None):
    """The table of classical estimates alongside the threshold each of them implicitly assumes.
       `extra` is an optional {label: (value, lo, hi)} dictionary for external estimates (e.g.
       LCMCR). Returns the list of (label, value, tau, tau_lo, tau_hi) rows."""
    rows = []
    items = list(est.items()) + list((extra or {}).items())
    if Sobs is not None:
        items.append(("S_obs (no extrapolation)", (float(Sobs), float(Sobs), float(Sobs))))
    print(f"{'estimator':26s}{'S estimate':>12s}{'implied tau':>14s}{'95% CI on tau':>20s}")
    for label, triple in items:
        v = triple[0] if isinstance(triple, (tuple, list, np.ndarray)) else float(triple)
        if not np.isfinite(v):
            print(f"{label:26s}{'undefined':>12s}{'--':>14s}{'--':>20s}")
            continue
        t, lo, hi = implied_tau(v, tau_grid, draws)
        ts = "off grid" if not np.isfinite(t) else f"{t:.3f}"
        cis = "--" if not np.isfinite(lo) else f"[{lo:.3f}, {hi:.3f}]"
        print(f"{label:26s}{v:12.0f}{ts:>14s}{cis:>20s}")
        rows.append((label, v, t, lo, hi))
    return rows


def plot_S_tau_estimators(tau_grid, draws, Sobs, rows, ylim=None, ax=None, logx=True,
                          title="Classical estimators read on the S(τ) curve"):
    """The S(tau) curve with each classical estimate placed on it at its implied threshold.
       `rows` is the output of :func:`implied_tau_table`. Each estimator becomes a point on the
       curve: its height is the number of works it claims, its abscissa the detection threshold
       that number silently corresponds to."""
    tau_grid, med, lo, hi = _S_tau_quantiles(tau_grid, draws)
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(tau_grid, med, color="navy", lw=2, label="median S(τ)")
    ax.fill_between(tau_grid, lo, hi, color="navy", alpha=.15, label="95% CI")
    ax.axhline(Sobs, color="k", ls=":", lw=1, label=f"S_obs={Sobs}")
    cols = _husl_like(max(len(rows), 1))
    placed = []
    for (label, v, t, tlo, thi), c in zip(rows, cols):
        if not np.isfinite(t):
            continue
        placed.append(t)
        ax.plot([tau_grid.min(), t], [v, v], color=c, ls="--", lw=1, alpha=.7)
        ax.plot([t, t], [0, v], color=c, ls="--", lw=1, alpha=.7)
        if np.isfinite(tlo):
            ax.plot([tlo, thi], [v, v], color=c, lw=4, alpha=.35, solid_capstyle="butt")
        ax.plot([t], [v], "o", color=c, ms=10, zorder=5, mec="white", mew=1.2,
                label=f"{label} = {v:.0f}   (τ ≈ {t:.3g})")   # .3g: the implied thresholds span
                                                              # two decades, .2f would print 0.00
    if ylim is not None:
        ax.set_ylim(*((0, ylim) if np.isscalar(ylim) else ylim))
    # focus the axis on the region the estimators actually occupy, with a little air on each side
    left = min(placed) / 1.8 if placed else tau_grid.min()
    left = max(tau_grid.min(), left)
    ax.set_xlim(left, tau_grid.max())
    if logx:
        # a log abscissa is unavoidable (the implied thresholds span two decades), but the
        # default decade ticks are unreadable here: fix a legible set of round thresholds
        ax.set_xscale("log")
        cand = np.array([.001, .002, .005, .01, .02, .05, .1, .2, .5, .9])
        cand = cand[(cand >= left) & (cand <= tau_grid.max())]
        span = np.log10(tau_grid.max() / left)
        keep, last = [], -np.inf                    # thin them out so the labels never collide,
        for t in cand:                              # whatever number of decades the axis spans
            if np.log10(t) - last >= span / 7:
                keep.append(t)
                last = np.log10(t)
        ax.set_xticks(keep)
        ax.set_xticklabels([f"{t:g}" for t in keep])
        ax.minorticks_off()
    _style_axis(ax, xlabel="τ (detection-probability threshold)", ylabel="S(τ)", title=title,
                legend=False)
    # the legend carries the whole reading of the figure (value + implied threshold), so it goes
    # OUTSIDE the axes rather than covering the curve it is meant to explain
    ax.legend(fontsize=LEGEND_FS - 2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, handletextpad=0.6, borderaxespad=0)
    plt.tight_layout()
    return ax


# =============================================================================
# 7. Posterior predictive checking (the same eight statistics for both models)
# =============================================================================

STATS = ['Sobs', 'f1', 'f2', 'tot', 'maxc', 'Q1', 'Q2', 'maxsrc']

STAT_LABELS = {'Sobs': 'texts seen', 'f1': 'singletons (cited 1×)', 'f2': 'cited 2×',
               'tot': 'total citations', 'maxc': 'max citations (one work)',
               'Q1': 'seen by 1 source', 'Q2': 'seen by 2 sources',
               'maxsrc': 'max sources (one work)'}

STAT_GROUPS = [("FREQUENCIES (citations/work)", ['Sobs', 'f1', 'f2', 'tot', 'maxc']),
               ("INCIDENCES (sources/work)", ['Q1', 'Q2', 'maxsrc'])]


def corpus_stats(M):
    """Summary statistics of a corpus (works x sources matrix), seen rows only. Two families:
       FREQUENCIES (number of citations per work: Sobs, f1, f2, total, max) and INCIDENCES
       (number of distinct sources per work: Q1, Q2, max) -- the latter being the rarity
       information that Chao2 and the incidence jackknives exploit."""
    M = M[M.sum(1) >= 1]
    ni = M.sum(1)
    ic = (M > 0).sum(1)
    return dict(Sobs=int(M.shape[0]), f1=int((ni == 1).sum()), f2=int((ni == 2).sum()),
                tot=int(ni.sum()), maxc=int(ni.max() if ni.size else 0),
                Q1=int((ic == 1).sum()), Q2=int((ic == 2).sum()),
                maxsrc=int(ic.max() if ic.size else 0))


def pbayes(a, o):
    """Two-sided Bayesian p-value: 2*min(P(rep>=obs), P(rep<=obs))."""
    return min(1.0, 2 * min(np.mean(a >= o), np.mean(a <= o)))


def ppc(post, aux, N, ndraw=400, seed=0):
    """Posterior predictive check: for many posterior draws, re-simulate a COMPLETE corpus under
       the model, truncate it to the detected rows and compare its summary statistics with those
       of the observed corpus. Returns (obs, reps) where reps[k] is the array of replicated values
       of statistic k over the draws.

       family="pln": per century g, Lambda_g|data ~ Gamma(a0g+n_g, b0+1-pi0_g), S_g ~ Poisson,
         citabilities xi ~ N(mu_x, s_x), citations thinned by distance and source category.
       family="gp" : S ~ Poisson(Lam), a NEW population of citabilities X ~ Gamma(alpha, beta),
         but the SAME inferred transmitters Y_j (requires gibbs(store_latent=True)).

       The two families sub-sample the posterior draws differently (rng.choice with replacement
       for the lognormal, rng.integers for the Gamma); both are unbiased sub-samples of the same
       posterior, and each is fixed by `seed` so that its table reproduces exactly."""
    rng = np.random.default_rng(seed)
    obs = corpus_stats(N)
    reps = {k: [] for k in STATS}
    family = aux.get("family", "pln")
    ndraws_post = len(post["mu_x"]) if family == "pln" else len(post["alpha"])

    if family == "pln":
        Dg, Tg, n_g = aux["Dg"], aux["Tg"], aux["n_g"]
        a0g, b0 = aux["a0g"], aux["b0"]
        cd = post["c_dist"] if aux["cell"] else np.zeros(ndraws_post)
        cu = post["u_type"] if aux["cell"] else np.zeros((ndraws_post, Tg.shape[-1]))
        for t in rng.choice(ndraws_post, size=ndraw):
            eta, mu_x, s_x = post["eta"][t], post["mu_x"][t], post["s_x"][t]
            rows_ = []
            for g in range(len(n_g)):
                Lam_g = rng.gamma(a0g + n_g[g], 1.0 / (b0 + 1 - post["pi0"][t][g]))
                Sg = rng.poisson(Lam_g)
                if Sg == 0:
                    continue
                xi = mu_x + rng.normal(0, s_x, size=Sg)
                lam = np.exp(xi[:, None] + eta[None, :] + cd[t] * Dg[g][None, :]
                             + (Tg[g] @ cu[t])[None, :])
                rows_.append(rng.poisson(lam))
            M = np.vstack(rows_) if rows_ else np.zeros((0, N.shape[1]))
            s = corpus_stats(M)
            for k in STATS:
                reps[k].append(s[k])
    elif family == "gp":
        if "Y" not in post:
            raise ValueError("the Gamma-Poisson PPC needs the individual source efforts Y_j: "
                             "run gibbs(..., store_latent=True)")
        for t in rng.integers(0, ndraws_post, ndraw):
            Ssim = rng.poisson(post["Lam"][t])
            if Ssim == 0:
                continue
            Xr = rng.gamma(post["alpha"][t], 1 / post["beta"][t], size=Ssim)  # new population
            M = rng.poisson(np.outer(Xr, post["Y"][t]))                       # same transmitters
            s = corpus_stats(M)
            for k in STATS:
                reps[k].append(s[k])
    else:
        raise ValueError(f"unknown family {family!r}")

    return obs, {k: np.array(v) for k, v in reps.items()}


def print_ppc_table(obs, reps, keys=None, grouped=True):
    """Observed value, predictive median, 95% predictive interval and two-sided Bayesian p-value
       for each statistic, with a MISFIT flag below 0.05."""
    header = f"{'stat':8s}{'observed':>10s}{'pred. med.':>12s}{'95% pred. interval':>20s}{'p Bayes':>9s}"
    print(header)
    groups = STAT_GROUPS if grouped else [(None, keys or STATS)]
    for title, gkeys in groups:
        gkeys = [k for k in gkeys if keys is None or k in keys]
        if not gkeys:
            continue
        if title:
            short = "frequencies (citations per work)" if title.startswith("FREQ") \
                else "incidences (distinct sources per work)"
            print(f"-- {short} " + "-" * max(0, len(header) - len(short) - 4))
        for k in gkeys:
            a = reps[k]
            lo, md, hi = np.quantile(a, [.025, .5, .975])
            pB = pbayes(a, obs[k])
            flag = "  <-- MISFIT" if pB < 0.05 else ""
            print(f"{k:8s}{obs[k]:10d}{md:12.0f}   [{lo:6.0f},{hi:7.0f}]{pB:9.2f}{flag}")


def _hist_with_kde(ax, data, color, bins, log=False):
    """Histogram with a KDE overlay rescaled to the histogram's counts (rather than to a
       density), so that the curve tracks the bars it annotates.

       On a log-scaled axis the bins are uniform in log10(x), not in x (np.logspace bin edges):
       the histogram is really a histogram OF log10(data) with a constant bin width. The KDE is
       therefore fit and scaled entirely in that log10 space -- fitting it on the raw data and
       converting back to a linear-x density via the change-of-variables Jacobian would rescale
       each bin by a different, wrong factor (bins have wildly different linear widths on a log
       axis), producing a curve that does not track the histogram it is meant to overlay."""
    counts, edges, _ = ax.hist(data, bins=bins, color=color, alpha=0.85,
                               edgecolor="white", linewidth=0.6)
    if len(np.unique(data)) > 1:
        if log:
            x = np.log10(data)
            bin_width = np.log10(edges[1]) - np.log10(edges[0])   # constant: log-uniform bins
        else:
            x = data
            bin_width = edges[1] - edges[0]
        kde = gaussian_kde(x)
        xs = np.linspace(x.min(), x.max(), 300)
        ys = kde(xs) * len(data) * bin_width
        xs_plot = 10 ** xs if log else xs
        ax.plot(xs_plot, ys, color=color, linewidth=2.2)
    return counts, edges


def plot_ppc_hists(obs, reps, keys, log_keys=("maxc",), ncols=2, figsize=None):
    """Posterior-predictive histograms, one panel per statistic: a qualitative palette, a KDE
       overlay, a dashed green line at the predictive median and a solid red line at the observed
       value, annotated with an "Observation / Median" text box. The panels are laid out in a
       grid rather than a single row, which is not readable once the full set of eight statistics
       (frequencies AND incidences) is shown together."""
    keys = list(keys)
    nrows = -(-len(keys) // ncols)
    if figsize is None:
        figsize = (7.5 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    palette = _husl_like(len(keys))

    for i, (k, color) in enumerate(zip(keys, palette)):
        ax = axes[i]
        d = np.asarray(reps[k], float)
        logx = k in log_keys
        median_val = np.median(d)
        if logx:
            d = np.clip(d, 1, None)
            bins = np.logspace(0, np.log10(max(d.max(), obs[k]) + 1), 30)
            _hist_with_kde(ax, d, color, bins, log=True)
            ax.set_xscale("log")
        else:
            bins = min(30, max(10, len(np.unique(d))))
            _hist_with_kde(ax, d, color, bins, log=False)

        ax.axvline(median_val, color="green", linestyle="--", alpha=0.7, linewidth=2)
        ax.axvline(obs[k], color="red", linestyle="-", alpha=0.85, linewidth=2,
                  label="Observation")
        ax.text(0.95, 0.95, f"Observation: {obs[k]:.0f}\nMedian: {median_val:.1f}",
               transform=ax.transAxes, va="top", ha="right", fontsize=PANEL_LABEL_FS,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray",
                        alpha=0.9, linewidth=1.2))
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6, color="gray")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=PANEL_TICK_FS)
        ax.set_xlabel("Value", fontsize=PANEL_LABEL_FS)
        ax.set_ylabel("Count", fontsize=PANEL_LABEL_FS)
        ax.set_title(STAT_LABELS[k], fontsize=PANEL_TITLE_FS, pad=10)

    for ax in axes[len(keys):]:
        ax.remove()
    plt.tight_layout()
    return fig, axes[:len(keys)]


def load_lognormal_ppc(obs, path="ppc_lognormal.npz", fallback=None):
    """Loads the predictive draws exported by Poisson_lognormal.ipynb and recomputes, under the
       identical p-value convention, (median, p) for each statistic. Falls back on the published
       values of that notebook when the file is absent, which then serve only as a reference
       point. Returns (table, provenance string)."""
    if os.path.exists(path):
        d = np.load(path)
        ln = {k: (np.median(d[k]), pbayes(d[k], obs[k])) for k in STATS if k in d.files}
        if len(ln) == len(STATS):
            return ln, f"recomputed from {path} (identical p-value convention)"
    return dict(fallback or {}), "published values of Poisson_lognormal.ipynb (reference point)"


def plot_ppc_pvalues(pvals_a, pvals_b, label_a, label_b, keys=STATS, ax=None):
    """The Bayesian p-values of two models side by side, with the misfit region (p<0.05) shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, [pvals_a[k] for k in keys], w, label=label_a, color="indianred")
    ax.bar(x + w / 2, [pvals_b[k] for k in keys], w, label=label_b, color="steelblue")
    ax.axhspan(0, 0.05, color="grey", alpha=.15)
    ax.axhline(0.05, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([STAT_LABELS[k] for k in keys], rotation=35, ha="right",
                       fontsize=MULTI_TICK_FS)
    ax.set_ylim(0, 1.02)
    _style_axis(ax, ylabel="Bayesian p-value (two-sided)", legend=True,
               label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS, legend_fs=MULTI_LEGEND_FS)
    return ax


# =============================================================================
# 8. Latent effects: who transmits, which works are citable
# =============================================================================

def gini(x):
    """Gini coefficient of a vector of non-negative shares."""
    x = np.asarray(x, float)
    return 2 * np.sum(np.arange(1, len(x) + 1) * np.sort(x)) / (len(x) * x.sum()) \
        - (len(x) + 1) / len(x)


def effort_shares(eta_or_Y, log=False):
    """Share of transmission effort of each source, w_j = Y_j / sum_k Y_k -- scale-invariant, and
       therefore identified where Y_j alone is not. Pass the posterior draws of Y_j (Gamma-Poisson)
       or of eta_j with log=True (Poisson-lognormal, where Y_j = exp(eta_j))."""
    Y = np.exp(eta_or_Y) if log else np.asarray(eta_or_Y)
    return Y / Y.sum(1, keepdims=True)


def print_effort_table(cols, w, N=None, top=8, detail=True, gini_step=20):
    """Concentration of the transmission effort: the `top` leading sources with their credible
       intervals (detail=True), then the top-1 / top-3 shares and the Gini coefficient."""
    wm = w.mean(0)
    order = np.argsort(-wm)
    if detail:
        print("Sources by share of transmission effort (posterior):")
        for k in order[:top]:
            lo, md, hi = np.quantile(w[:, k], [.025, .5, .975])
            cit = f"  ({int(N[:, k].sum())} cit.)" if N is not None else ""
            print(f"  {cols[k][:34]:34s} {100 * md:5.1f}%  CI[{100 * lo:4.1f},{100 * hi:4.1f}]{cit}")
    g = np.median([gini(wr) for wr in w[::gini_step]])
    if detail:
        print(f"  top-1 = {100 * np.sort(wm)[-1]:.0f}%   top-3 = {100 * np.sort(wm)[-3:].sum():.0f}%   "
              f"Gini(effort) = {g:.2f}")
    else:
        print(f"Transmission: top-1 = {cols[order[0]]} ({100 * wm[order[0]]:.0f}%), "
              f"top-3 = {100 * np.sort(wm)[-3:].sum():.0f}%, Gini = {g:.2f}")
    return order, wm, g


def plot_effort_bars(ax, cols, w, order, top=8, color="teal"):
    """Horizontal bar chart of the leading sources' share of transmission effort, with their 95%
       credible intervals."""
    sel = order[:top]
    wm = w.mean(0)
    lo = 100 * (wm[sel] - np.quantile(w[:, sel], .025, 0))
    hi = 100 * (np.quantile(w[:, sel], .975, 0) - wm[sel])
    ax.barh([cols[k][:28] for k in sel][::-1], (100 * wm[sel])[::-1],
            xerr=[lo[::-1], hi[::-1]], color=color, edgecolor="white", linewidth=0.6)
    _style_axis(ax, xlabel="share of transmission effort (%)", title="Who transmits? (effort Y_j)",
               label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS, title_fs=MULTI_TITLE_FS,
               spines_off=("top", "right"))
    return ax


def print_citability_summary(rows, Ni, mu, pdet, top=4, fragile=False, pooled=False):
    """The work side: the most citable texts, the singletons and (optionally) the most fragile
       works -- those known to us today but with the lowest detection probability.
       `mu` and `pdet` are the posterior draws (n_draw, S_obs). pooled=True takes the median of
       the singletons over all draws AND all works at once (the Gamma-Poisson convention) rather
       than over the per-work posterior means (the Poisson-lognormal one)."""
    mum, pdm = mu.mean(0), pdet.mean(0)
    sing = np.where(Ni == 1)[0]
    print("Most 'citable' texts (mu = E[number of citations]):")
    for k in np.argsort(-mum)[:top]:
        extra = f"  p_det={pdm[k]:.2f}" if fragile else ""
        print(f"  {rows[k][:40]:40s} N={int(Ni[k]):4d} -> mu={mum[k]:.0f}{extra}")
    mu_s = mu[:, sing] if pooled else mum[sing]
    pd_s = pdet[:, sing] if pooled else pdm[sing]
    print(f"\nSingletons (N_i.=1, {len(sing)} texts): median mu={np.median(mu_s):.2f}, "
          f"median detection probability={np.median(pd_s):.2f}")
    if fragile:
        print("Most 'fragile' works (lowest p_det -- known to us, but they nearly disappeared):")
        for k in np.argsort(pdm)[:top]:
            print(f"  {rows[k][:40]:40s} N={int(Ni[k]):4d}  p_det={pdm[k]:.2f}")
    return sing


def plot_shrinkage(ax, Ni, mum, sing, title="Citability: shrinkage"):
    """Observed citations against posterior mu_i, log-log, with the singletons highlighted: the
       model confirms the counts where the data are abundant and shrinks them where they are not."""
    ax.scatter(Ni, mum, s=30, alpha=.7, edgecolor="white", linewidth=0.4)
    ax.plot([1, Ni.max()], [1, Ni.max()], "k--", lw=1.5)
    ax.scatter(Ni[sing], mum[sing], color="crimson", s=40, label="singletons",
              edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axis(ax, xlabel="observed $N_{i\\cdot}$", ylabel="posterior $\\mu_i$", title=title,
               legend=True, label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS,
               title_fs=MULTI_TITLE_FS, legend_fs=MULTI_LEGEND_FS)
    return ax


def pln_latent(post, aux):
    """Poisson-lognormal, work side: the effective transmission effort seen by each WORK
       (chronological distance and source category x century included), the expected number of
       citations mu_i = exp(xi_i) * Y_eff(i) and the detection probability 1-exp(-mu_i), for every
       posterior draw. Shapes (n_draw, S_obs)."""
    xi, eta = post["xi"], post["eta"]
    nd = len(xi)
    cd = post["c_dist"] if aux["cell"] else np.zeros(nd)
    cu = post["u_type"] if aux["cell"] else np.zeros((nd, aux["Tm"].shape[-1]))
    Yp_eff = np.stack([np.sum(np.exp(eta[t][None, :] + cd[t] * aux["Dm"] + aux["Tm"] @ cu[t]),
                              axis=1) for t in range(nd)])
    mu = np.exp(xi) * Yp_eff
    return mu, 1 - np.exp(-mu)


def _group_quantiles(Sg, obs_g, taus):
    """Median and 95% CI of S_g(tau) and of the threshold completeness obs_g/S_g, per stratum."""
    Sq, Fq = {}, {}
    for t in taus:
        Sq[t] = (np.median(Sg[t], 0), np.quantile(Sg[t], .025, 0), np.quantile(Sg[t], .975, 0))
        frac = obs_g[t] / np.clip(Sg[t], 1e-9, None)
        Fq[t] = (np.median(frac, 0), np.quantile(frac, .025, 0), np.quantile(frac, .975, 0))
    return Sq, Fq


def print_group_richness(wc_g, n_g, Sg, obs_g, taus):
    """Threshold richness per century S_g(tau) and threshold completeness obs_g(tau)/S_g(tau).

       1-pi0(g) is NOT identified (same argument as for S), so S_g(tau) is substituted for it: the
       century-wise decomposition of S(tau), calibrated exactly as S(tau) itself. THRESHOLD
       COMPLETENESS is obs_g(tau)/S_g(tau), where obs_g(tau) is the number of OBSERVED works of
       century g with p_det >= tau (and NOT n_g, which also counts observed works BELOW the
       threshold and which therefore do not belong to the population S_g(tau): n_g/S_g can exceed
       1 and is not interpretable). obs_g/S_g lies in [0,1]: the proportion of the works of
       century g that were "citable enough to exceed tau" and that the surviving tradition did in
       fact record. This does not display a gradient of DETECTABILITY (the Y_bullet(g) channel
       remains anaemic): it is the chronological distribution of threshold richness, a distinct
       question, and an estimable one."""
    Sq, Fq = _group_quantiles(Sg, obs_g, taus)
    print("\nThreshold richness PER CENTURY S_g(tau) and threshold completeness "
          "obs_g(tau)/S_g(tau):")
    for t in taus:
        smd, slo, shi = Sq[t]
        fmd, flo, fhi = Fq[t]
        print(f"\n  century -> S_g({t}) [95% CI]  |  obs_g/S_g({t}) [95% CI]  (n_g works seen)")
        for g in np.argsort(wc_g):
            print(f"    {wc_g[g]:6.2f} : S_g={smd[g]:5.1f} [{slo[g]:.1f},{shi[g]:.1f}]   "
                  f"completeness={fmd[g]:.2f} [{flo[g]:.2f},{fhi[g]:.2f}]   "
                  f"(n_g={int(n_g[g])})")
    return Sq, Fq


def plot_group_richness(ax_counts, ax_ratio, wc_g, n_g, Sq, Fq, taus, tau_ratio=0.5):
    """Left axis: threshold richness per century S_g(tau) with the observed n_g overlaid.
       Right axis: threshold completeness obs_g/S_g at `tau_ratio`, marker size proportional to
       n_g and each point annotated with its observed count, so that the strata too thin to be
       read can be discounted."""
    o2 = np.argsort(wc_g)
    x = wc_g[o2]
    for t, col in zip(taus, ["navy", "darkorange", "seagreen"]):
        md, lo, hi = Sq[t]
        ax_counts.plot(x, md[o2], "o-", color=col, label=f"S_g({t})", linewidth=2.5, markersize=7)
        ax_counts.fill_between(x, lo[o2], hi[o2], color=col, alpha=.15)
    ax_counts.plot(x, n_g[o2], "s--", color="gray", lw=1.5, label="n_g (observed)", markersize=6)
    ax_counts.set_ylim(bottom=0)
    _style_axis(ax_counts, xlabel="century of the work (negative = BCE)", ylabel="number of works",
               title="Threshold richness per century S_g(τ)\n(works with ≥ τ chance of being cited)",
               legend=True, label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS,
               title_fs=MULTI_TITLE_FS, legend_fs=MULTI_LEGEND_FS)

    md, lo, hi = Fq[tau_ratio]
    ax_ratio.errorbar(x, md[o2], yerr=[md[o2] - lo[o2], hi[o2] - md[o2]],
                      fmt="none", ecolor="darkgreen", capsize=3, zorder=1, linewidth=1.5)
    ax_ratio.scatter(x, md[o2], s=20 + 18 * n_g[o2], color="darkgreen", zorder=2,
                     edgecolor="white", linewidth=0.5)
    for g in o2:
        ax_ratio.annotate(f"n={int(n_g[g])}", (wc_g[g], md[g]), textcoords="offset points",
                          xytext=(6, 6), fontsize=MULTI_TICK_FS - 2, color="dimgray")
    ax_ratio.axhline(1, color="k", ls=":", lw=1.2)
    ax_ratio.set_ylim(0, 1.05)
    _style_axis(ax_ratio, xlabel="century of the work (negative = BCE)",
               ylabel=f"threshold completeness  obs_g / S_g({tau_ratio})",
               title="Threshold completeness per century\n(fraction of works ≥ τ actually seen)",
               label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS, title_fs=MULTI_TITLE_FS)
    return ax_counts, ax_ratio


def plot_cell_coefficients(post, types=None, figsize=(14, 6)):
    """Posteriors of the two cell-covariate coefficients: chronological distance (left) and the
       three category effects of the sum-to-zero contrast (right)."""
    types = types or TYPES_EN
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    _hist_with_kde(ax[0], np.asarray(post["c_dist"], float), "teal", 50)
    ax[0].axvline(0, color="k", ls="--", linewidth=1.5)
    _style_axis(ax[0], xlabel="c_dist", title="chronological distance  c_dist",
               label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS, title_fs=MULTI_TITLE_FS)
    for k, (t, col) in enumerate(zip(types, _husl_like(3))):
        _hist_with_kde(ax[1], np.asarray(post["c_type"][:, k], float), col, 50)
        ax[1].plot([], [], color=col, label=t, linewidth=2.5)   # legend proxy (histogram, not line)
    ax[1].axvline(0, color="k", ls="--", linewidth=1.5)
    _style_axis(ax[1], xlabel="c_type", title="source category × work century  c_type",
               legend=True, label_fs=MULTI_LABEL_FS, tick_fs=MULTI_TICK_FS,
               title_fs=MULTI_TITLE_FS, legend_fs=MULTI_LEGEND_FS)
    plt.tight_layout()
    return fig, ax
