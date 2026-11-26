"""Explanatory figures for the technical report (docs/report).

Usage::

    python surrogate/report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_validation import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report"


def gp_illustration():
    """A Gaussian process in one dimension: prior samples, then the posterior after five observations."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 300)
    ell, sf, sn = 0.15, 1.0, 1e-4

    def k(a, b):
        return sf ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)

    Kxx = k(x, x) + 1e-9 * np.eye(len(x))
    prior = rng.multivariate_normal(np.zeros(len(x)), Kxx, 3)
    xt = np.array([0.08, 0.3, 0.45, 0.7, 0.92])
    yt = np.sin(6 * xt) + 0.3 * xt
    K = k(xt, xt) + sn * np.eye(len(xt))
    ks = k(x, xt)
    alpha = np.linalg.solve(K, yt)
    mu = ks @ alpha
    var = sf ** 2 - np.einsum("ij,jk,ik->i", ks, np.linalg.inv(K), ks)
    sd = np.sqrt(np.maximum(var, 0))
    post = rng.multivariate_normal(mu, k(x, x) - ks @ np.linalg.solve(K, ks.T) + 1e-8 * np.eye(len(x)), 3)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    ax[0].fill_between(x, -2 * sf, 2 * sf, color="C0", alpha=0.12, label="prior: mean 0 ± 2σ")
    for p in prior:
        ax[0].plot(x, p, lw=1)
    ax[0].set_title("Prior: functions the kernel considers plausible", fontsize=10)
    ax[1].fill_between(x, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.18, label="posterior mean ± 2σ")
    ax[1].plot(x, mu, "C0", lw=2, label="posterior mean (the prediction)")
    for p in post:
        ax[1].plot(x, p, lw=0.7, alpha=0.6)
    ax[1].plot(xt, yt, "ko", label="observations (simulations)")
    ax[1].set_title("Posterior: functions that also pass through the data", fontsize=10)
    for a in ax:
        a.set_xlabel("design variable (scaled)")
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8, loc="lower left")
    ax[0].set_ylabel("output")
    fig.tight_layout()
    fig.savefig(OUT / "gp_illustration.png", dpi=150)
    plt.close(fig)


def partial_dependence():
    """Each output as a function of one design variable, the others fixed at the baseline design."""
    model = json.loads((ROOT / "tesseracts" / "vane_surrogate" / "model.json").read_text())
    lo, hi = np.array(model["lower"]), np.array(model["upper"])
    names = model["inputs"]
    base = np.array([7, 0.40, 0.10, 0.80, 0.03])
    outs = [("T_mean", "mean temperature [K]", 1), ("vm_p99_core", "stress p99 [MPa]", 1e-6),
            ("f1", "f1 [Hz]", 1), ("coolant_dp", "coolant Δp [kPa]", 1e-3)]
    fig, ax = plt.subplots(len(outs), len(names), figsize=(14, 9), sharex="col")
    for j, name in enumerate(names):
        g = np.linspace(lo[j], hi[j], 60)
        X = np.repeat(base[None], len(g), 0)
        X[:, j] = g
        u = (X - lo) / (hi - lo)
        for i, (k, label, s) in enumerate(outs):
            mu, sd = predict(model["outputs"][k], u)
            a = ax[i, j]
            a.fill_between(g, (mu - 2 * sd) * s, (mu + 2 * sd) * s, color="C0", alpha=0.2)
            a.plot(g, mu * s, "C0", lw=1.6)
            a.axvline(base[j], c="gray", lw=0.6, ls=":")
            a.grid(alpha=0.3)
            if j == 0:
                a.set_ylabel(label, fontsize=9)
            if i == len(outs) - 1:
                a.set_xlabel(name)
            a.tick_params(labelsize=7)
    fig.suptitle("Surrogate response to one design variable at a time (others at the baseline design, dotted); "
                 "band: ±2σ", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "partial_dependence.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gp_illustration()
    partial_dependence()
    print(sorted(p.name for p in OUT.glob("*.png")))
