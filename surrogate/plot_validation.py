"""Parity plots of the deployed surrogate on the held-out test designs.

The prediction is recomputed from ``model.json`` (the file shipped in the Tesseract image), not from the
training session, so the figure validates the artefact that is actually deployed.

Usage::

    python surrogate/plot_validation.py --data ../vane-thermomech/results/dataset/sweep_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHOW = [("T_mean", "mean metal temperature [K]", 1), ("vm_p99_core", "von Mises p99 [MPa]", 1e-6),
        ("f1", "first natural frequency [Hz]", 1), ("coolant_dp", "coolant pressure drop [kPa]", 1e-3)]


def predict(g, u):
    X, ls = np.array(g["X"]), np.array(g["length_scale"])
    k = g["constant"] * np.exp(-0.5 * (((u[:, None, :] - X[None]) / ls) ** 2).sum(-1))
    mean = k @ np.array(g["alpha"])
    var = np.maximum(g["constant"] + g["noise"] - np.einsum("ij,jk,ik->i", k, np.array(g["K_inv"]), k), 1e-12)
    return mean * g["y_std"] + g["y_mean"], np.sqrt(var) * g["y_std"]


def main(data: Path, out: Path):
    model = json.loads((ROOT / "tesseracts" / "vane_surrogate" / "model.json").read_text())
    with h5py.File(data / "dataset.h5") as f:
        X, Y = f["inputs"][:], f["outputs"][:]
        cols_out = list(f["outputs"].attrs["columns"])
        test = np.array([s.decode() for s in f["split"][:]]) == "test"
    lo, hi = np.array(model["lower"]), np.array(model["upper"])
    u = (X[test] - lo) / (hi - lo)
    fig, ax = plt.subplots(1, len(SHOW), figsize=(15, 3.8))
    for a, (k, label, s) in zip(ax, SHOW):
        y = Y[test, cols_out.index(k)] * s
        mu, sd = predict(model["outputs"][k], u)
        mu, sd = mu * s, sd * s
        a.errorbar(y, mu, yerr=2 * sd, fmt="o", ms=4, capsize=2, lw=0.8)
        lims = [min(y.min(), mu.min()), max(y.max(), mu.max())]
        a.plot(lims, lims, "k-", lw=0.8)
        r2 = 1 - ((mu - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        a.set_title(f"{label}\nMAE {np.abs(mu - y).mean():.3g}, R² {r2:.3f}", fontsize=9)
        a.set_xlabel("FEA (CalculiX)")
        a.set_ylabel("surrogate (±2σ)")
        a.grid(alpha=0.3)
    fig.suptitle(f"Surrogate on {int(test.sum())} held-out designs (trained on {model['n_train']}; "
                 f"data set sha256 {model['dataset_sha256'][:12]})", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "surrogate_validation.png"))
    a = ap.parse_args()
    main(Path(a.data), Path(a.out))
