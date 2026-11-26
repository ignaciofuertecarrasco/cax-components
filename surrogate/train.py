"""Fit the cooling-design surrogate on the vane FEA data set and export it for the JAX Tesseract.

One Gaussian process per quantity of interest (anisotropic squared-exponential kernel plus a noise
term, hyper-parameters by maximum likelihood with restarts), on inputs scaled to [0, 1] with the
bounds of the design sweep. The passage count n is treated as a continuous input, which keeps the
model differentiable; designs are only evaluated at integer n.

Validation on the held-out test designs of the data set: MAE, R^2 and the fraction of test points
inside the 95 % predictive interval (calibration).

The exported ``model.json`` holds what a prediction needs (training inputs, the weights
alpha = K^-1 y, the inverse covariance for the variance, kernel parameters and scalings); the
Tesseract re-implements the prediction in JAX so that it can be differentiated.

Usage::

    python surrogate/train.py --data ../vane-thermomech/results/dataset/sweep_v1 --out tesseracts/vane_surrogate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

TARGETS = ["T_max", "T_mean", "vm_p99_core", "f1", "coolant_dp", "coolant_Mach"]


def load(data_dir: Path):
    with h5py.File(data_dir / "dataset.h5") as f:
        X = f["inputs"][:]
        Y = f["outputs"][:]
        cols_in = list(f["inputs"].attrs["columns"])
        cols_out = list(f["outputs"].attrs["columns"])
        split = np.array([s.decode() for s in f["split"][:]])
        bounds = json.loads(f.attrs["bounds"])
    return X, Y, cols_in, cols_out, split, bounds


def fit(X, y, seed=0):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e2)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=8, random_state=seed)
    gp.fit(X, y)
    return gp


def export_gp(gp, X, y_mean, y_std):
    k = gp.kernel_
    const, rbf, white = k.k1.k1.constant_value, k.k1.k2.length_scale, k.k2.noise_level
    K = const * np.exp(-0.5 * _sqdist(X / rbf, X / rbf)) + (white + gp.alpha) * np.eye(len(X))
    return dict(X=X.tolist(), alpha=gp.alpha_.tolist(), K_inv=np.linalg.inv(K).tolist(), constant=float(const),
                length_scale=np.atleast_1d(rbf).tolist(), noise=float(white), y_mean=float(y_mean), y_std=float(y_std))


def _sqdist(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def main(data_dir: Path, out_dir: Path):
    X, Y, cols_in, cols_out, split, bounds = load(data_dir)
    lo = np.array([bounds[c][0] for c in cols_in])
    hi = np.array([bounds[c][1] for c in cols_in])
    U = (X - lo) / (hi - lo)
    tr, te = split == "train", split == "test"
    models, report = {}, {}
    for name in TARGETS:
        y = Y[:, cols_out.index(name)]
        m, s = y[tr].mean(), y[tr].std()
        gp = fit(U[tr], (y[tr] - m) / s)
        mu, sd = gp.predict(U[te], return_std=True)
        mu, sd = mu * s + m, sd * s
        err = mu - y[te]
        report[name] = dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                            r2=float(1 - (err ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()),
                            coverage95=float(np.mean(np.abs(err) <= 1.96 * sd)), range=float(np.ptp(y)),
                            length_scales=np.atleast_1d(gp.kernel_.k1.k2.length_scale).round(3).tolist())
        models[name] = export_gp(gp, U[tr], m, s)
        print(f"{name:14s} MAE {report[name]['mae']:.4g}  R2 {report[name]['r2']:.3f}  "
              f"95% coverage {report[name]['coverage95']:.2f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    (out_dir / "model.json").write_text(json.dumps(dict(inputs=cols_in, lower=lo.tolist(), upper=hi.tolist(),
                                                        outputs=models, dataset_sha256=manifest["sha256"],
                                                        n_train=int(tr.sum()), n_test=int(te.sum()))))
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="tesseracts/vane_surrogate")
    a = ap.parse_args()
    main(Path(a.data), Path(a.out))
