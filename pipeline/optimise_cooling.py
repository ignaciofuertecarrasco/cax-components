"""Gradient-based cooling design through a differentiable Tesseract, verified by the FEA Tesseract.

Problem: minimise the coolant mass flow of the vane subject to

* volume-averaged metal temperature  T_mean  <= T_LIMIT (the bulk temperature governs creep life; the
  peak temperature sits at the uncooled trailing edge and hardly depends on the passage layout, see the
  vane-thermomech README),
* 99th-percentile von Mises stress  <= S_LIMIT (away from the idealised supports),
* coolant Mach number in the passages <= MACH_LIMIT (pressure losses, choking margin),
* the manufacturing constraints of the layout (wall thickness, ligament), evaluated geometrically.

For every passage count n = 4 ... 9 a sequential quadratic programme (SLSQP) is solved with exact
gradients: the objective and constraints are JAX functions of the inputs of the ``vane-surrogate``
Tesseract, called through ``tesseract_jax.apply_tesseract`` so that ``jax.grad`` differentiates
through the component's vector-Jacobian-product endpoint. To account for the surrogate error, the
thermal and stress limits are applied to mean + 2 standard deviations.

The best design is then re-analysed with the ``vane-fea`` Tesseract (CAD + mesh + CalculiX), and the
prediction is compared with the simulation.

Usage (images built with scripts/build.sh)::

    python pipeline/optimise_cooling.py                      # surrogate and FEA as Docker images
    python pipeline/optimise_cooling.py --local              # surrogate from its tesseract_api.py, no Docker
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
T_LIMIT = 1150.0          # mean metal temperature [K]
S_LIMIT = 650e6           # [Pa]
MACH_LIMIT = 0.45
VARS = ["ratio", "f_start", "f_end", "coolant"]


def bounds(model):
    lo = dict(zip(model["inputs"], model["lower"]))
    hi = dict(zip(model["inputs"], model["upper"]))
    return [(lo[k], hi[k]) for k in VARS]


def make_functions(surrogate: Tesseract, n: int):
    def outputs(z):
        inp = {"n": jnp.asarray(float(n)), **{k: z[i] for i, k in enumerate(VARS)}}
        return apply_tesseract(surrogate, inp)

    def objective(z):
        return z[3] * 100.0                           # coolant flow in per cent of the main flow

    def constraints(z):                               # >= 0 when satisfied (scaled to O(1))
        o = outputs(z)
        sg = jax.lax.stop_gradient           # the predictive spread is not a differentiable output
        return jnp.stack([(T_LIMIT - (o["T_mean"] + 2 * sg(o["T_mean_std"]))) / 50.0,
                          (S_LIMIT - (o["vm_p99_core"] + 2 * sg(o["vm_p99_core_std"]))) / 1e8,
                          (MACH_LIMIT - o["coolant_Mach"]) / 0.1])

    # one reverse-mode gradient per constraint (three VJP calls to the component), compiled once
    n_con = 3
    jac = jax.jit(lambda z: jnp.stack([jax.grad(lambda zz, i=i: constraints(zz)[i])(z) for i in range(n_con)]))
    return outputs, objective, jax.jit(constraints), jac


def geometry_constraint(n):
    """Manufacturing margins (wall, ligament) in mm; finite-difference gradients (cheap, geometric)."""
    from vanefea.geometry import CoolingLayout

    def g(z):
        m = CoolingLayout(n=n, ratio=z[0], f_start=z[1], f_end=z[2]).check()
        return np.array([m["wall_margin"], m["ligament_margin"]]) * 1e3
    return g


def starting_points(model: dict, n: int, n_random: int, seed: int) -> list:
    """Warm starts from the training designs with this passage count (feasible layouts by construction,
    since the sweep rejected the others), plus a few random points; SLSQP is a local method."""
    lo, hi = np.array(model["lower"]), np.array(model["upper"])
    X = lo + np.array(model["outputs"]["T_mean"]["X"]) * (hi - lo)
    cols = [model["inputs"].index(k) for k in VARS]
    data = [x[cols] for x in X if round(x[model["inputs"].index("n")]) == n]
    rng = np.random.default_rng(seed + n)
    rand = [np.array([a + rng.uniform(0.2, 0.8) * (c - a) for a, c in bounds(model)]) for _ in range(n_random)]
    return data[:10] + rand


def optimise(surrogate: Tesseract, model: dict, n: int, n_random: int = 3, seed: int = 0):
    outputs, objective, constraints, jac = make_functions(surrogate, n)
    gfun = geometry_constraint(n)
    b = bounds(model)
    best = None
    for z0 in starting_points(model, n, n_random, seed):
        cons = [dict(type="ineq", fun=lambda z: np.asarray(constraints(jnp.asarray(z))),
                     jac=lambda z: np.asarray(jac(jnp.asarray(z)))),
                dict(type="ineq", fun=gfun)]
        res = minimize(lambda z: float(objective(jnp.asarray(z))), z0,
                       jac=lambda z: np.array([0, 0, 0, 100.0]), bounds=b, constraints=cons,
                       method="SLSQP", options=dict(maxiter=200, ftol=1e-9))
        feasible = (np.asarray(constraints(jnp.asarray(res.x))) >= -1e-4).all() and (gfun(res.x) >= -1e-4).all()
        if feasible and (best is None or res.fun < best[1]):
            best = (res.x, res.fun)
    if best is None:
        return None
    z = best[0]
    o = {k: float(v) for k, v in outputs(jnp.asarray(z)).items()}
    return dict(n=n, **dict(zip(VARS, map(float, z))), coolant_percent=float(best[1]), predicted=o)


def plot(result, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = [d for d in result["designs"] if "fea" in d]
    n = [d["n"] for d in ds]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].bar(n, [d["coolant_percent"] for d in ds], color="C0", alpha=0.8)
    ax[0].set_xlabel("number of cooling passages")
    ax[0].set_ylabel("minimum coolant flow [% of main flow]")
    ax[0].set_title(f"Least coolant meeting T_mean <= {result['limits']['T_mean']:.0f} K (mean + 2 sigma),\n"
                    f"stress p99 <= {result['limits']['vm_p99'] / 1e6:.0f} MPa, coolant Mach <= "
                    f"{result['limits']['coolant_mach']}", fontsize=9)
    for k, (key, label, scale) in enumerate((("T_mean", "T_mean [K]", 1), ("vm_p99_core", "stress p99 [MPa]", 1e-6))):
        a = ax[1] if k == 0 else ax[1].twinx()
        pred = [d["predicted"][key] * scale for d in ds]
        fea = [d["fea"][key] * scale for d in ds]
        off = -0.12 if k == 0 else 0.12
        a.plot(np.array(n) + off, pred, "o", mfc="w", c=f"C{k}", label=f"{label}, surrogate")
        a.plot(np.array(n) + off, fea, "s", c=f"C{k}", label=f"{label}, FEA container")
        a.set_ylabel(label, color=f"C{k}")
    ax[1].axhline(result["limits"]["T_mean"], c="C0", ls=":", lw=1)
    ax[1].set_xlabel("number of cooling passages")
    ax[1].set_title("Optimum of each passage count: surrogate prediction vs FEA re-analysis", fontsize=9)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = fig.axes[-1].get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main(local: bool, verify: bool, out: Path):
    model = json.loads((ROOT / "tesseracts" / "vane_surrogate" / "model.json").read_text())
    if local:
        surrogate = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "vane_surrogate" / "tesseract_api.py")
    else:
        surrogate = Tesseract.from_image("vane-surrogate")
        surrogate.serve()
    t0 = time.time()
    designs = []
    try:
        for n in range(4, 10):
            d = optimise(surrogate, model, n)
            print(f"n = {n}: " + (f"coolant {d['coolant_percent']:.2f} %  T_mean {d['predicted']['T_mean']:.0f} K"
                                  if d else "no feasible design"), flush=True)
            if d:
                designs.append(d)
        # gradient check of the component against finite differences, at the best design
        best = min(designs, key=lambda d: d["coolant_percent"])
        probe = {"n": float(best["n"]), **{k: best[k] for k in VARS}}
        grad_T = jax.grad(lambda z: apply_tesseract(surrogate, {**probe, "coolant": z})["T_mean"])(
            jnp.asarray(probe["coolant"]))
        eps = 1e-6
        fd = (surrogate.apply({**probe, "coolant": probe["coolant"] + eps})["T_mean"]
              - surrogate.apply({**probe, "coolant": probe["coolant"] - eps})["T_mean"]) / (2 * eps)
    finally:
        if not local:
            surrogate.teardown()
    result = dict(limits=dict(T_mean=T_LIMIT, vm_p99=S_LIMIT, coolant_mach=MACH_LIMIT), designs=designs, best=best,
                  gradient_check=dict(dTmean_dcoolant_autodiff=float(grad_T), dTmean_dcoolant_fd=float(fd)),
                  optimisation_time_s=time.time() - t0)
    if verify:
        # every per-n optimum is re-analysed: the surrogate is checked exactly where it was trusted
        fea = Tesseract.from_image("vane-fea", timeout=3600)
        fea.serve()
        try:
            for d in designs:
                t1 = time.time()
                r = fea.apply(dict(n_passages=d["n"], diameter_ratio=d["ratio"], f_start=d["f_start"],
                                   f_end=d["f_end"], coolant_fraction=d["coolant"], return_fields=False))
                d["fea"] = dict(T_max=float(r["T_max"]), T_mean=float(r["T_mean"]), vm_p99_core=float(r["vm_p99_core"]),
                                f1=float(r["frequencies"][0]), coolant_Mach=float(r["coolant_mach"]),
                                energy_balance_rel=float(r["energy_balance_rel"]), time_s=time.time() - t1)
                print(f"n = {d['n']}: FEA T_mean {d['fea']['T_mean']:.1f} K (predicted {d['predicted']['T_mean']:.1f}), "
                      f"stress p99 {d['fea']['vm_p99_core'] / 1e6:.0f} MPa "
                      f"(predicted {d['predicted']['vm_p99_core'] / 1e6:.0f})", flush=True)
        finally:
            fea.teardown()
        result["verification"] = best.get("fea")
        plot(result, out.with_suffix(".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="load the surrogate from source instead of its image")
    ap.add_argument("--no-verify", action="store_true", help="skip the FEA check of the optimum")
    ap.add_argument("--out", default=str(ROOT / "results" / "optimise_cooling.json"))
    a = ap.parse_args()
    main(a.local, not a.no_verify, Path(a.out))
