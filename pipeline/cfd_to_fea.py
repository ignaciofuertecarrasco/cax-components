"""Coupled CFD -> FEA chain across two containerised solvers.

1. ``cascade-cfd`` (OpenFOAM) runs the LS-89 cascade at engine conditions in heat-transfer mode
   (adiabatic + isothermal runs) and returns the hot-gas table: wall pressure, adiabatic wall
   temperature and heat transfer coefficient on the blade surface.
2. ``vane-fea`` (CadQuery + gmsh + CalculiX) takes that table as its ``hot_gas`` input and runs the
   thermal, static and modal analysis of the cooled vane.

Nothing but arrays crosses the boundary between the two components: they could run on different
machines (``Tesseract.from_url``) or be swapped for other implementations with the same schema.

Usage::

    python pipeline/cfd_to_fea.py --P0 10e5 --T0 1600 --M2is 0.9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

ROOT = Path(__file__).resolve().parents[1]


def jsonable(x):
    """Arrays and NumPy scalars to plain Python, recursively (for the summary file)."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(P0, T0, M2is, wall_T, design: dict, out: Path):
    timings = {}
    cfd = Tesseract.from_image("cascade-cfd", timeout=7200)
    cfd.serve()
    try:
        t = time.time()
        aero = cfd.apply(dict(P0=P0, T0=T0, M2is=M2is, heat_transfer=True, wall_T=wall_T))
        timings["cfd_s"] = time.time() - t
    finally:
        cfd.teardown()
    gas = aero["hot_gas"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name("cfd_hot_gas.json").write_text(json.dumps(jsonable(aero)))     # intermediate hand-over
    print(f"CFD: zeta = {aero['zeta'] * 100:.2f} %, mass flow {aero['mdot_per_m']:.2f} kg/(s m), "
          f"T_aw {np.min(gas['T_aw']):.0f}-{np.max(gas['T_aw']):.0f} K, h {np.min(gas['h']):.0f}-{np.max(gas['h']):.0f}",
          flush=True)
    fea = Tesseract.from_image("vane-fea", timeout=7200)
    fea.serve()
    try:
        t = time.time()
        res = fea.apply(dict(**design, hot_gas=gas, return_fields=False))
        timings["fea_s"] = time.time() - t
    finally:
        fea.teardown()
    summary = dict(conditions=dict(P0=P0, T0=T0, M2is=M2is, wall_T=wall_T), design=design,
                   cfd=dict(zeta=aero["zeta"], alpha2_deg=aero["alpha2_deg"], mdot_per_m=aero["mdot_per_m"],
                            mass_imbalance=aero["mass_imbalance"], T_aw_max=float(np.max(gas["T_aw"])),
                            h_mean=float(np.mean(gas["h"]))),
                   fea={k: (np.asarray(v).tolist() if k == "frequencies" else v) for k, v in res.items()
                        if not k.startswith("surface")},
                   timings=timings)
    out.write_text(json.dumps(jsonable(summary), indent=2))
    print(json.dumps(jsonable(summary["fea"]), indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--P0", type=float, default=10e5)
    ap.add_argument("--T0", type=float, default=1600.0)
    ap.add_argument("--M2is", type=float, default=0.9)
    ap.add_argument("--wall-T", type=float, default=1100.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "cfd_to_fea.json"))
    a = ap.parse_args()
    design = dict(n_passages=7, diameter_ratio=0.40, f_start=0.10, f_end=0.80, coolant_fraction=0.03)
    main(a.P0, a.T0, a.M2is, a.wall_T, design, Path(a.out))
