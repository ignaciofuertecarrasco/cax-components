"""Tesseract component: OpenFOAM RANS of the LS-89 turbine cascade (turboai.api).

Two modes:

* ``heat_transfer = False``: one adiabatic run; returns loss coefficient, exit angle, mass flow,
  blade force and the wall distributions (static pressure, isentropic Mach number);
* ``heat_transfer = True``: an adiabatic and an isothermal run on the same mesh (in parallel); returns
  in addition the hot-gas table (adiabatic wall temperature, heat transfer coefficient) whose fields
  match the ``hot_gas`` input of the ``vane-fea`` component.

The blade is a member of the six-parameter LS-89 family (all zero = the original vane), optionally
with local Gaussian bumps or dents. One run takes 5-10 minutes on one core.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Float64


def _load_openfoam_environment():
    """Import the variables of OpenFOAM's bashrc into this process (the runtime does not use a login shell)."""
    if os.environ.get("WM_PROJECT_DIR"):
        return
    rc = subprocess.run(["bash", "-c", "source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1; env -0"],
                        capture_output=True, check=True).stdout
    for item in rc.split(b"\0"):
        if b"=" in item:
            k, v = item.decode().split("=", 1)
            os.environ[k] = v
    extra = os.environ.get("TURBOAI_FOAM_EXTRA_LIBS")
    if extra:   # project extension (viscousWork fvOption), loaded by name through LD_LIBRARY_PATH
        os.environ["LD_LIBRARY_PATH"] = extra + ":" + os.environ.get("LD_LIBRARY_PATH", "")


_load_openfoam_environment()


class Bump(BaseModel):
    side: str = Field(description="'upper' (suction side) or 'lower' (pressure side)")
    s0: float = Field(ge=0, le=1, description="centre, fraction of the side length from the LE")
    w: float = Field(gt=0, description="Gaussian width, fraction of the side length")
    amp: float = Field(description="amplitude in chord units; positive = bump, negative = dent")


class Shape(BaseModel):
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    dstagger: float = 0.0


class InputSchema(BaseModel):
    P0: float = Field(1.6e5, gt=0, description="inlet total pressure [Pa]")
    T0: float = Field(420.0, gt=0, description="inlet total temperature [K]")
    M2is: float = Field(0.9, gt=0.3, lt=1.3, description="exit isentropic Mach number")
    alpha: float = Field(0.0, ge=-20, le=20, description="inlet flow angle [deg]")
    Tu: float = Field(0.01, ge=0, le=0.2, description="inlet turbulence intensity")
    shape: Shape = Field(default_factory=Shape, description="LS-89 family parameters")
    bumps: list[Bump] = Field(default_factory=list)
    heat_transfer: bool = Field(False, description="also compute the hot-gas boundary conditions")
    wall_T: float = Field(1100.0, description="heat-transfer mode: wall temperature of the isothermal run [K]")
    mesh: str = Field("coarse", pattern="^(coarse|fine)$")
    iterations: int = Field(3000, ge=400, le=20000)


class Wall(BaseModel):
    x: Array[(None,), Float64]
    y: Array[(None,), Float64]
    s: Array[(None,), Float64] = Field(description="surface distance from the LE, > 0 on the suction side [m]")
    p: Array[(None,), Float64]
    Mis: Array[(None,), Float64]


class HotGas(BaseModel):
    x: Array[(None,), Float64]
    y: Array[(None,), Float64]
    p: Array[(None,), Float64]
    T_aw: Array[(None,), Float64]
    h: Array[(None,), Float64]
    mdot_per_m: float


class OutputSchema(BaseModel):
    zeta: float = Field(description="total-pressure loss coefficient")
    alpha2_deg: float = Field(description="mass-averaged exit flow angle [deg]")
    mdot_per_m: float = Field(description="mass flow per unit span [kg/(s m)]")
    M2is_actual: float
    zweifel: float
    mass_imbalance: float = Field(description="inlet/outlet mass-flow imbalance, a convergence check")
    throat_mm: float
    wall: Wall
    hot_gas: HotGas | None = None


def apply(inputs: InputSchema) -> OutputSchema:
    from turboai import api
    common = dict(P0=inputs.P0, T0=inputs.T0, M2is=inputs.M2is, alpha=inputs.alpha, Tu=inputs.Tu,
                  shape=inputs.shape.model_dump(), bumps=[b.model_dump() for b in inputs.bumps],
                  mesh=inputs.mesh, iters=inputs.iterations)
    hot = None
    if inputs.heat_transfer:
        r = api.aerothermal(wall_T=inputs.wall_T, **common)
        sc, geo = r["scalars"], r["geometry"]
        wall = dict(x=r["x"], y=r["y"], s=r["s"], p=r["p"], Mis=r["Mis"])
        hot = HotGas(x=r["x"], y=r["y"], p=r["p"], T_aw=r["T_aw"], h=r["h"], mdot_per_m=r["mdot_per_m"])
    else:
        r = api.simulate(**common)
        sc, geo = r["scalars"], r["geometry"]
        wall = {k: np.asarray(r["wall"][k], dtype=float) for k in ("x", "y", "s", "p", "Mis")}
    return OutputSchema(zeta=sc["zeta"], alpha2_deg=sc["alpha2_deg"], mdot_per_m=sc["mdot_per_m"],
                        M2is_actual=sc["M2is_actual"], zweifel=sc["zweifel"],
                        mass_imbalance=sc.get("mass_imbalance", float("nan")), throat_mm=geo["throat_mm"],
                        wall=Wall(**wall), hot_gas=hot)
