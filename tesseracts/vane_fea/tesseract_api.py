"""Tesseract component: thermo-mechanical and modal FEA of the cooled vane (vanefea + CalculiX).

``apply`` runs CAD -> mesh -> thermal (coupled with the coolant model) -> static -> modal for one
cooling design and returns the quantities of interest and the surface fields.

The hot-gas boundary conditions are either the table bundled with vanefea (aero-thermal CFD of the
LS-89 at P0 = 10 bar, T0 = 1600 K, exit Mach 0.9) or arrays passed in by another component, for
example the ``cascade-cfd`` Tesseract in heat-transfer mode (see pipeline/cfd_to_fea.py).

The solver is not differentiable; gradients of the design response come from the
``vane-surrogate`` component, trained on data produced by this one.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Float64

from vanefea.analysis import VaneCase, run_vane
from vanefea.geometry import CoolingLayout
from vanefea.loads import ExternalBC


class HotGasTable(BaseModel):
    """Hot-gas side of the blade from 2D aero-thermal CFD (wall faces of the blade-to-blade mesh)."""
    x: Array[(None,), Float64] = Field(description="axial coordinate of the wall faces [m]")
    y: Array[(None,), Float64] = Field(description="tangential coordinate [m]")
    p: Array[(None,), Float64] = Field(description="static pressure [Pa]")
    T_aw: Array[(None,), Float64] = Field(description="adiabatic wall temperature [K]")
    h: Array[(None,), Float64] = Field(description="heat transfer coefficient [W/(m2 K)]")
    mdot_per_m: float = Field(description="main-stream mass flow per unit span of one passage [kg/(s m)]")


class InputSchema(BaseModel):
    n_passages: int = Field(7, ge=1, le=12, description="number of radial cooling passages")
    diameter_ratio: float = Field(0.40, gt=0, lt=0.8, description="passage diameter / local section thickness")
    f_start: float = Field(0.10, ge=0.0, le=1.0, description="first passage, fraction of the camber line")
    f_end: float = Field(0.80, ge=0.0, le=1.0, description="last passage, fraction of the camber line")
    coolant_fraction: float = Field(0.03, gt=0, lt=0.2, description="coolant / main-stream mass flow")
    coolant_inlet_temperature: float = Field(800.0, description="[K]")
    radial_peak: float = Field(0.0, description="mid-span excess of the gas temperature [K]")
    mesh_size: float = Field(3.0e-3, description="global element size [m]")
    hole_mesh_size: float = Field(0.9e-3, description="largest element size on the passage walls [m]")
    hot_gas: HotGasTable | None = Field(None, description="override of the bundled hot-gas table")
    return_fields: bool = Field(True, description="return the surface temperature and stress fields")

    @model_validator(mode="after")
    def _feasible(self):
        lay = CoolingLayout(n=self.n_passages, ratio=self.diameter_ratio, f_start=self.f_start, f_end=self.f_end)
        if not lay.valid():
            raise ValueError(f"cooling layout violates the manufacturing constraints: {lay.check()}")
        return self


class OutputSchema(BaseModel):
    T_max: float = Field(description="peak metal temperature [K]")
    T_mean: float = Field(description="volume-averaged metal temperature [K]")
    vm_max_core: float = Field(description="peak von Mises stress away from the supports [Pa]")
    vm_p99_core: float = Field(description="99th percentile of the von Mises stress away from the supports [Pa]")
    frequencies: Array[(None,), Float64] = Field(description="natural frequencies of the loaded vane [Hz]")
    T_coolant_out: float = Field(description="mean coolant exit temperature [K]")
    Q_gas: float = Field(description="heat load from the hot gas [W]")
    energy_balance_rel: float = Field(description="(Q_gas - Q_coolant) / Q_gas, a solution check")
    coolant_mach: float
    coolant_dp: float = Field(description="coolant pressure drop along a passage [Pa]")
    mass: float = Field(description="[kg]")
    n_nodes: int
    passage_diameters: Array[(None,), Float64] = Field(description="[m]")
    surface_xyz: Array[(None, 3), Float64] | None = None
    surface_T: Array[(None,), Float64] | None = None
    surface_vm: Array[(None,), Float64] | None = None


def apply(inputs: InputSchema) -> OutputSchema:
    case = VaneCase(layout=CoolingLayout(n=inputs.n_passages, ratio=inputs.diameter_ratio, f_start=inputs.f_start,
                                         f_end=inputs.f_end),
                    coolant_fraction=inputs.coolant_fraction, T_coolant_in=inputs.coolant_inlet_temperature,
                    radial_peak=inputs.radial_peak, h_size=inputs.mesh_size, h_hole=inputs.hole_mesh_size,
                    h_min=0.5e-3, n_curv=10)
    ext = None
    if inputs.hot_gas is not None:
        g = inputs.hot_gas
        ext = ExternalBC.from_arrays(g.x, g.y, g.p, g.T_aw, g.h, g.mdot_per_m)
    with tempfile.TemporaryDirectory(prefix="vane_fea_") as work:
        q = run_vane(case, Path(work), threads=2, external=ext)
        fields = np.load(Path(work) / "fields.npz") if inputs.return_fields else None
        out = dict(T_max=q["T_max"], T_mean=q["T_mean"], vm_max_core=q["vm_max_core"], vm_p99_core=q["vm_p99_core"],
                   frequencies=np.array([q[f"f{i}"] for i in range(1, 7) if f"f{i}" in q]),
                   T_coolant_out=q["T_coolant_out"], Q_gas=q["Q_gas_W"], energy_balance_rel=q["energy_balance_rel"],
                   coolant_mach=q["coolant_Mach"], coolant_dp=q["coolant_dp"], mass=q["mass"], n_nodes=q["n_nodes"],
                   passage_diameters=np.array(q["diameters"]))
        if fields is not None:
            out.update(surface_xyz=fields["xyz"].astype(float), surface_T=fields["T"].astype(float),
                       surface_vm=fields["vm"].astype(float))
    return OutputSchema(**out)
