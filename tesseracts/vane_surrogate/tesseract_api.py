"""Tesseract component: differentiable surrogate of the cooled-vane FEA (Gaussian processes in JAX).

Inputs are the cooling design (passage count, diameter-to-thickness ratio, first/last passage
position, coolant flow fraction); outputs are the peak and mean metal temperature, the 99th-percentile von Mises stress,
the first natural frequency and the coolant pressure drop and Mach number, each with its
predictive standard deviation.

The model is trained on data produced by the ``vane-fea`` component (surrogate/train.py) and is
re-implemented in JAX, so every endpoint is available: ``apply`` in microseconds, and exact
``jacobian``, ``jacobian_vector_product`` and ``vector_jacobian_product`` for gradient-based design
(see pipeline/optimise_cooling.py, which differentiates through this Tesseract with tesseract-jax
and checks the optimum with the FEA component).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Differentiable, Float64
from tesseract_core.runtime.jax_recipes import jax_abstract_eval, jax_apply, jax_jacobian, jax_jvp, jax_vjp

jax.config.update("jax_enable_x64", True)

_MODEL = json.loads((Path(__file__).parent / "model.json").read_text())
_LO = jnp.array(_MODEL["lower"])
_HI = jnp.array(_MODEL["upper"])
_GPS = {k: {kk: (jnp.array(vv) if isinstance(vv, list) else vv) for kk, vv in g.items()}
        for k, g in _MODEL["outputs"].items()}


class InputSchema(BaseModel):
    n: Differentiable[Float64] = Field(description="number of passages (continuous relaxation, 4-9)")
    ratio: Differentiable[Float64] = Field(description="passage diameter / local section thickness")
    f_start: Differentiable[Float64] = Field(description="first passage position on the camber line")
    f_end: Differentiable[Float64] = Field(description="last passage position on the camber line")
    coolant: Differentiable[Float64] = Field(description="coolant / main-stream mass-flow ratio")


class OutputSchema(BaseModel):
    T_max: Differentiable[Float64]
    T_mean: Differentiable[Float64]
    vm_p99_core: Differentiable[Float64]
    f1: Differentiable[Float64]
    coolant_dp: Differentiable[Float64]
    coolant_Mach: Differentiable[Float64]
    T_max_std: Float64
    T_mean_std: Float64
    vm_p99_core_std: Float64
    f1_std: Float64


def _gp(g, u):
    k = g["constant"] * jnp.exp(-0.5 * jnp.sum(((u - g["X"]) / g["length_scale"]) ** 2, axis=-1))
    mean = k @ g["alpha"]
    var = jnp.maximum(g["constant"] + g["noise"] - k @ g["K_inv"] @ k, 1e-12)
    return mean * g["y_std"] + g["y_mean"], jnp.sqrt(var) * g["y_std"]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    x = jnp.stack([inputs[k] for k in ("n", "ratio", "f_start", "f_end", "coolant")])
    u = (x - _LO) / (_HI - _LO)
    out = {}
    for name, g in _GPS.items():
        out[name], std = _gp(g, u)
        if name in ("T_max", "T_mean", "vm_p99_core", "f1"):
            out[f"{name}_std"] = jax.lax.stop_gradient(std)
    return out


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(inputs: InputSchema, jvp_inputs: set[str], jvp_outputs: set[str],
                            tangent_vector: dict[str, Any]):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str],
                            cotangent_vector: dict[str, Any]):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
