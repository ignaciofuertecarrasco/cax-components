"""The differentiable component, loaded from source (no Docker): schema, physics trends and gradients."""
from pathlib import Path

import numpy as np
import pytest

API = Path(__file__).resolve().parents[1] / "tesseracts" / "vane_surrogate" / "tesseract_api.py"
pytestmark = pytest.mark.skipif(not (API.parent / "model.json").exists(), reason="surrogate not trained yet")

DESIGN = {"n": 7.0, "ratio": 0.40, "f_start": 0.12, "f_end": 0.80, "coolant": 0.03}


@pytest.fixture(scope="module")
def surrogate():
    from tesseract_core import Tesseract
    return Tesseract.from_tesseract_api(API)


def test_apply_returns_physical_values(surrogate):
    out = surrogate.apply(DESIGN)
    assert 900 < out["T_mean"] < 1500
    assert out["T_mean"] < out["T_max"] < 1600
    assert out["T_max_std"] > 0 and out["f1"] > 0


def test_more_coolant_means_cooler_metal(surrogate):
    lo = surrogate.apply({**DESIGN, "coolant": 0.02})["T_mean"]
    hi = surrogate.apply({**DESIGN, "coolant": 0.04})["T_mean"]
    assert hi < lo


def test_vjp_matches_finite_differences(surrogate):
    eps = 1e-6
    for var in ("coolant", "ratio", "f_end"):
        vjp = surrogate.vector_jacobian_product(DESIGN, vjp_inputs=[var], vjp_outputs=["T_mean"],
                                                cotangent_vector={"T_mean": np.array(1.0)})[var]
        fd = (surrogate.apply({**DESIGN, var: DESIGN[var] + eps})["T_mean"]
              - surrogate.apply({**DESIGN, var: DESIGN[var] - eps})["T_mean"]) / (2 * eps)
        assert np.isclose(float(vjp), float(fd), rtol=1e-3, atol=1e-3 * abs(float(fd)) + 1e-6)


def test_jax_gradient_through_the_component(surrogate):
    import jax
    import jax.numpy as jnp
    from tesseract_jax import apply_tesseract
    jax.config.update("jax_enable_x64", True)

    def t_mean(c):
        return apply_tesseract(surrogate, {**{k: jnp.asarray(v) for k, v in DESIGN.items()}, "coolant": c})["T_mean"]

    g = jax.grad(t_mean)(jnp.asarray(DESIGN["coolant"]))
    assert float(g) < 0                     # more coolant, lower mean temperature
