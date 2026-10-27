from edgar.llm.utils import translate_to_jax


def test_translate_to_jax_replaces_numpy_import_and_all_numpy_references():
    code = """import numpy as np

def model(data, params):
    return np.sum(np.asarray(data["x"]) * params["w"])
"""

    translated = translate_to_jax(code)

    assert (
        translated
        == """import jax.numpy as jnp

def model(data, params):
    return jnp.sum(jnp.asarray(data["x"]) * params["w"])
"""
    )


def test_translate_to_jax_handles_none_and_empty_code():
    assert translate_to_jax(None) == ""
    assert translate_to_jax("") == ""
