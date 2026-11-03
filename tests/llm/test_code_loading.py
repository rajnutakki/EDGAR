import pytest

from edgar.llm.code_loading import load_class_from_source, load_function_from_source


def test_load_function_from_source_returns_callable_from_fresh_namespace():
    source = """
MULTIPLIER = 3

def scale(value):
    return value * MULTIPLIER
"""

    scale = load_function_from_source(source, "scale")

    assert callable(scale)
    assert scale(4) == 12


@pytest.mark.parametrize("source", [None, "", "   \n\t"])
def test_load_function_from_source_returns_none_for_missing_source(source):
    assert load_function_from_source(source, "model") is None


def test_load_function_from_source_returns_none_for_missing_entrypoint():
    source = "def other_function():\n    return 1\n"

    assert load_function_from_source(source, "model") is None


def test_load_function_from_source_returns_none_when_entrypoint_is_not_callable():
    source = "model = 42\n"

    assert load_function_from_source(source, "model") is None


def test_load_function_from_source_warns_and_returns_none_on_exec_error():
    source = "def model(:\n    return 1\n"

    with pytest.warns(UserWarning, match=r"exec failed for 'model'"):
        result = load_function_from_source(source, "model")

    assert result is None


def test_load_class_from_source_returns_class_from_fresh_namespace():
    source = """
class Diagnostics:
    def __init__(self, value=5):
        self.value = value

    def compute(self):
        return self.value * 2
"""

    diagnostics_cls = load_class_from_source(source, "Diagnostics")

    assert isinstance(diagnostics_cls, type)
    assert diagnostics_cls().compute() == 10


@pytest.mark.parametrize("source", [None, "", "   \n\t"])
def test_load_class_from_source_returns_none_for_missing_source(source):
    assert load_class_from_source(source, "Diagnostics") is None


def test_load_class_from_source_returns_none_when_entrypoint_is_not_a_class():
    source = "def Diagnostics():\n    return 1\n"

    assert load_class_from_source(source, "Diagnostics") is None


def test_load_class_from_source_warns_and_returns_none_on_exec_error():
    source = "class Diagnostics(:\n    pass\n"

    with pytest.warns(UserWarning, match=r"exec failed for 'Diagnostics'"):
        result = load_class_from_source(source, "Diagnostics")

    assert result is None
