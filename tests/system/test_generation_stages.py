"""Stage orchestration without provider calls; persisted schema and timing are checked."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import jax
import jax.numpy as jnp

from edgar.run import run
from tests.system.fake_runner import build_fake_spec


MODEL = "import jax.numpy as jnp\ndef model(data, params):\n    return params['a'] * data['x']"
TRANSLATED = MODEL + "\n# translated"


@pytest.mark.parametrize(
    "stages",
    [
        ["generate_models"],
        ["generate_models", "generate_param_ests"],
        ["generate_models", "translate_programs"],
        ["generate_models", "generate_param_ests", "translate_programs"],
    ],
)
def test_stage_order_seeds_scoring_and_persistence(tmp_path, stages):
    spec = build_fake_spec(tmp_path / "out")
    spec.run["generation_stages"] = stages
    spec.evolution.update(
        n_generations=1, n_islands=1, batch_size=1, topology=[0], n_migrants=0
    )
    for seed in spec.seed_programs:
        seed.code.model = MODEL
    events = []
    translation = "translate_programs" in stages

    async def generate(population, *args, **kwargs):
        events.append("generate_models")
        for p in population:
            if p.code.model is None:
                p.code.model = MODEL
                p.default_params = {"a": 1.0}

    async def estimators(population, *args, **kwargs):
        events.append("generate_param_ests")
        for p in population:
            if p.birth.generation == 0:
                p.code.param_est = (
                    "def parameter_estimator(data):\n    return {'a': 1.0}"
                )

    async def translate(population, *args, **kwargs):
        is_seed_translation = all(p.birth.generation == -1 for p in population)
        events.append(
            "translate_seeds" if is_seed_translation else "translate_programs"
        )
        for p in population:
            if p.code.model is not None and p.code.model_jax is None:
                p.code.model_jax = TRANSLATED

    def score(population, *args, **kwargs):
        events.append(
            "score_" + ("seeds" if kwargs.get("n_items") == 2 else "generated")
        )
        for p in population:
            if p.status != "alive":
                continue
            assert p.code.model_jax == (TRANSLATED if translation else MODEL)
            if not translation:
                # Exercise the real model loader and JAX tracer, not just field presence.
                assert (
                    float(
                        jax.jit(p.compile_model())(
                            {"x": jnp.array(2.0)}, {"a": jnp.array(3.0)}
                        )
                    )
                    == 6.0
                )
            if p.birth.generation == -1:
                assert p.code.param_est  # seed estimators survive skipped generation
            elif "generate_param_ests" not in stages:
                assert p.code.param_est is None
            p.program_losses.discover.final = 1.0

    with (
        patch("edgar.llm.generate.generate_models", generate),
        patch("edgar.llm.generate.generate_param_ests", estimators),
        patch("edgar.run.t_translate_seeds", translate),
        patch("edgar.llm.generate.translate_programs", translate),
        patch("edgar.run.t_score_seeds", score),
        patch("edgar.run.t_score", score),
        patch("edgar.run.t_score_validate"),
        patch("edgar.run.rank"),
        patch("edgar.run.t_fits_seeds"),
        patch("edgar.run.t_fits"),
        patch("edgar.run.t_deduplicate"),
        patch("edgar.run.t_prune"),
        patch("edgar.run.t_migrate"),
    ):
        asyncio.run(run(spec))

    assert events == (["translate_seeds"] if translation else []) + [
        "score_seeds",
        *stages,
        "score_generated",
    ]
    output = Path(spec.output_dir)
    rows = [
        json.loads(line)
        for line in (output / "population.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(
        row["code"]["model_jax"] == (TRANSLATED if translation else MODEL)
        for row in rows
    )
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    metrics = (output / "metrics.jsonl").read_text()
    if not translation:
        assert '"translate_programs"' not in metrics
        assert '"translate_seeds"' not in metrics
