"""Scaffold tests for SFT v9 (lever2) mix / recipe wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.iffc import format_iffc_execution_user_message
from sft.build_mix_v7 import MixAbort
from sft.build_mix_v9 import assemble_v9_pool, build_v9_mix
from sft.config import (
    SEED_V9,
    SOURCE_APIGEN_MT,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
    SPLITS_DIR,
    SPLITS_DIR_V7,
    SPLITS_DIR_V8,
    SPLITS_DIR_V9,
    V9_EXPERIMENT,
    V9_MAX_APIGEN_FRAC,
    V9_MAX_SEQ_LENGTH,
    V9_MIN_TOOLACE_XLAM_FRAC,
    V9_PROBE_IDS,
    V9_PROBE_IDS_VC_SIDECAR,
)
from sft.filter_empty_after_exec import (
    RECIPE_V4,
    RECIPE_V5,
    RECIPE_V7,
    RECIPE_V9,
    flags_for_recipe,
    recipe_v4_drop_flags,
    recipe_v5_flags,
    resolve_recipe,
)
from sft.mix_stats_v9 import collect_stats, gate_reasons
from sft.pref_pairs_antiloop_v9 import dry_run_report, pair_schema_example


def _call(name: str = "ls", **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _exec_obs(func_string: str = "ls()", result: object | None = None) -> str:
    return format_iffc_execution_user_message(
        [func_string], [result if result is not None else {"files": ["a"]}]
    )


def _rec(
    example_id: str,
    messages: list[dict],
    *,
    source: str,
    n_tools: int = 1,
) -> dict:
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": True,
        "n_tools_in_gold": n_tools,
        "n_tools_bin": "ge2" if n_tools >= 2 else "1",
        "chat_format": "iffc",
        "messages": messages,
    }


def _toolace(i: int) -> dict:
    return _rec(
        f"toolace-{i}",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"task {i}"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": _exec_obs()},
            {"role": "assistant", "content": _call("cat", path=str(i))},
        ],
        source=SOURCE_TOOLACE,
    )


def _xlam(i: int) -> dict:
    return _rec(
        f"xlam-{i}",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"call {i}"},
            {"role": "assistant", "content": _call("fn", i=i)},
        ],
        source=SOURCE_XLAM,
    )


def _apigen(i: int, kind: str = "final") -> dict:
    if kind == "ask":
        asst = "Which parameter should I use?"
    elif kind == "refuse":
        asst = "I cannot fulfill that request; it is out of scope."
    else:
        asst = f"Final answer for {i}."
    return _rec(
        f"apigen-{kind}-{i}",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"q {i}"},
            {"role": "assistant", "content": _call("tool")},
            {"role": "user", "content": _exec_obs("tool()")},
            {"role": "assistant", "content": asst},
        ],
        source=SOURCE_APIGEN_MT,
    )


def _pools(n_t: int = 60, n_x: int = 25, n_a: int = 30):
    toolace = [_toolace(i) for i in range(n_t)]
    xlam = [_xlam(i) for i in range(n_x)]
    apigen = (
        [_apigen(i, "final") for i in range(n_a // 3)]
        + [_apigen(i, "ask") for i in range(n_a // 3)]
        + [_apigen(i, "refuse") for i in range(n_a - 2 * (n_a // 3))]
    )
    return apigen, toolace, xlam


def test_v9_constants_locked():
    assert SEED_V9 == 49
    assert V9_EXPERIMENT == "sft-v9"
    assert V9_MAX_APIGEN_FRAC == 0.35
    assert V9_MAX_SEQ_LENGTH == 8192
    assert SPLITS_DIR_V9.name == "splits-v9"
    assert "_4" in V9_PROBE_IDS and "_163" in V9_PROBE_IDS
    assert len(V9_PROBE_IDS) == 10
    assert "_52" not in V9_PROBE_IDS and "_96" not in V9_PROBE_IDS
    assert V9_PROBE_IDS_VC_SIDECAR == ("_52", "_60", "_74", "_76", "_83", "_96")


def test_v9_recipe_is_v4_empty_no_v5_synth():
    assert resolve_recipe(recipe="v9") == RECIPE_V9
    assert recipe_v4_drop_flags(RECIPE_V9) == (True, True)
    assert recipe_v5_flags(RECIPE_V9) == (False, False, False, False)
    flags = flags_for_recipe(RECIPE_V9)
    assert flags["synthesize_complete_turn0"] is False
    assert flags["drop_underspecified_turn0"] is False


def test_assemble_v9_pool_source_shares():
    apigen, toolace, xlam = _pools()
    pool = assemble_v9_pool(apigen, toolace, xlam, seed=SEED_V9)
    n = len(pool)
    n_a = sum(1 for r in pool if r["source"] == SOURCE_APIGEN_MT)
    n_tx = sum(
        1 for r in pool if r["source"] in {SOURCE_TOOLACE, SOURCE_XLAM}
    )
    assert n_a / n <= V9_MAX_APIGEN_FRAC + 1e-9
    assert n_a / n >= 0.15 - 1e-9
    assert n_tx / n >= V9_MIN_TOOLACE_XLAM_FRAC - 1e-9


def test_build_mix_v9_writes_splits_seed_49_refuses_frozen(
    tmp_path: Path,
):
    apigen, toolace, xlam = _pools()
    out = tmp_path / "splits-v9"
    manifest = build_v9_mix(
        apigen=apigen,
        toolace=toolace,
        xlam=xlam,
        out_dir=out,
        seed=SEED_V9,
        skip_gates=False,
    )
    assert manifest["experiment"] == V9_EXPERIMENT
    assert manifest["seed"] == SEED_V9
    assert manifest["lever"] == "lever2"
    assert "loss_mask" in manifest
    assert (out / "train.jsonl").exists()
    assert (out / "MANIFEST.json").exists()
    # Refuse frozen dirs
    with pytest.raises(MixAbort):
        build_v9_mix(
            apigen=apigen,
            toolace=toolace,
            xlam=xlam,
            out_dir=SPLITS_DIR,
            seed=SEED_V9,
            skip_gates=True,
        )
    with pytest.raises(MixAbort):
        build_v9_mix(
            apigen=apigen,
            toolace=toolace,
            xlam=xlam,
            out_dir=SPLITS_DIR_V7,
            seed=SEED_V9,
            skip_gates=True,
        )
    with pytest.raises(MixAbort):
        build_v9_mix(
            apigen=apigen,
            toolace=toolace,
            xlam=xlam,
            out_dir=SPLITS_DIR_V8,
            seed=SEED_V9,
            skip_gates=True,
        )


def test_build_mix_v9_has_no_loop_filter_import():
    import sft.build_mix_v9 as mod

    # No runtime import / symbol from v8 teacher loop filter.
    assert not hasattr(mod, "filter_loop_trajectories")
    assert "student_filter_v8" not in getattr(mod, "__dict__", {})
    import ast
    from pathlib import Path

    tree = ast.parse(Path(mod.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    joined = " ".join(imported)
    assert "filter_loop_trajectories" not in joined
    assert "student_filter_v8" not in joined


def test_mix_stats_v9_conversation_gates(tmp_path: Path):
    apigen, toolace, xlam = _pools()
    out = tmp_path / "splits-v9"
    build_v9_mix(
        apigen=apigen,
        toolace=toolace,
        xlam=xlam,
        out_dir=out,
        seed=SEED_V9,
    )
    from sft.io import read_jsonl

    rows = read_jsonl(out / "train.jsonl")
    stats = collect_stats(rows)
    reasons = gate_reasons(stats, require_unrolled_actions=False)
    assert reasons == [], reasons


def test_pref_pairs_exact_prefix_dry_run():
    schema = pair_schema_example()
    assert schema["pair_kind"] == "advance_vs_loop"
    assert schema["schema_version"] == 2
    apigen, toolace, xlam = _pools(n_t=5, n_x=2, n_a=3)
    report = dry_run_report(toolace + apigen)
    assert report["training"] is False
    assert "pairs_mined" in report


def test_prepare_mlx_v9_quota_wiring(tmp_path: Path, capsys):
    from sft.train_mlx import prepare_mlx_data

    apigen, toolace, xlam = _pools(n_t=40, n_x=15, n_a=20)
    splits = tmp_path / "splits-v9"
    build_v9_mix(
        apigen=apigen,
        toolace=toolace,
        xlam=xlam,
        out_dir=splits,
        seed=SEED_V9,
        skip_gates=True,
    )
    mlx = tmp_path / "mlx-v9"
    prepare_mlx_data(
        splits_dir=splits,
        mlx_dir=mlx,
        recipe=RECIPE_V9,
        smoke=True,
        smoke_train_rows=50,
        smoke_valid_rows=5,
        max_seq_length=V9_MAX_SEQ_LENGTH,
    )
    assert (mlx / "train.jsonl").exists()
    out = capsys.readouterr().out
    assert "v9 action-class quota" in out or "v9_quota=on" in out
