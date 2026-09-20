"""Registry / select_model identity — SFT Gemma must not collide with base Gemma."""

from benchmark import config


def test_sft_iffc_model_is_distinct_from_base_gemma():
    key = "gemma4-e2b-sft-iffc"
    assert key in config.MODELS

    sft = config.MODELS[key]
    base = config.MODELS["gemma4-e2b"]
    assert sft["api_name"] == "gemma4-e2b-sft-iffc"
    assert sft["display"] == "Gemma-4-E2B-it SFT-IFFC (Q4_K_M)"
    assert sft["slug"] == "Gemma-4-E2B-it-SFT-IFFC-Q4_K_M"
    assert sft["api_name"] != base["api_name"]
    assert sft["slug"] != base["slug"]
    assert sft["display"] != base["display"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc"
        assert config.MODEL_DISPLAY == "Gemma-4-E2B-it SFT-IFFC (Q4_K_M)"
        assert config.MODEL_SLUG == "Gemma-4-E2B-it-SFT-IFFC-Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / sft["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / base["slug"]
    finally:
        config.select_model(previous)


def test_sft_iffc_v2_model_is_distinct_from_v1_and_base():
    key = "gemma4-e2b-sft-iffc-v2"
    assert key in config.MODELS

    v2 = config.MODELS[key]
    v1 = config.MODELS["gemma4-e2b-sft-iffc"]
    base = config.MODELS["gemma4-e2b"]
    assert v2["api_name"] == "gemma4-e2b-sft-iffc-v2"
    assert v2["display"] == "Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] (Q4_K_M)"
    assert v2["slug"] == "Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M"
    assert v2["gguf_path"] == "sft/fused/gemma-4-e2b-it-iffc-v2-best2000-Q4_K_M.gguf"
    assert v2["api_name"] != v1["api_name"]
    assert v2["slug"] != v1["slug"]
    assert v2["display"] != v1["display"]
    assert v2["api_name"] != base["api_name"]
    assert v2["slug"] != base["slug"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc-v2"
        assert config.MODEL_DISPLAY == "Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] (Q4_K_M)"
        assert config.MODEL_SLUG == "Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / v2["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v1["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / base["slug"]
    finally:
        config.select_model(previous)


def test_iffc_strip_empty_assistant_nests_figures_without_changing_model_key():
    previous = config.DEFAULT_MODEL
    try:
        config.select_model("gemma4-e2b-sft-iffc-v2")
        standard_figures = config.FIGURES_DIR
        config.set_iffc_strip_empty_assistant(True)
        config.apply_iffc_strip_empty_assistant_result_paths()
        assert config.IFFC_STRIP_EMPTY_ASSISTANT is True
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc-v2"
        nested = (
            config.MODEL_RESULTS_DIR / config.IFFC_STRIP_EMPTY_ASSISTANT_DIRNAME
        )
        assert config.FIGURES_DIR == nested / "figures"
        assert config.METRICS_DIR == nested / "metrics"
        assert config.SCORES_DIR == nested / "scores"
        assert config.FIGURES_DIR != standard_figures
        assert config.FIGURES_DIR.is_dir()
    finally:
        config.set_iffc_strip_empty_assistant(False)
        config.select_model(previous)
