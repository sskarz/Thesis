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
