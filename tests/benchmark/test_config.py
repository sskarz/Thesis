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


def test_sft_iffc_v4_model_is_distinct_from_v1_v2_and_base():
    key = "gemma4-e2b-sft-iffc-v4"
    assert key in config.MODELS

    v4 = config.MODELS[key]
    v2 = config.MODELS["gemma4-e2b-sft-iffc-v2"]
    v1 = config.MODELS["gemma4-e2b-sft-iffc"]
    base = config.MODELS["gemma4-e2b"]
    assert v4["api_name"] == "gemma4-e2b-sft-iffc-v4"
    assert v4["display"] == "Gemma-4-E2B-it SFT-IFFC-v4 (Q4_K_M)"
    assert v4["slug"] == "Gemma-4-E2B-it-SFT-IFFC-v4-Q4_K_M"
    assert v4["gguf_path"] == "sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf"
    assert v4["api_name"] != v2["api_name"]
    assert v4["slug"] != v2["slug"]
    assert v4["display"] != v2["display"]
    assert v4["api_name"] != v1["api_name"]
    assert v4["slug"] != v1["slug"]
    assert v4["api_name"] != base["api_name"]
    assert v4["slug"] != base["slug"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc-v4"
        assert config.MODEL_DISPLAY == "Gemma-4-E2B-it SFT-IFFC-v4 (Q4_K_M)"
        assert config.MODEL_SLUG == "Gemma-4-E2B-it-SFT-IFFC-v4-Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / v4["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v2["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v1["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / base["slug"]
    finally:
        config.select_model(previous)


def test_sft_iffc_v5_model_is_distinct_from_v1_v2_v4_and_base():
    key = "gemma4-e2b-sft-iffc-v5"
    assert key in config.MODELS

    v5 = config.MODELS[key]
    v4 = config.MODELS["gemma4-e2b-sft-iffc-v4"]
    v2 = config.MODELS["gemma4-e2b-sft-iffc-v2"]
    v1 = config.MODELS["gemma4-e2b-sft-iffc"]
    base = config.MODELS["gemma4-e2b"]
    assert v5["api_name"] == "gemma4-e2b-sft-iffc-v5"
    assert v5["display"] == "Gemma-4-E2B-it SFT-IFFC-v5 (Q4_K_M)"
    assert v5["slug"] == "Gemma-4-E2B-it-SFT-IFFC-v5-Q4_K_M"
    assert v5["gguf_path"] == "sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf"
    assert v5["api_name"] != v4["api_name"]
    assert v5["slug"] != v4["slug"]
    assert v5["display"] != v4["display"]
    assert v5["api_name"] != v2["api_name"]
    assert v5["slug"] != v2["slug"]
    assert v5["api_name"] != v1["api_name"]
    assert v5["slug"] != v1["slug"]
    assert v5["api_name"] != base["api_name"]
    assert v5["slug"] != base["slug"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc-v5"
        assert config.MODEL_DISPLAY == "Gemma-4-E2B-it SFT-IFFC-v5 (Q4_K_M)"
        assert config.MODEL_SLUG == "Gemma-4-E2B-it-SFT-IFFC-v5-Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / v5["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v4["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v2["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v1["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / base["slug"]
    finally:
        config.select_model(previous)


def test_sft_iffc_v6_model_is_distinct_from_v1_v2_v4_v5_and_base():
    key = "gemma4-e2b-sft-iffc-v6"
    assert key in config.MODELS

    v6 = config.MODELS[key]
    v5 = config.MODELS["gemma4-e2b-sft-iffc-v5"]
    v4 = config.MODELS["gemma4-e2b-sft-iffc-v4"]
    v2 = config.MODELS["gemma4-e2b-sft-iffc-v2"]
    v1 = config.MODELS["gemma4-e2b-sft-iffc"]
    base = config.MODELS["gemma4-e2b"]
    assert v6["api_name"] == "gemma4-e2b-sft-iffc-v6"
    assert v6["display"] == "Gemma-4-E2B-it SFT-IFFC-v6 (Q4_K_M)"
    assert v6["slug"] == "Gemma-4-E2B-it-SFT-IFFC-v6-Q4_K_M"
    assert v6["gguf_path"] == "sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf"
    assert v6["api_name"] != v5["api_name"]
    assert v6["slug"] != v5["slug"]
    assert v6["display"] != v5["display"]
    assert v6["api_name"] != v4["api_name"]
    assert v6["slug"] != v4["slug"]
    assert v6["api_name"] != v2["api_name"]
    assert v6["slug"] != v2["slug"]
    assert v6["api_name"] != v1["api_name"]
    assert v6["slug"] != v1["slug"]
    assert v6["api_name"] != base["api_name"]
    assert v6["slug"] != base["slug"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc-v6"
        assert config.MODEL_DISPLAY == "Gemma-4-E2B-it SFT-IFFC-v6 (Q4_K_M)"
        assert config.MODEL_SLUG == "Gemma-4-E2B-it-SFT-IFFC-v6-Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / v6["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v5["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v4["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v2["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v1["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / base["slug"]
    finally:
        config.select_model(previous)


def test_sft_iffc_v7_model_is_distinct_from_prior_sft_keys_and_base():
    key = "gemma4-e2b-sft-iffc-v7"
    assert key in config.MODELS

    v7 = config.MODELS[key]
    v6 = config.MODELS["gemma4-e2b-sft-iffc-v6"]
    v5 = config.MODELS["gemma4-e2b-sft-iffc-v5"]
    v4 = config.MODELS["gemma4-e2b-sft-iffc-v4"]
    v2 = config.MODELS["gemma4-e2b-sft-iffc-v2"]
    v1 = config.MODELS["gemma4-e2b-sft-iffc"]
    base = config.MODELS["gemma4-e2b"]
    assert v7["api_name"] == "gemma4-e2b-sft-iffc-v7"
    assert v7["display"] == "Gemma-4-E2B-it SFT-IFFC-v7 (Q4_K_M)"
    assert v7["slug"] == "Gemma-4-E2B-it-SFT-IFFC-v7-Q4_K_M"
    assert v7["gguf_path"] == "sft/fused/gemma-4-e2b-it-iffc-v7-best1800-Q4_K_M.gguf"
    assert v7["api_name"] != v6["api_name"]
    assert v7["slug"] != v6["slug"]
    assert v7["display"] != v6["display"]
    assert v7["api_name"] != v5["api_name"]
    assert v7["slug"] != v5["slug"]
    assert v7["api_name"] != v4["api_name"]
    assert v7["slug"] != v4["slug"]
    assert v7["api_name"] != v2["api_name"]
    assert v7["slug"] != v2["slug"]
    assert v7["api_name"] != v1["api_name"]
    assert v7["slug"] != v1["slug"]
    assert v7["api_name"] != base["api_name"]
    assert v7["slug"] != base["slug"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "gemma4-e2b-sft-iffc-v7"
        assert config.MODEL_DISPLAY == "Gemma-4-E2B-it SFT-IFFC-v7 (Q4_K_M)"
        assert config.MODEL_SLUG == "Gemma-4-E2B-it-SFT-IFFC-v7-Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / v7["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v6["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v5["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v4["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v2["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / v1["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / base["slug"]
    finally:
        config.select_model(previous)


def test_xlam_1b_model_is_distinct_from_other_registry_keys():
    key = "xlam-1b"
    assert key in config.MODELS

    xlam = config.MODELS[key]
    gemma = config.MODELS["gemma4-e2b"]
    fg = config.MODELS["functiongemma-270m"]
    qwen = config.MODELS["qwen2.5-3b"]
    assert xlam["hf_repo"] == "Salesforce/xLAM-1b-fc-r-gguf"
    assert xlam["quant"] == "Q4_K_M"
    assert xlam["api_name"] == "xlam-1b"
    assert xlam["display"] == "xLAM-1b-fc-r (Q4_K_M)"
    assert xlam["slug"] == "xLAM-1b-fc-r-Q4_K_M"
    assert xlam["api_name"] != gemma["api_name"]
    assert xlam["slug"] != gemma["slug"]
    assert xlam["display"] != gemma["display"]
    assert xlam["api_name"] != fg["api_name"]
    assert xlam["slug"] != fg["slug"]
    assert xlam["api_name"] != qwen["api_name"]
    assert xlam["slug"] != qwen["slug"]

    previous = config.DEFAULT_MODEL
    try:
        config.select_model(key)
        assert config.MODEL_NAME == "xlam-1b"
        assert config.MODEL_DISPLAY == "xLAM-1b-fc-r (Q4_K_M)"
        assert config.MODEL_SLUG == "xLAM-1b-fc-r-Q4_K_M"
        assert config.MODEL_HF_REPO == "Salesforce/xLAM-1b-fc-r-gguf"
        assert config.MODEL_QUANT == "Q4_K_M"
        assert config.MODEL_RESULTS_DIR == config.RESULTS_DIR / xlam["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / gemma["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / fg["slug"]
        assert config.MODEL_RESULTS_DIR != config.RESULTS_DIR / qwen["slug"]
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
