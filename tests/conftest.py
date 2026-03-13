import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import config


@pytest.fixture(autouse=True)
def isolated_config_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    results_dir = tmp_path / "results"
    responses_dir = results_dir / "responses"
    metrics_dir = results_dir / "metrics"
    scores_dir = results_dir / "scores"
    figures_dir = results_dir / "figures"

    for path in (results_dir, responses_dir, metrics_dir, scores_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "MODEL_NAME", "test-model")
    monkeypatch.setattr(config, "MODEL_DISPLAY", "Test Model")
    monkeypatch.setattr(config, "MODEL_SLUG", "Test-Model")
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "MODEL_RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "RESPONSES_DIR", responses_dir)
    monkeypatch.setattr(config, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(config, "SCORES_DIR", scores_dir)
    monkeypatch.setattr(config, "FIGURES_DIR", figures_dir)
    return tmp_path


@pytest.fixture
def bfcl_fixture_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "bfcl_data"
    answers_dir = data_dir / "possible_answer"
    func_doc_dir = data_dir / "multi_turn_func_doc"
    data_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    func_doc_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "BFCL_DATA_DIR", data_dir)
    monkeypatch.setattr(config, "BFCL_POSSIBLE_ANSWER_DIR", answers_dir)
    monkeypatch.setattr(config, "MULTI_TURN_FUNC_DOC_DIR", func_doc_dir)
    return data_dir, answers_dir
