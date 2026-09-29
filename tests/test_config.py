from wonca_rag.config import load_experiment_config, load_models_config
from wonca_rag.paths import PATHS


def test_experiment_config_loads():
    config = load_experiment_config()
    assert config.project.name == "WONCA_RAG"
    assert set(config.screening.labels) == {"S", "N", "D"}
    assert config.temporal_rag.enabled is True
    assert config.temporal_rag.allow_future_articles is False
    assert config.chunking.chunk_overlap_tokens < config.chunking.chunk_size_tokens


def test_models_config_loads():
    config = load_models_config()
    assert len(config.models) >= 1

    domains = {model.domain for model in config.models.values()}
    assert {"general", "health"}.issubset(domains)

    ownerships = {model.ownership for model in config.models.values()}
    assert "proprietary" in ownerships
    assert "open_weight" in ownerships


def test_runtime_directories_can_be_created():
    PATHS.ensure_runtime_directories()
    assert (PATHS.data_root / "raw" / "pdf").exists()
    assert (PATHS.artifacts_root / "chunks").exists()
    assert (PATHS.outputs_root / "checkpoints").exists()
