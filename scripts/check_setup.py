from wonca_rag.config import load_experiment_config, load_models_config
from wonca_rag.paths import PATHS


def main() -> None:
    PATHS.ensure_runtime_directories()
    experiment = load_experiment_config()
    models = load_models_config()

    print("WONCA RAG setup: OK")
    print(f"Project root: {PATHS.project_root}")
    print(f"Project name: {experiment.project.name}")
    print(f"Pipeline version: {experiment.pipeline.pipeline_version}")
    print(
        "Temporal future retrieval allowed:",
        experiment.temporal_rag.allow_future_articles,
    )
    print(f"Models registered: {len(models.models)}")
    print("Models enabled:", sum(m.enabled for m in models.models.values()))


if __name__ == "__main__":
    main()
