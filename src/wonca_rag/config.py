from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from wonca_rag.paths import PATHS


class ScreeningConfig(BaseModel):
    labels: list[str]
    dimensions: list[str]
    fulltext_review_if_d: bool = True


class TemporalRAGConfig(BaseModel):
    enabled: bool = True
    allow_future_articles: bool = False


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = Field(gt=0)
    chunk_overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


class RetrievalConfig(BaseModel):
    method: str
    semantic_top_k: int = Field(gt=0)
    bm25_top_k: int = Field(gt=0)
    final_top_k: int = Field(gt=0)


class OCRConfig(BaseModel):
    enabled: bool = True
    engine: str = "tesseract"
    trigger: str = "zero_text_only"
    language: str = "eng"
    dpi: int = Field(default=300, ge=150, le=600)
    psm: int = Field(default=3, ge=0, le=13)
    min_characters_after_ocr: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_ocr(self) -> "OCRConfig":
        if self.engine != "tesseract":
            raise ValueError("Only tesseract is supported in the current OCR stage.")
        if self.trigger not in {"zero_text_only", "likely_scanned"}:
            raise ValueError(
                "ocr.trigger must be 'zero_text_only' or 'likely_scanned'"
            )
        return self


class ModelRuntimeConfig(BaseModel):
    temperature: float = Field(ge=0)
    max_output_tokens: int = Field(gt=0)
    save_raw_response: bool = True
    save_evidence: bool = True
    save_token_usage: bool = True
    save_latency: bool = True


class PipelineConfig(BaseModel):
    resume: bool = True
    pipeline_version: str
    prompt_version: str


class ProjectConfig(BaseModel):
    name: str
    experiment_name: str
    random_seed: int


class ExperimentConfig(BaseModel):
    project: ProjectConfig
    screening: ScreeningConfig
    temporal_rag: TemporalRAGConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    ocr: OCRConfig
    model_runtime: ModelRuntimeConfig
    pipeline: PipelineConfig

    @model_validator(mode="after")
    def validate_methodology(self) -> "ExperimentConfig":
        if set(self.screening.labels) != {"S", "N", "D"}:
            raise ValueError("screening.labels must contain exactly S, N and D")
        if self.temporal_rag.allow_future_articles:
            raise ValueError(
                "allow_future_articles must remain false for the temporal design"
            )
        return self


class ModelDefinition(BaseModel):
    display_name: str
    family: str
    ownership: str
    size_category: str
    domain: str
    provider: str
    infrastructure: str
    interface: str
    parameters_b: float | int | None = None
    enabled: bool = False
    runtime: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    models: dict[str, ModelDefinition]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    config_path = path or PATHS.config_root / "experiment.yaml"
    return ExperimentConfig.model_validate(load_yaml(config_path))


def load_models_config(path: Path | None = None) -> ModelsConfig:
    config_path = path or PATHS.config_root / "models.yaml"
    return ModelsConfig.model_validate(load_yaml(config_path))
