from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    artifacts_root: Path
    outputs_root: Path
    config_root: Path
    notebooks_root: Path
    tests_root: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        project_root = Path(
            os.getenv("WONCA_RAG_PROJECT_ROOT", _default_project_root())
        ).resolve()

        data_root = Path(
            os.getenv("WONCA_RAG_DATA_ROOT", project_root / "data")
        ).resolve()

        artifacts_root = Path(
            os.getenv("WONCA_RAG_ARTIFACTS_ROOT", project_root / "artifacts")
        ).resolve()

        outputs_root = Path(
            os.getenv("WONCA_RAG_OUTPUTS_ROOT", project_root / "outputs")
        ).resolve()

        return cls(
            project_root=project_root,
            data_root=data_root,
            artifacts_root=artifacts_root,
            outputs_root=outputs_root,
            config_root=project_root / "config",
            notebooks_root=project_root / "notebooks",
            tests_root=project_root / "tests",
        )

    def ensure_runtime_directories(self) -> None:
        directories = [
            self.data_root / "raw" / "pdf",
            self.data_root / "human",
            self.data_root / "metadata",
            self.data_root / "processed",
            self.artifacts_root / "chunks",
            self.artifacts_root / "embeddings",
            self.artifacts_root / "indexes",
            self.artifacts_root / "ledgers",
            self.outputs_root / "predictions",
            self.outputs_root / "evidence",
            self.outputs_root / "raw_responses",
            self.outputs_root / "checkpoints",
            self.outputs_root / "costs",
            self.outputs_root / "logs",
            self.outputs_root / "validation",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths.from_environment()
