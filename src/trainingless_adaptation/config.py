from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    values: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return Path(self.values["project_root"]).expanduser().resolve()

    @property
    def seed(self) -> int:
        return int(self.values["seed"])

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.project_root / path

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values[name])

    def model(self, name: str) -> dict[str, Any]:
        return dict(self.values["models"][name])

    def output_dir(self, track: str) -> Path:
        return self.resolve(self.values["outputs"]["root"]) / track

    @property
    def manifest_dir(self) -> Path:
        """Return the validation-manifest directory for this configuration.

        The original strict run stored manifests at the project root. Keep
        that legacy location stable while isolating newer output roots.
        """
        output_root = self.resolve(self.values["outputs"]["root"])
        legacy_output_root = self.project_root / "outputs"
        if output_root == legacy_output_root:
            return self.project_root / "manifests"
        return output_root / "manifests"

    def dump(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.values, sort_keys=True), encoding="utf-8")

    def dump_run(self, destination: Path, **execution: Any) -> None:
        payload = copy.deepcopy(self.values)
        payload["execution"] = execution
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    required = {"project_root", "data", "vendor", "weights", "outputs", "corruption", "models"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    return ExperimentConfig(config_path, values)
