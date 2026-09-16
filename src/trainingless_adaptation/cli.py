from __future__ import annotations

import argparse
import json

from .assets import prepare_assets
from .config import load_config
from .experiments import run_figure7, run_sensitivity, run_table1, train_beats, validate_clean
from .reporting import build_report


def add_execution_arguments(parser: argparse.ArgumentParser, include_track: bool = True) -> None:
    parser.add_argument("--config", default="configs/strict.yaml")
    if include_track:
        parser.add_argument("--track", choices=("strict", "aligned"), default="strict")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="trainingless")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/strict.yaml")
    add_execution_arguments(commands.add_parser("train-beats"), include_track=False)
    add_execution_arguments(commands.add_parser("validate-clean"), include_track=False)
    add_execution_arguments(commands.add_parser("run-table1"))
    add_execution_arguments(commands.add_parser("run-figure7"))
    add_execution_arguments(commands.add_parser("run-sensitivity"), include_track=False)
    report = commands.add_parser("build-report")
    report.add_argument("--config", default="configs/strict.yaml")
    report.add_argument("--track", choices=("strict", "aligned"), default="strict")
    return root


def main() -> None:
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    if arguments.command == "prepare":
        prepare_assets(config)
    elif arguments.command == "train-beats":
        train_beats(config, arguments.device, arguments.resume)
    elif arguments.command == "validate-clean":
        validate_clean(config, arguments.device, arguments.resume)
    elif arguments.command == "run-table1":
        run_table1(config, arguments.track, arguments.device, arguments.resume)
    elif arguments.command == "run-figure7":
        run_figure7(config, arguments.track, arguments.device, arguments.resume)
    elif arguments.command == "run-sensitivity":
        result = run_sensitivity(config, arguments.device, arguments.resume)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif arguments.command == "build-report":
        result = build_report(config, arguments.track)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        raise RuntimeError(f"Unhandled command: {arguments.command}")
