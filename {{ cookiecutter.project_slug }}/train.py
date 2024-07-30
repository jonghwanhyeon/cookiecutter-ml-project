import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tyro
from rich.console import Console

from {{ cookiecutter.project_package }}.configurations import TrainingConfig
from {{ cookiecutter.project_package }}.utils import set_random_seed


@dataclass
class Config:
    name: str
    training: TrainingConfig


def main():
    console = Console()

    config: Config = tyro.cli(Config)
    console.print(config)

    current_path = Path.cwd().absolute()
    experiments_path = current_path / "experiments"
    experiment_path = experiments_path / config.name

    if experiment_path.exists():
        if not config.training.overwrite_output_path:
            console.print(f"Output path ({experiment_path}) already exists")
            sys.exit(1)

        shutil.rmtree(experiment_path)
        experiment_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_random_seed(config.training.seed)


if __name__ == "__main__":
    main()
