from dataclasses import dataclass


@dataclass
class TrainingConfig:
    overwrite_output_path: bool = True

    seed: int = 4909
