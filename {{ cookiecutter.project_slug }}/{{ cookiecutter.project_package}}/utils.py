import os
import random
from typing import Iterable

import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
from torch.nn.utils.rnn import pad_sequence


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in tensors.items()}


def truncate(tensors: list[torch.Tensor], max_length: int):
    truncated = []
    for tensor in tensors:
        needs_to_be_truncated = len(tensor) > max_length

        if needs_to_be_truncated:
            tensor = tensor[:max_length]

        truncated.append(tensor)

    return truncated


def attention_mask_from_input_lengths(
    input_lengths: Iterable[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    attention_mask = pad_sequence(
        [torch.ones(length, dtype=torch.int64) for length in input_lengths],
        batch_first=True,
    )

    if device is not None:
        attention_mask = attention_mask.to(device)

    return attention_mask
