import torch
from typing import Union


def moveTo(
    data: Union[dict, torch.Tensor, list, tuple, float, str],
    device: Union[torch.device, str] = "cpu",
    non_blocking: bool = False,
) -> Union[dict, torch.Tensor, list, tuple, float, str]:
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {key: moveTo(value, device, non_blocking) for key, value in data.items()}
    elif isinstance(data, list):
        return [moveTo(item, device, non_blocking) for item in data]
    elif isinstance(data, tuple):
        return tuple(moveTo(item, device, non_blocking) for item in data)
    else:
        return data
