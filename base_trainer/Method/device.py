import torch
from typing import Union


def moveTo(
    data: Union[dict, torch.Tensor, list, tuple, float, str],
    device: Union[torch.device, str] = "cpu",
) -> Union[dict, torch.Tensor, list, tuple, float, str]:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: moveTo(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [moveTo(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(moveTo(item, device) for item in data)
    else:
        return data
