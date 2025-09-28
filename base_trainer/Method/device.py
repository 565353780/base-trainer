import torch
from typing import Union


def moveTo(
    data: Union[dict, torch.Tensor, list, tuple, float, str],
    device: Union[torch.device, str] = "cpu",
    dtype=torch.float32,
) -> Union[dict, torch.Tensor, list, tuple, float, str]:
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
            return data.to(device, dtype=dtype)
        return data.to(device)
    elif isinstance(data, dict):
        return {key: moveTo(value, device, dtype) for key, value in data.items()}
    elif isinstance(data, list):
        return [moveTo(item, device, dtype) for item in data]
    elif isinstance(data, tuple):
        return tuple(moveTo(item, device, dtype) for item in data)
    else:
        return data
