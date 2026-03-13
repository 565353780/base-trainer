import torch
from typing import Optional, Callable, Dict


_EXHAUSTED = object()


class DataPrefetcher:
    """Prefetch next batch with H2D transfer on a dedicated CUDA stream."""

    def __init__(
        self,
        loader,
        device: str = "cpu",
        gpu_preprocess_fn: Optional[Callable[[Dict], Optional[Dict]]] = None,
    ):
        self.loader = iter(loader)
        self.device = torch.device(device)
        self.use_cuda = self.device.type != "cpu"
        self.gpu_preprocess_fn = gpu_preprocess_fn
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        self.batch = _EXHAUSTED

        self.preload()

    @property
    def exhausted(self) -> bool:
        return self.batch is _EXHAUSTED

    def _move_to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_device(v) for v in data)
        return data

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = _EXHAUSTED
            return

        if self.batch is None:
            return

        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.batch = self._move_to_device(self.batch)

                if self.gpu_preprocess_fn is not None:
                    self.batch = self.gpu_preprocess_fn(self.batch)
        else:
            if self.gpu_preprocess_fn is not None:
                self.batch = self.gpu_preprocess_fn(self.batch)

    def next(self):
        if self.batch is _EXHAUSTED:
            return None
        if self.use_cuda and self.batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
