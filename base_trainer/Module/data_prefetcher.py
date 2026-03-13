import torch
from time import time as _wall_time
from typing import Optional, Callable, Dict


_EXHAUSTED = object()


class DataPrefetcher:
    """Prefetch next batch with H2D transfer on a dedicated CUDA stream.

    Timing breakdown (available via ``last_timings`` after each ``next()``):
      * **loader_wait** – wall-clock time blocked on the underlying iterator
        (AsyncDataLoader / DataLoader).  Dominated by remote I/O or CPU
        preprocessing when the prefetch queue is drained.
      * **h2d_transfer** – wall-clock time for the synchronous H2D copy
        (measured by recording CUDA events on the side stream and
        synchronizing).
    """

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

        self.last_timings: Dict[str, float] = {}

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
        t0 = _wall_time()
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = _EXHAUSTED
            self.last_timings['loader_wait'] = _wall_time() - t0
            self.last_timings['h2d_transfer'] = 0.0
            return
        loader_wait = _wall_time() - t0

        if self.batch is None:
            self.last_timings['loader_wait'] = loader_wait
            self.last_timings['h2d_transfer'] = 0.0
            return

        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                h2d_t0 = _wall_time()
                self.batch = self._move_to_device(self.batch)
                self.stream.synchronize()
                h2d_time = _wall_time() - h2d_t0

                if self.gpu_preprocess_fn is not None:
                    self.batch = self.gpu_preprocess_fn(self.batch)
        else:
            h2d_time = 0.0
            if self.gpu_preprocess_fn is not None:
                self.batch = self.gpu_preprocess_fn(self.batch)

        self.last_timings['loader_wait'] = loader_wait
        self.last_timings['h2d_transfer'] = h2d_time

    def next(self):
        if self.batch is _EXHAUSTED:
            return None
        if self.use_cuda and self.batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
