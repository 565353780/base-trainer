from time import time, sleep
from typing import Optional

import torch


class Timer(object):
    def __init__(self) -> None:
        self.start_times: dict[str, float | None] = {}
        self.time_sums: dict[str, float] = {}
        self.step_counts: dict[str, int] = {}
        self.last_durations: dict[str, float] = {}

        self._cuda_start_events: dict[str, torch.cuda.Event] = {}
        self._cuda_end_events: dict[str, torch.cuda.Event] = {}
        self._cuda_pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        return

    def start(self, name: str) -> bool:
        if name in self.start_times and self.start_times[name] is not None:
            return True

        self.start_times[name] = time()
        if name not in self.time_sums:
            self.time_sums[name] = 0.0
            self.step_counts[name] = 0
            self.last_durations[name] = 0.0
        return True

    def pause(self, name: str) -> bool:
        if name not in self.start_times or self.start_times[name] is None:
            return True

        duration = time() - self.start_times[name]
        self.time_sums[name] += duration
        self.last_durations[name] = duration
        self.start_times[name] = None
        self.step_counts[name] += 1
        return True

    def startCuda(self, name: str, stream: Optional[torch.cuda.Stream] = None) -> bool:
        """Record a CUDA event at the current point on *stream* (default: current stream).
        Does NOT call torch.cuda.synchronize(), so the GPU pipeline is not stalled."""
        evt = torch.cuda.Event(enable_timing=True)
        if stream is not None:
            evt.record(stream)
        else:
            evt.record()
        self._cuda_start_events[name] = evt
        if name not in self.time_sums:
            self.time_sums[name] = 0.0
            self.step_counts[name] = 0
            self.last_durations[name] = 0.0
        return True

    def pauseCuda(self, name: str, stream: Optional[torch.cuda.Stream] = None) -> bool:
        """Record an end event and stash the (start, end) pair for later collection."""
        if name not in self._cuda_start_events:
            return True
        end_evt = torch.cuda.Event(enable_timing=True)
        if stream is not None:
            end_evt.record(stream)
        else:
            end_evt.record()
        start_evt = self._cuda_start_events.pop(name)
        self._cuda_pending.setdefault(name, []).append((start_evt, end_evt))
        return True

    def collectCudaTimes(self) -> None:
        """Call once per step **after** torch.cuda.synchronize() to harvest all
        pending CUDA event pairs into the normal time_sums / last_durations."""
        for name, pairs in self._cuda_pending.items():
            for start_evt, end_evt in pairs:
                duration = start_evt.elapsed_time(end_evt) / 1000.0
                self.time_sums[name] += duration
                self.last_durations[name] = duration
                self.step_counts[name] = self.step_counts.get(name, 0) + 1
        self._cuda_pending.clear()

    def addTime(self, name: str, duration: float) -> bool:
        """Directly add a measured duration without using start/pause."""
        if name not in self.time_sums:
            self.time_sums[name] = 0.0
            self.step_counts[name] = 0
            self.last_durations[name] = 0.0
            self.start_times[name] = None
        self.time_sums[name] += duration
        self.last_durations[name] = duration
        self.step_counts[name] += 1
        return True

    def totalTime(self, name: str) -> float:
        if name not in self.time_sums:
            return 0.0

        total = self.time_sums[name]
        if self.start_times.get(name) is not None:
            total += time() - self.start_times[name]
        return total

    def stepCount(self, name: str) -> int:
        return self.step_counts.get(name, 0)

    def avgTime(self, name: str) -> float:
        count = self.stepCount(name)
        if count == 0:
            return 0.0
        return self.totalTime(name) / count

    def lastTime(self, name: str) -> float:
        return self.last_durations.get(name, 0.0)

    def reset(self, name: str | None = None) -> bool:
        if name is None:
            self.start_times.clear()
            self.time_sums.clear()
            self.step_counts.clear()
            self.last_durations.clear()
            self._cuda_start_events.clear()
            self._cuda_end_events.clear()
            self._cuda_pending.clear()
            return True

        self.start_times.pop(name, None)
        self.time_sums.pop(name, None)
        self.step_counts.pop(name, None)
        self.last_durations.pop(name, None)
        self._cuda_start_events.pop(name, None)
        self._cuda_end_events.pop(name, None)
        self._cuda_pending.pop(name, None)
        return True

    def toDict(self) -> dict[str, float]:
        result = {}
        for name in self.time_sums:
            result[name] = self.totalTime(name)
        return result

    def sleep(self, second: float) -> bool:
        sleep(second)
        return True
