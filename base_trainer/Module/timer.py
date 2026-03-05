from time import time, sleep


class Timer(object):
    def __init__(self) -> None:
        self.start_times: dict[str, float | None] = {}
        self.time_sums: dict[str, float] = {}
        self.step_counts: dict[str, int] = {}
        self.last_durations: dict[str, float] = {}
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
            return True

        self.start_times.pop(name, None)
        self.time_sums.pop(name, None)
        self.step_counts.pop(name, None)
        self.last_durations.pop(name, None)
        return True

    def toDict(self) -> dict[str, float]:
        result = {}
        for name in self.time_sums:
            result[name] = self.totalTime(name)
        return result

    def sleep(self, second: float) -> bool:
        sleep(second)
        return True
