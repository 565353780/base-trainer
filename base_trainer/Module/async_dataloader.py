from collections import deque
from concurrent.futures import ThreadPoolExecutor


class AsyncDataLoader:
    def __init__(self, dataloader, preprocess_fn, max_workers=2, prefetch_depth=4):
        self.dataloader = dataloader
        self.preprocess_fn = preprocess_fn
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prefetch_depth = prefetch_depth
        self.futures: deque = deque()
        return

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.futures.clear()
        for _ in range(self.prefetch_depth):
            if not self._schedule_next():
                break
        return self

    def _schedule_next(self) -> bool:
        try:
            data = next(self.iterator)
            self.futures.append(self.executor.submit(self.preprocess_fn, data))
            return True
        except StopIteration:
            return False

    def __next__(self):
        if not self.futures:
            raise StopIteration
        result = self.futures.popleft().result()
        self._schedule_next()
        return result

    def __len__(self):
        return len(self.dataloader)

    def close(self):
        self.executor.shutdown(wait=True)
