from concurrent.futures import ThreadPoolExecutor


class AsyncDataLoader:
    def __init__(self, dataloader, preprocess_fn, max_workers=2):
        self.dataloader = dataloader
        self.preprocess_fn = preprocess_fn
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.future = None
        return

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self._schedule_next()
        return self

    def _schedule_next(self):
        try:
            data = next(self.iterator)
            self.future = self.executor.submit(self.preprocess_fn, data)
        except StopIteration:
            self.future = None

    def __next__(self):
        if self.future is None:
            raise StopIteration
        result = self.future.result()
        self._schedule_next()
        return result

    def close(self):
        self.executor.shutdown(wait=True)
