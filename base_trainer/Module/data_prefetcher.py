import torch

class DataPrefetcher:
    def __init__(self, loader, device: str = "cpu"):
        self.loader = iter(loader)
        self.device = torch.device(device)
        self.use_cuda = self.device.type != "cpu"
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        self.batch = None
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

        # GPU 模式：异步预取
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                for k in self.batch:
                    if isinstance(self.batch[k], torch.Tensor):
                        self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)
        # CPU 模式：直接加载（同步）
        else:
            for k in self.batch:
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(device=self.device)

    def next(self):
        if self.batch is None:
            return None
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
