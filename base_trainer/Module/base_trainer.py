import os
import torch
import threading
import numpy as np
import torch.distributed as dist

from torch import nn
from tqdm import tqdm
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, List
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.utils.data import DataLoader, DistributedSampler

from base_trainer.Method.device import moveTo
from base_trainer.Method.time import getCurrentTime
from base_trainer.Method.fsdp import default_fsdp_shard_fn
from base_trainer.Method.path import createFileFolder, renameFile, removeFile
from base_trainer.Module.data_prefetcher import DataPrefetcher
from base_trainer.Module.async_dataloader import AsyncDataLoader
from base_trainer.Module.logger import Logger
from base_trainer.Module.timer import Timer

try:
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()
except ImportError:
    print('[WARN][BaseTrainer::import]')
    print('\t allow_ops_in_compiled_graph failed!')
    pass


def setup_distributed(backend: Optional[str] = None):
    """
    初始化分布式环境，自动兼容 CPU / GPU 训练。
    返回: (node_rank, local_rank, device)
    """

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if "SLURM_NODEID" in os.environ:
        node_rank = int(os.environ["SLURM_NODEID"])
    else:
        node_rank = int(os.environ.get("RANK", 0))

    return node_rank, local_rank, device

def check_and_replace_nan_in_grad(model):
    """Check for NaN/Inf gradients and ensure ALL ranks agree on the result.

    Each rank computes a local flag from its own FSDP shard gradients, then
    an all-reduce ensures every rank takes the same branch (skip or update).
    Without this synchronization, some ranks may skip optim.step while others
    proceed, causing FSDP weight divergence and eventual NCCL timeout.

    Returns True if gradients are clean on ALL ranks, False otherwise.
    """
    has_nan = torch.zeros(1, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                has_nan.fill_(1.0)
                break

    if dist.is_initialized():
        dist.all_reduce(has_nan, op=dist.ReduceOp.MAX)

    if has_nan.item() > 0:
        print(f"[WARN] NaN/Inf detected in gradient (rank {dist.get_rank() if dist.is_initialized() else 0}), zeroing all grads.")
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        return False
    return True


def _check_and_abort(success: bool, msg: str = "") -> None:
    """Collective check: if ANY rank reports failure, ALL ranks abort.

    Uses all_reduce(MIN) so that if any rank sets success=False the
    reduced value is 0 on every rank. All ranks then tear down the
    process group and exit together — no NCCL hang.
    """
    if dist.is_initialized():
        flag = torch.tensor([1 if success else 0], dtype=torch.int32)
        if torch.cuda.is_available():
            flag = flag.to(torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"))
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        success = flag.item() > 0

    if not success:
        if msg:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[ABORT][rank {rank}] {msg}")
        if dist.is_initialized():
            dist.destroy_process_group()
        exit(1)


class BaseTrainer(ABC):
    def __init__(
        self,
        batch_size: int = 32,
        accum_iter: int = 1,
        num_workers: int = 16,
        model_file_path: Optional[str] = None,
        weights_only: bool = False,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Optional[str] = None,
        save_log_folder_path: Optional[str] = None,
        best_model_metric_name: Optional[str] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        quick_test: bool = False,
        save_checkpoint_freq: int = -1,
        prefetch_factor: int = 4,
        fsdp_shard_fn: Optional[Callable] = default_fsdp_shard_fn,
        compile_fn: Optional[Callable] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32),
        load_model_fn: Optional[Callable]=None,
        save_model_fn: Optional[Callable]=None,
    ) -> None:
        self.backend = "nccl" if torch.cuda.is_available() else "gloo"
        self.node_rank, self.local_rank, self.device = setup_distributed(self.backend)
        self.is_logger = (self.node_rank == 0) and (self.local_rank == 0)

        self.batch_size = batch_size
        self.accum_iter = accum_iter
        self.num_workers = num_workers

        if torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        self.warm_step_num = warm_step_num / accum_iter
        self.finetune_step_num = finetune_step_num

        if lr_batch_size > 0:
            self.lr = (
                lr * batch_size / lr_batch_size * self.accum_iter * dist.get_world_size()
            )
        else:
            self.lr = lr

        self.ema_start_step = ema_start_step
        self.ema_decay_init = ema_decay_init
        self.ema_decay = ema_decay

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.best_model_metric_name = best_model_metric_name
        self.is_metric_lower_better = is_metric_lower_better
        self.sample_results_freq = sample_results_freq
        self.quick_test = quick_test

        self.fsdp_shard_fn = fsdp_shard_fn
        self.compile_fn = compile_fn
        if mp_policy is None:
            if torch.cuda.is_bf16_supported():
                mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            else:
                mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
        self.mp_policy = mp_policy
        self.load_model_fn = load_model_fn
        self.save_model_fn = save_model_fn

        self.save_checkpoint_freq = save_checkpoint_freq
        self.prefetch_factor = prefetch_factor

        self.step = 0
        self.epoch = 0
        self.loss_dict_list = []

        self.loss_min = float("inf")

        self.timer = None
        self.logger = None
        if self.is_logger:
            self.timer = Timer()
            self.logger = Logger()

        self.dataloader_dict = {}
        dataset_ok = self.createDatasets()
        if not dataset_ok:
            print("[ERROR][BaseTrainer::__init__]")
            print("\t createDatasets failed!")
        _check_and_abort(dataset_ok, "createDatasets failed")

        for key, item in self.dataloader_dict.items():
            collate_fn = item.get("collate_fn", None)

            if key == "eval":
                self.dataloader_dict[key]["dataloader"] = DataLoader(
                    dataset=item["dataset"],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    persistent_workers=num_workers > 0,
                    collate_fn=collate_fn,
                )
                continue

            self.dataloader_dict[key]["sampler"] = DistributedSampler(item["dataset"])
            self.dataloader_dict[key]["dataloader"] = DataLoader(
                dataset=item["dataset"],
                sampler=self.dataloader_dict[key]["sampler"],
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn,
            )

        self.model: nn.Module
        model_ok = self.createModel()
        if not model_ok and self.is_logger:
            print("[ERROR][BaseTrainer::__init__]")
            print("\t createModel failed!")
        _check_and_abort(model_ok, "createModel failed")

        self.ema_loss = None

        if self.compile_fn is not None:
            self.compile_fn(self.model)

        device_type = "cuda" if self.backend == "nccl" else "cpu"
        world_size = dist.get_world_size()
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        num_nodes = world_size // local_world_size

        if num_nodes > 1 and world_size % local_world_size == 0:
            self.device_mesh = init_device_mesh(
                device_type,
                (num_nodes, local_world_size),
                mesh_dim_names=("replicate", "shard"),
            )
            if self.is_logger:
                print("[INFO][BaseTrainer::__init__]")
                print(f"\t Using 2D HSDP mesh: {num_nodes} nodes x {local_world_size} GPUs/node")
        else:
            self.device_mesh = init_device_mesh(device_type, (world_size,))
            if self.is_logger:
                print("[INFO][BaseTrainer::__init__]")
                print(f"\t Using 1D FSDP mesh: {world_size} GPUs")

        if self.fsdp_shard_fn is not None:
            self.fsdp_shard_fn(self.model, self.device_mesh, self.mp_policy)
        fully_shard(self.model, mesh=self.device_mesh, mp_policy=self.mp_policy)

        if model_file_path is not None:
            load_ok = self.loadModel(model_file_path, weights_only)
            if not load_ok and self.is_logger:
                print("[ERROR][BaseTrainer::__init__]")
                print("\t loadModel failed!")
            _check_and_abort(load_ok, "loadModel failed")

        self._init_ema_shards()

        self.optim = AdamW(self.model.parameters(), lr=self.lr)
        self.sched = LambdaLR(self.optim, lr_lambda=self.warmup_lr)

        self.initRecords()
        return

    @abstractmethod
    def createDatasets(self) -> bool:
        """
        self.dataloader_dict = {
            'loader_name': {
                'dataset': None,
                'repeat_num': 1,
            },
            'eval': {
                'dataset': None,
            },
        }
        """
        pass

    @abstractmethod
    def createModel(self) -> bool:
        """
        self.model = None
        """
        pass

    def loadModel(self, model_file_path: str, weights_only: bool = False) -> bool:
        if self.load_model_fn is not None:
            if self.is_logger:
                self.load_model_fn(model_file_path)
            if dist.is_initialized():
                dist.barrier()

        if not os.path.exists(model_file_path):
            if self.is_logger:
                print("[ERROR][BaseTrainer::loadModel]")
                print("\t model file not exist!")
                print("\t model_file_path:", model_file_path)
            return False

        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if is_rank0:
            model_state_dict = torch.load(
                model_file_path, map_location="cpu", weights_only=False, mmap=True,
            )
        else:
            model_state_dict = {}

        has_model_key = False
        if is_rank0:
            has_model_key = "model" in model_state_dict.keys()
            model_sd = model_state_dict.get("model", {})
        else:
            model_sd = {}

        flag = [has_model_key]
        if dist.is_initialized():
            dist.broadcast_object_list(flag, src=0)
        has_model_key = flag[0]

        if has_model_key:
            try:
                set_model_state_dict(
                    self.model,
                    model_state_dict=model_sd,
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
                    ),
                )
            except Exception as e:
                if self.is_logger:
                    print("[WARN][BaseTrainer::loadModel]")
                    print(
                        "\t model state dict not fully match current model! will train from scratch!"
                    )
                    print("\t  Exception:")
                    print("\t", e)

        metadata = [None, None, None]
        if is_rank0:
            if not weights_only:
                if "step" in model_state_dict.keys():
                    self.step = model_state_dict["step"]
                    metadata[0] = self.step

            if not weights_only:
                if self.is_logger:
                    if "ema_loss" in model_state_dict.keys():
                        self.ema_loss = model_state_dict["ema_loss"]
                        metadata[1] = self.ema_loss
                if "loss_min" in model_state_dict.keys():
                    self.loss_min = model_state_dict["loss_min"]
                    metadata[2] = self.loss_min

        if self.is_logger:
            if "ema_model" in model_state_dict.keys():
                self._pending_ema_state_dict = model_state_dict["ema_model"]
                self._has_pending_ema = True

        if dist.is_initialized():
            dist.broadcast_object_list(metadata, src=0)
            if not is_rank0:
                if metadata[0] is not None:
                    self.step = metadata[0]
                if metadata[1] is not None and self.is_logger:
                    self.ema_loss = metadata[1]
                if metadata[2] is not None:
                    self.loss_min = metadata[2]

        if self.is_logger:
            print("[INFO][BaseTrainer::loadModel]")
            print("\t model loaded from:", model_file_path)

        #if self.is_logger and self.load_model_fn is not None:
        #    removeFile(model_file_path)
        return True

    def initRecords(self) -> bool:
        if self.logger is None:
            return True

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        # 若路径中不包含 timestamp（格式见 Method/time.py），则追加 current_time + "/"
        if self.save_result_folder_path is not None:
            if os.path.exists(self.save_result_folder_path):
                self.save_result_folder_path = self.save_result_folder_path + current_time + "/"
        if self.save_log_folder_path is not None:
            if os.path.exists(self.save_log_folder_path):
                self.save_log_folder_path = self.save_log_folder_path + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def getModelSize(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def getLr(self) -> float:
        return self.optim.state_dict()["param_groups"][0]["lr"]

    def warmup_lr(self, step: int) -> float:
        if self.warm_step_num == 0:
            return 1.0

        return min(step, self.warm_step_num) / self.warm_step_num

    def _init_ema_shards(self) -> None:
        """Snapshot every FSDP-sharded parameter to a CPU clone.

        Called after FSDP wrapping (and optional checkpoint loading) so that
        each rank holds only its own shard — zero extra GPU memory.

        If a full EMA state dict was cached from a checkpoint, we temporarily
        load it into the FSDP model to obtain the correct shard layout, copy
        the shards out, then restore the original training weights.
        """
        has_pending = getattr(self, "_has_pending_ema", False)
        if dist.is_initialized():
            flag_list = [has_pending]
            dist.broadcast_object_list(flag_list, src=0)
            has_pending = flag_list[0]

        if has_pending:
            pending_sd = getattr(self, "_pending_ema_state_dict", None)
            if pending_sd is None:
                pending_sd = {}

            orig_shards = [p.data.detach().clone() for p in self.model.parameters()]

            set_model_state_dict(
                self.model,
                model_state_dict=pending_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )

            self._ema_params: List[torch.Tensor] = []
            for p in self.model.parameters():
                self._ema_params.append(p.data.detach().float().cpu().clone())

            for p, orig in zip(self.model.parameters(), orig_shards):
                p.data.copy_(orig)
            del orig_shards

            if hasattr(self, "_pending_ema_state_dict"):
                del self._pending_ema_state_dict
            self._has_pending_ema = False
        else:
            self._ema_params: List[torch.Tensor] = []
            for p in self.model.parameters():
                self._ema_params.append(p.data.detach().float().cpu().clone())

    def toEMADecay(self) -> float:
        if self.step <= self.ema_start_step:
            return self.ema_decay_init + self.step / self.ema_start_step * (
                self.ema_decay - self.ema_decay_init
            )

        return self.ema_decay

    @torch.no_grad()
    def updateEMA(self) -> bool:
        """Update EMA directly on flat FSDP shards (CPU), no all-gather."""
        beta = self.toEMADecay()
        for p, p_ema in zip(self.model.parameters(), self._ema_params):
            if not p.requires_grad:
                continue
            p_ema.mul_(beta).add_(p.data.detach().float().cpu(), alpha=1 - beta)
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = True) -> Optional[Dict]:
        """
        if is_training:
            data_dict['new_name'] = new_value
            return data_dict
        """
        return data_dict

    def preProcessDataWithGPU(self, data_dict: dict, is_training: bool = True) -> Optional[Dict]:
        """
        if is_training:
            data_dict['new_name'] = new_value
            return data_dict
        """
        return data_dict

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return

    def _measure_serial_time(self, name: str, fn: Callable):
        self._sync_device()
        if self.is_logger:
            self.timer.start(name)
        result = fn()
        self._sync_device()
        if self.is_logger:
            self.timer.pause(name)
        return result

    @abstractmethod
    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        """
        loss_dict = {
            'loss': 0.0,
        }
        return loss_dict
        """
        pass

    def trainStep(self, data_dict: dict) -> dict:
        self.model.train()

        #data_dict = moveTo(data_dict, self.device)

        is_accumulating = self.step % self.accum_iter != 0
        self.model.set_requires_gradient_sync(not is_accumulating)

        result_dict = self._measure_serial_time(
            'forward',
            lambda: self.model(data_dict),
        )

        loss_dict = self._measure_serial_time(
            'loss',
            lambda: self.getLossDict(
                data_dict,
                self.postProcessData(data_dict, result_dict, True),
            ),
        )

        if "Loss" not in loss_dict:
            raise RuntimeError("Loss not found in loss_dict")

        loss = loss_dict["Loss"]
        accum_loss = loss / self.accum_iter

        self._measure_serial_time(
            'backward',
            lambda: accum_loss.backward(),
        )

        grad_is_finite = self._measure_serial_time(
            'grad_check',
            lambda: check_and_replace_nan_in_grad(self.model),
        )
        if not grad_is_finite:
            print(f"[WARN] step {self.step}: grad NaN detected, skipping update.")
            self.optim.zero_grad(set_to_none=True)
            return {}

        if not is_accumulating:
            def _optimizer_step():
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                self.sched.step()
                self.updateEMA()
                self.optim.zero_grad(set_to_none=True)
                return True

            self._measure_serial_time('optimizer', _optimizer_step)

        loss_item_dict = {}
        for key, item in loss_dict.items():
            if isinstance(item, torch.Tensor):
                loss_item_dict[key] = item.detach().cpu().float().numpy()
            elif not isinstance(item, str):
                loss_item_dict[key] = item

        return loss_item_dict

    def trainEpoch(self, data_name: str) -> bool:
        if data_name not in self.dataloader_dict.keys():
            print("[ERROR][BaseTrainer::trainEpoch]")
            print("\t data not exist!")
            print("\t data_name:", data_name)
            return False

        dataloader_dict = self.dataloader_dict[data_name]
        dataloader_dict["sampler"].set_epoch(self.epoch)

        dataloader = dataloader_dict["dataloader"]

        async_dataloader = AsyncDataLoader(
            dataloader,
            partial(self.preProcessData, is_training=True),
            max_workers=self.num_workers,
            prefetch_depth=self.prefetch_factor,
        )

        gpu_preprocess_fn = partial(self.preProcessDataWithGPU, is_training=True)
        data_prefetcher = DataPrefetcher(async_dataloader, self.device)

        if self.is_logger:
            pbar = tqdm(total=len(dataloader))
            self.timer.start('step')
        while not data_prefetcher.exhausted:
            if self.is_logger:
                self.timer.start('data_prefetch')
            try:
                data_dict = data_prefetcher.next()
            except:
                print("[WARN][BaseTrainer::trainEpoch]")
                print(
                    "\t call next for DataPrefetcher failed! will early stop this training epoch!"
                )
                break
            if self.is_logger:
                self.timer.pause('data_prefetch')
                for _tk, _tv in data_prefetcher.last_timings.items():
                    self.timer.addTime(_tk, _tv)

            if data_dict is None:
                if self.is_logger:
                    pbar.update(1)
                continue

            data_dict = self._measure_serial_time(
                'gpu_preprocess',
                lambda: gpu_preprocess_fn(data_dict),
            )

            if data_dict is None:
                if self.is_logger:
                    pbar.update(1)
                continue

            if self.is_logger:
                for key, value in data_dict.items():
                    if key[:5] in ['Time_']:
                        self.timer.addTime(key[5:], value.mean().item())

            train_loss_dict = self.trainStep(data_dict)

            if self.is_logger:
                self.timer.pause('step')

            self.loss_dict_list.append(train_loss_dict)

            lr = self.getLr()

            if self.is_logger and self.step % self.accum_iter == 0:
                for key in train_loss_dict.keys():
                    value = 0
                    for i in range(len(self.loss_dict_list)):
                        value += self.loss_dict_list[i][key]
                    value /= len(self.loss_dict_list)
                    self.logger.addScalar("Train/" + key, value, self.step)
                self.logger.addScalar("Train/Lr", lr, self.step)

                if self.ema_loss is None:
                    self.ema_loss = train_loss_dict["Loss"]
                else:
                    ema_decay = self.toEMADecay()
                    self.ema_loss = self.ema_loss * ema_decay + train_loss_dict[
                        "Loss"
                    ] * (1 - ema_decay)

                self.logger.addScalar("Train/EMALoss", self.ema_loss, self.step)

                self.loss_dict_list = []

            if self.is_logger:
                for name in self.timer.time_sums:
                    self.logger.addScalar(f"Time/{name}", self.timer.lastTime(name), self.step)

                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3

                self.logger.addScalar('GPU/Allocated_GB', allocated, self.step)
                self.logger.addScalar('GPU/Reserved_GB', reserved, self.step)

                pbar.set_description(
                    "EPOCH %d LOSS %.6f LR %.4f"
                    % (
                        self.epoch,
                        train_loss_dict["Loss"],
                        self.getLr() / self.lr,
                    )
                )

            if self.save_checkpoint_freq > 0 and self.step % self.save_checkpoint_freq == 0:
                self.autoSaveModel(f'{self.step:06d}')

            self.step += 1

            if self.is_logger:
                pbar.update(1)

            if self.quick_test:
                break

        if self.is_logger:
            pbar.close()

        if self.is_logger:
            self.logger.addScalar("Train/Epoch", self.epoch, self.step)

        self.epoch += 1

        return True

    @torch.no_grad()
    def evalStep(
        self,
        data_dict: dict,
    ) -> dict:
        self.model.eval()

        data_dict = moveTo(data_dict, self.device)

        result_dict = self.model(data_dict)
        result_dict = self.postProcessData(data_dict, result_dict, False)
        loss_dict = self.getLossDict(data_dict, result_dict)

        orig_shards = self._swap_to_ema()
        ema_result_dict = self.model(data_dict)
        ema_loss_dict = self.getLossDict(data_dict, ema_result_dict)
        self._swap_from_ema(orig_shards)
        del orig_shards

        loss_item_dict = {}

        for key, item in loss_dict.items():
            if isinstance(item, torch.Tensor):
                loss_item_dict[key] = (
                    item.clone().detach().cpu().to(torch.float32).numpy()
                )
            elif not isinstance(item, str):
                loss_item_dict[key] = item

        for key, item in ema_loss_dict.items():
            if isinstance(item, torch.Tensor):
                loss_item_dict["EMA_" + key] = (
                    item.clone().detach().cpu().to(torch.float32).numpy()
                )
            elif not isinstance(item, str):
                loss_item_dict["EMA_" + key] = item

        return loss_item_dict

    @torch.no_grad()
    def evalEpoch(self) -> bool:
        if "eval" not in self.dataloader_dict.keys():
            return True

        best_metric_value = None

        if self.is_logger:
            dataloader = self.dataloader_dict["eval"]["dataloader"]

            async_dataloader = AsyncDataLoader(
                dataloader,
                partial(self.preProcessData, is_training=False),
                max_workers=self.num_workers,
                prefetch_depth=self.prefetch_factor,
            )

            avg_loss_dict = {}

            print("[INFO][BaseTrainer::evalEpoch]")
            print("\t start evaluating ...")
            pbar = tqdm(total=len(dataloader))
            for data_dict in async_dataloader:
                if data_dict is not None:
                    data_dict = moveTo(data_dict, self.device)
                    data_dict = self.preProcessDataWithGPU(data_dict, is_training=False)

                if data_dict is None:
                    pbar.update(1)
                    continue

                eval_loss_dict = self.evalStep(data_dict)

                for key, item in eval_loss_dict.items():
                    if key not in avg_loss_dict.keys():
                        avg_loss_dict[key] = [item]
                    else:
                        avg_loss_dict[key].append(item)

                pbar.set_description(
                    "EPOCH %d LOSS %.6f LR %.4f"
                    % (
                        self.epoch,
                        eval_loss_dict["Loss"],
                        self.getLr() / self.lr,
                    )
                )

                pbar.update(1)

                if self.quick_test:
                    break

            pbar.close()

            for key, item in avg_loss_dict.items():
                avg_item = np.mean(item)
                self.logger.addScalar("Eval/" + key, avg_item, self.step)

                if self.best_model_metric_name is not None:
                    if key == self.best_model_metric_name:
                        best_metric_value = float(avg_item)

        # Broadcast best metric from logger to all ranks so all can participate in autoSaveModel
        if self.best_model_metric_name is not None:
            metric_list = [best_metric_value]
            if dist.is_initialized():
                dist.broadcast_object_list(metric_list, src=0)
            best_metric_value = metric_list[0]

            if best_metric_value is not None:
                self.autoSaveModel("best", best_metric_value, self.is_metric_lower_better)

        return True

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        """
        self.logger.addScalar('Sample/' + model_name + '_name', value, self.step)
        self.logger.addPointCloud(model_name + '/name', value, self.step)
        return True
        """
        return True

    @torch.no_grad()
    def sampleStep(self) -> bool:
        if not self.is_logger:
            return True

        if self.quick_test:
            self.sampleModelStep(self.model, "Model")
            torch.cuda.empty_cache()
            return True

        if self.sample_results_freq <= 0:
            return True

        if self.epoch % self.sample_results_freq != 0:
            return True

        self.sampleModelStep(self.model, "Model")
        torch.cuda.empty_cache()
        return True

    def _swap_to_ema(self) -> List[torch.Tensor]:
        """Replace FSDP model shards with EMA weights; return original shards (CPU)."""
        orig_shards: List[torch.Tensor] = []
        for p, p_ema in zip(self.model.parameters(), self._ema_params):
            orig_shards.append(p.data.detach().cpu().clone())
            p.data.copy_(p_ema.to(device=p.device, dtype=p.dtype))
        return orig_shards

    def _swap_from_ema(self, orig_shards: List[torch.Tensor]) -> None:
        """Restore original training shards from CPU backup."""
        for p, orig in zip(self.model.parameters(), orig_shards):
            p.data.copy_(orig.to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def sampleEMAStep(self) -> bool:
        if not self.is_logger:
            return True

        if not self.quick_test:
            if self.sample_results_freq <= 0:
                return True
            if self.epoch % self.sample_results_freq != 0:
                return True

        orig_shards = self._swap_to_ema()

        self.model.eval()
        self.sampleModelStep(self.model, "EMA")

        self._swap_from_ema(orig_shards)
        del orig_shards
        torch.cuda.empty_cache()
        return True

    def postProcessData(
        self, data_dict: dict, result_dict: dict, is_training: bool = True
    ) -> dict:
        """
        if is_training:
            result_dict['new_name'] = new_value
            return result_dict
        """
        return result_dict

    def train(self) -> bool:
        self.step += 1
        self.epoch += 1

        final_step = self.step + self.finetune_step_num

        if self.is_logger:
            print("[INFO][BaseTrainer::train]")
            print("\t start training ...")

        while self.step < final_step or self.finetune_step_num < 0:
            for data_name in self.dataloader_dict.keys():
                if data_name == "eval":
                    continue

                repeat_num = self.dataloader_dict[data_name]["repeat_num"]

                for i in range(repeat_num):
                    if self.is_logger:
                        print("[INFO][BaseTrainer::train]")
                        print(
                            "\t start training on dataset [",
                            data_name,
                            "] ,",
                            i + 1,
                            "/",
                            repeat_num,
                            "...",
                        )

                    if not self.trainEpoch(data_name):
                        if self.is_logger:
                            print("[ERROR][BaseTrainer::train]")
                            print("\t trainEpoch failed!")
                        return False

                # self.autoSaveModel("last")

                if not self.evalEpoch():
                    if self.is_logger:
                        print("[ERROR][BaseTrainer::train]")
                        print("\t evalEpoch failed!")
                    return False

                if not self.sampleStep():
                    if self.is_logger:
                        print("[ERROR][BaseTrainer::train]")
                        print("\t sampleStep failed!")
                    return False

                if not self.sampleEMAStep():
                    if self.is_logger:
                        print("[ERROR][BaseTrainer::train]")
                        print("\t sampleEMAStep failed!")
                    return False

                # Sync all ranks after logger-only operations (eval, sample)
                dist.barrier()

        return True

    def _gather_ema_full_state_dict(self) -> dict:
        """Temporarily swap EMA shards into the FSDP model, gather full
        state dict, then restore the original training weights.

        orig_shards are kept on CPU to avoid doubling GPU memory usage.
        """
        orig_shards = [p.data.detach().cpu().clone() for p in self.model.parameters()]

        for p, p_ema in zip(self.model.parameters(), self._ema_params):
            p.data.copy_(p_ema.to(device=p.device, dtype=p.dtype))

        ema_full_sd = get_model_state_dict(
            self.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        for p, orig in zip(self.model.parameters(), orig_shards):
            p.data.copy_(orig.to(device=p.device, dtype=p.dtype))
        del orig_shards

        return ema_full_sd

    def collectModelStateDict(self) -> Dict:
        # All ranks must participate in get_model_state_dict (FSDP all-gather)
        full_model_sd = get_model_state_dict(
            self.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        ema_full_sd = self._gather_ema_full_state_dict()

        model_state_dict = {
            "model": full_model_sd,
            "ema_model": ema_full_sd,
            "ema_loss": self.ema_loss,
            "step": self.step,
            "epoch": self.epoch,
            "loss_min": self.loss_min,
        }

        return model_state_dict

    def saveModel(
        self,
        save_model_file_path: str,
    ) -> bool:
        # All ranks must participate in collectModelStateDict for FSDP all-gather
        model_state_dict = self.collectModelStateDict()

        if self.is_logger and save_model_file_path is not None:
            createFileFolder(save_model_file_path)

            tmp_save_model_file_path = save_model_file_path[:-4] + "_tmp.pth"

            torch.save(model_state_dict, tmp_save_model_file_path)

            removeFile(save_model_file_path)
            renameFile(tmp_save_model_file_path, save_model_file_path)

            if self.save_model_fn is not None:
                threading.Thread(
                    target=self.save_model_fn,
                    args=(save_model_file_path,),
                    daemon=True,
                ).start()

        dist.barrier()
        return True

    def autoSaveModel(
        self,
        name: str,
        value: Optional[float] = None,
        check_lower: bool = True,
    ) -> bool:
        skip = False

        if not self.is_logger or self.save_result_folder_path is None:
            skip = True

        if not skip and value is not None:
            if self.loss_min == float("inf"):
                if not check_lower:
                    self.loss_min = -float("inf")

            if check_lower:
                if value > self.loss_min:
                    skip = True
            elif value < self.loss_min:
                skip = True

            if not skip:
                self.loss_min = value

        skip_flag = [skip]
        if dist.is_initialized():
            dist.broadcast_object_list(skip_flag, src=0)
        skip = skip_flag[0]

        if skip:
            return False

        save_model_file_path = (
            self.save_result_folder_path + "model_" + name + ".pth"
        ) if self.is_logger else None

        self.saveModel(save_model_file_path)
        return True
