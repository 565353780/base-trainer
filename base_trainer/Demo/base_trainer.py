import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Callable, Optional, Union

from base_trainer.Module.base_trainer import BaseTrainer


def demo_compile_fn(model) -> None:
    dit = model.dit
    for i, block in enumerate(dit.double_blocks):
        dit.double_blocks[i] = torch.compile(
            block,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    for i, block in enumerate(dit.single_blocks):
        dit.single_blocks[i] = torch.compile(
            block,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    return

class Trainer(BaseTrainer):
    def __init__(
        self,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        prefetch_factor: int = 4,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        quick_test: bool = False,
        save_checkpoint_freq: int = -1,
        compile_fn: Optional[Callable]=None,
        load_model_fn: Optional[Callable]=None,
        save_model_fn: Optional[Callable]=None,
    ) -> None:
        # super params definition here
        # self.name = value
        # ...

        super().__init__(
            batch_size=batch_size,
            accum_iter=accum_iter,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            model_file_path=model_file_path,
            weights_only=weights_only,
            warm_step_num=warm_step_num,
            finetune_step_num=finetune_step_num,
            lr=lr,
            lr_batch_size=lr_batch_size,
            ema_start_step=ema_start_step,
            ema_decay_init=ema_decay_init,
            ema_decay=ema_decay,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            best_model_metric_name=best_model_metric_name,
            is_metric_lower_better=is_metric_lower_better,
            sample_results_freq=sample_results_freq,
            quick_test=quick_test,
            save_checkpoint_freq=save_checkpoint_freq,
            compile_fn=compile_fn,
            load_model_fn=load_model_fn,
            save_model_fn=save_model_fn,
        )
        return

    def createDatasets(self) -> bool:
        self.dataloader_dict["name"] = {
            "dataset": Dataset(self.dtype),
            "repeat_num": 1,
        }

        self.dataloader_dict["eval"] = {
            "dataset": Dataset(self.dtype),
        }

        # crop data num for faster evaluation
        self.dataloader_dict["eval"]["dataset"].data_list = self.dataloader_dict[
            "eval"
        ]["dataset"].data_list[:64]
        return True

    def createModel(self) -> bool:
        self.model = nn.Module().to(self.device, dtype=self.dtype)
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["drop_prob"] = 10.0
        else:
            data_dict["drop_prob"] = 0.0

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        ut = data_dict["ut"]
        vt = result_dict["vt"]

        loss = torch.pow(vt - ut, 2).mean()

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: torch.nn.Module, model_name: str) -> bool:
        sample_num = 3
        dataset = self.dataloader_dict["mash"]["dataset"]

        model.eval()

        data = dataset.__getitem__(0)

        # process data here
        pcd = data["pcd"]
        mesh = data["mesh"]

        self.logger.addPointCloud(model_name + "/pcd_0", pcd, self.step)
        self.logger.addMesh(model_name + "/mesh_0", mesh, self.step)

        return True


def demo():
    batch_size = 8
    accum_iter = 20
    num_workers = 16
    prefetch_factor = 4
    model_file_path = "../../output/20241225_15:14:36/model_last.pth".replace(
        "../../", "./"
    )
    model_file_path = None
    weights_only = False
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 2e-4
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 1
    quick_test = False
    save_checkpoint_freq = 1000
    compile_fn = demo_compile_fn
    save_model_fn = None

    trainer = Trainer(
        batch_size=batch_size,
        accum_iter=accum_iter,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        model_file_path=model_file_path,
        weights_only=weights_only,
        warm_step_num=warm_step_num,
        finetune_step_num=finetune_step_num,
        lr=lr,
        lr_batch_size=lr_batch_size,
        ema_start_step=ema_start_step,
        ema_decay_init=ema_decay_init,
        ema_decay=ema_decay,
        save_result_folder_path=save_result_folder_path,
        save_log_folder_path=save_log_folder_path,
        best_model_metric_name=best_model_metric_name,
        is_metric_lower_better=is_metric_lower_better,
        sample_results_freq=sample_results_freq,
        quick_test=quick_test,
        save_checkpoint_freq=save_checkpoint_freq,
        compile_fn=compile_fn,
        save_model_fn=save_model_fn,
    )

    trainer.train()
    return True
