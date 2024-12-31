import os
import torch
import numpy as np
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from typing import Union
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.optim.adamw import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from base_trainer.Method.device import moveTo
from base_trainer.Method.time import getCurrentTime
from base_trainer.Method.path import createFileFolder, renameFile, removeFile
from base_trainer.Module.data_prefetcher import DataPrefetcher
# from base_trainer.Module.dataloader_x import DataLoaderX
from base_trainer.Module.logger import Logger


def setup_distributed(backend: str = 'nccl'):
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def check_and_replace_nan_in_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradient: {name}")
            param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
    return True


class BaseTrainer(ABC):
    def __init__(
        self,
        batch_size: int = 32,
        accum_iter: int = 1,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "auto",
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
        use_amp: bool = False,
    ) -> None:
        self.backend = 'nccl' if device != 'cpu' else 'gloo'
        self.local_rank = setup_distributed(self.backend)

        self.accum_iter = accum_iter
        if device == 'auto':
            self.device = torch.device('cuda:' + str(self.local_rank))
        else:
            self.device = device
        self.warm_step_num = warm_step_num / accum_iter
        self.finetune_step_num = finetune_step_num
        self.lr = lr * batch_size / lr_batch_size * self.accum_iter * dist.get_world_size()
        self.ema_start_step = ema_start_step
        self.ema_decay_init = ema_decay_init
        self.ema_decay = ema_decay

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.best_model_metric_name = best_model_metric_name
        self.is_metric_lower_better = is_metric_lower_better
        self.sample_results_freq = sample_results_freq
        self.use_amp = use_amp

        self.scaler = None
        if self.use_amp:
            self.scaler = GradScaler()

        self.step = 0
        self.epoch = 0
        self.loss_dict_list = []

        self.loss_min = float("inf")

        self.logger = None
        if self.local_rank == 0:
            self.logger = Logger()

        self.dataloader_dict = {}
        if not self.createDatasets():
            print('[ERROR][BaseTrainer::__init__]')
            print('\t createDatasets failed!')
            exit()

        for key, item in self.dataloader_dict.items():
            if key == 'eval':
                self.dataloader_dict[key]['dataloader'] = DataLoader(
                    dataset=item['dataset'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                continue

            self.dataloader_dict[key]['sampler'] = DistributedSampler(item['dataset'])
            self.dataloader_dict[key]['dataloader'] = DataLoader(
                dataset=item['dataset'],
                sampler=self.dataloader_dict[key]['sampler'],
                batch_size=batch_size,
                num_workers=num_workers,
            )

        self.model: nn.Module
        if not self.createModel():
            print('[ERROR][BaseTrainer::__init__]')
            print('\t createModel failed!')
            exit()

        if self.local_rank == 0:
            self.ema_model = deepcopy(self.model)
            self.ema_loss = None

        if self.backend == 'nccl':
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = DDP(self.model)

        if model_file_path is not None:
            if not self.loadModel(model_file_path):
                print('[ERROR][BaseTrainer::__init__]')
                print('\t loadModel failed!')
                exit()

        self.optim = AdamW(self.model.parameters(), lr=self.lr)
        self.sched = LambdaLR(self.optim, lr_lambda=self.warmup_lr)

        self.initRecords()

        self.gt_sample_added_to_logger = False
        return

    @abstractmethod
    def createDatasets(self) -> bool:
        '''
        self.dataloader_dict = {
            'loader_name': {
                'dataset': None,
                'repeat_num': 1,
            },
            'eval': {
                'dataset': None,
            },
        }
        '''
        pass

    @abstractmethod
    def createModel(self) -> bool:
        '''
        self.model = None
        '''
        pass

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][BaseTrainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        if 'model' in model_state_dict.keys():
            self.model.module.load_state_dict(model_state_dict["model"])
        if 'step' in model_state_dict.keys():
            self.step = model_state_dict['step']

        if self.local_rank == 0:
            if 'ema_model' in model_state_dict.keys():
                self.ema_model.load_state_dict(model_state_dict["ema_model"])
            if 'ema_loss' in model_state_dict.keys():
                self.ema_loss = model_state_dict['ema_loss']
            if 'loss_min' in model_state_dict.keys():
                self.loss_min = model_state_dict['loss_min']

        print('[INFO][BaseTrainer::loadModel]')
        print('\t model loaded from:', model_file_path)

        return True

    def initRecords(self) -> bool:
        if self.logger is None:
            return True

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def getLr(self) -> float:
        return self.optim.state_dict()["param_groups"][0]["lr"]

    def warmup_lr(self, step: int) -> float:
        if self.warm_step_num == 0:
            return 1.0

        return min(step, self.warm_step_num) / self.warm_step_num

    def toEMADecay(self) -> float:
        if self.step <= self.ema_start_step:
            return self.ema_decay_init + self.step / self.ema_start_step * (self.ema_decay - self.ema_decay_init)

        return self.ema_decay

    def ema(self) -> bool:
        ema_decay = self.toEMADecay()

        source_dict = self.model.module.state_dict()
        target_dict = self.ema_model.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * ema_decay + source_dict[key].data * (1 - ema_decay)
            )
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = True) -> dict:
        '''
        if is_training:
            data_dict['new_name'] = new_value
            return data_dict
        '''
        return data_dict

    @abstractmethod
    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        '''
        loss_dict = {
            'loss': 0.0,
        }
        return loss_dict
        '''
        pass

    def trainStep(
        self,
        data_dict: dict,
    ) -> dict:
        self.model.train()

        data_dict = moveTo(data_dict, self.device)

        result_dict = self.model(data_dict)

        result_dict = self.postProcessData(data_dict, result_dict, True)

        if self.use_amp:
            with autocast('cuda'):
                loss_dict = self.getLossDict(data_dict, result_dict)
        else:
            loss_dict = self.getLossDict(data_dict, result_dict)

        if 'Loss' not in loss_dict.keys():
            print('[ERROR][BaseTrainer::trainStep]')
            print('\t loss not found!')
            exit()

        loss = loss_dict['Loss']

        accum_loss = loss / self.accum_iter

        if self.use_amp:
            self.scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if not check_and_replace_nan_in_grad(self.model):
            print('[ERROR][BaseTrainer::trainStep]')
            print('\t check_and_replace_nan_in_grad failed!')
            exit()

        if (self.step + 1) % self.accum_iter == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optim)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            if self.use_amp:
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.step()

            self.sched.step()
            if self.local_rank == 0:
                self.ema()
            self.optim.zero_grad()

        loss_item_dict = {}
        for key, item in loss_dict.items():
            if isinstance(item, torch.Tensor):
                loss_item_dict[key] = item.clone().detach().cpu().numpy()
            elif not isinstance(item, str):
                loss_item_dict[key] = item

        return loss_item_dict

    def trainEpoch(self, data_name: str) -> bool:
        if data_name not in self.dataloader_dict.keys():
            print('[ERROR][BaseTrainer::trainEpoch]')
            print('\t data not exist!')
            print('\t data_name:', data_name)
            return False

        dataloader_dict = self.dataloader_dict[data_name]
        dataloader_dict['sampler'].set_epoch(self.epoch)

        dataloader = dataloader_dict['dataloader']
        data_prefetcher = DataPrefetcher(dataloader, self.local_rank)

        data_dict = data_prefetcher.next()

        if self.local_rank == 0:
            pbar = tqdm(total=len(dataloader))
        while data_dict is not None:
            data_dict = self.preProcessData(data_dict, True)

            train_loss_dict = self.trainStep(data_dict)

            self.loss_dict_list.append(train_loss_dict)

            lr = self.getLr()

            if (self.step + 1) % self.accum_iter == 0 and self.local_rank == 0:
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
                    self.ema_loss = self.ema_loss * ema_decay + train_loss_dict["Loss"] * (1 - ema_decay)

                self.logger.addScalar("Train/EMALoss", self.ema_loss, self.step)

                self.loss_dict_list = []

            if self.local_rank == 0:
                pbar.set_description(
                    "EPOCH %d LOSS %.6f LR %.4f"
                    % (
                        self.epoch,
                        train_loss_dict["Loss"],
                        self.getLr() / self.lr,
                    )
                )

            self.step += 1

            if self.local_rank == 0:
                pbar.update(1)

            data_dict = data_prefetcher.next()

        if self.local_rank == 0:
            pbar.close()

        self.epoch += 1

        return True

    @torch.no_grad()
    def evalStep(
        self,
        data_dict: dict,
    ) -> dict:
        self.model.eval()

        data_dict = moveTo(data_dict, self.device)

        result_dict = self.model.module(data_dict)

        result_dict = self.postProcessData(data_dict, result_dict, False)

        loss_dict = self.getLossDict(data_dict, result_dict)

        ema_result_dict = self.ema_model(data_dict)

        ema_loss_dict = self.getLossDict(data_dict, ema_result_dict)

        loss_item_dict = {}

        for key, item in loss_dict.items():
            if isinstance(item, torch.Tensor):
                loss_item_dict[key] = item.clone().detach().cpu().numpy()
            elif not isinstance(item, str):
                loss_item_dict[key] = item

        for key, item in ema_loss_dict.items():
            if isinstance(item, torch.Tensor):
                loss_item_dict['EMA_' + key] = item.clone().detach().cpu().numpy()
            elif not isinstance(item, str):
                loss_item_dict['EMA_' + key] = item

        return loss_item_dict

    @torch.no_grad()
    def evalEpoch(self) -> bool:
        if self.local_rank != 0:
            return True

        if 'eval' not in self.dataloader_dict.keys():
            return True

        dataloader = self.dataloader_dict['eval']['dataloader']

        avg_loss_dict = {}

        print('[INFO][BaseTrainer::evalEpoch]')
        print('\t start evaluating ...')
        pbar = tqdm(total=len(dataloader))
        for data_dict in dataloader:
            data_dict = self.preProcessData(data_dict, False)

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

        pbar.close()

        for key, item in avg_loss_dict.items():
            avg_item = np.mean(item)
            self.logger.addScalar("Eval/" + key, avg_item, self.step)

            if self.best_model_metric_name is not None:
                if key == self.best_model_metric_name:
                    self.autoSaveModel('best', avg_item, self.is_metric_lower_better)

        return True

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        '''
        self.logger.addScalar('Sample/' + model_name + '_name', value, self.step)
        self.logger.addPointCloud(model_name + '/name', value, self.step)
        return True
        '''
        return True

    @torch.no_grad()
    def sampleStep(self) -> bool:
        if self.local_rank != 0:
            return True

        if self.sample_results_freq <= 0:
            return True

        if self.epoch % self.sample_results_freq != 0:
            return True

        self.sampleModelStep(self.model.module, 'Model')
        return True

    @torch.no_grad()
    def sampleEMAStep(self) -> bool:
        if self.local_rank != 0:
            return True

        if self.sample_results_freq <= 0:
            return True

        if self.epoch % self.sample_results_freq != 0:
            return True

        self.sampleModelStep(self.ema_model, 'EMA')
        return True

    def postProcessData(self, data_dict: dict, result_dict: dict, is_training: bool = True) -> dict:
        '''
        if is_training:
            result_dict['new_name'] = new_value
            return result_dict
        '''
        return result_dict

    def train(self) -> bool:
        final_step = self.step + self.finetune_step_num

        if self.local_rank == 0:
            print("[INFO][BaseTrainer::train]")
            print("\t start training ...")

        while self.step < final_step or self.finetune_step_num < 0:

            for data_name in self.dataloader_dict.keys():
                if data_name == 'eval':
                    continue

                repeat_num = self.dataloader_dict[data_name]['repeat_num']

                for i in range(repeat_num):
                    if self.local_rank == 0:
                        print('[INFO][BaseTrainer::train]')
                        print('\t start training on dataset [', data_name, '] ,', i + 1, '/', repeat_num, '...')

                    if not self.trainEpoch(data_name):
                        print('[ERROR][BaseTrainer::train]')
                        print('\t trainEpoch failed!')
                        return False

                    self.autoSaveModel("last")

                    if not self.evalEpoch():
                        print('[ERROR][BaseTrainer::train]')
                        print('\t evalEpoch failed!')
                        return False

                    if not self.sampleStep():
                        print('[ERROR][BaseTrainer::train]')
                        print('\t sampleStep failed!')
                        return False

                    if not self.sampleEMAStep():
                        print('[ERROR][BaseTrainer::train]')
                        print('\t sampleEMAStep failed!')
                        return False

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.module.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "ema_loss": self.ema_loss,
            "step": self.step,
            "loss_min": self.loss_min,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, name: str, value: Union[float, None] = None, check_lower: bool = True) -> bool:
        if self.local_rank != 0:
            return True

        if self.save_result_folder_path is None:
            return False

        if value is not None:
            if self.loss_min == float("inf"):
                if not check_lower:
                    self.loss_min = -float("inf")

            if check_lower:
                if value > self.loss_min:
                    return False
            elif value < self.loss_min:
                return False

            self.loss_min = value

        save_model_file_path = self.save_result_folder_path + "model_" + name + ".pth"

        tmp_save_model_file_path = save_model_file_path[:-4] + "_tmp.pth"

        self.saveModel(tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)

        return True
