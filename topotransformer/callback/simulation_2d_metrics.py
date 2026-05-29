from typing import Optional, Any, Dict

from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only
import torch
from omegaconf import DictConfig, OmegaConf

from pdetransformer.callback.ema import EMAOptimizer
from pdetransformer.metric.metric import get_metrics
from pdetransformer.callback import Simulation2DMetricLoggerCustom
import logging
log = logging.getLogger(__name__)

import torch.nn as nn

import numpy as np

class Simulation2DMetricLoggerCustom(Simulation2DMetricLoggerCustom):
    def __init__(self, frequency: int, batch_size: int,
                 num_inference_steps: int = 100, num_frames: int = 20,
                 trim_start: int = 1,
                 use_ema: bool = True,
                 log_first_step: bool = False, test_only: bool = False,
                 disabled: bool = False,
                 reference_boundary: bool = False,
                 metric_config: DictConfig = None,
                 sel_channels: Optional[list] = None,
                 sel_channels_norm: Optional[list] = None):
        """

        :param frequency:
        :param batch_size:
        :param num_inference_steps:
        :param num_frames:
        :param trim_start:
        :param use_ema:
        :param log_first_step:
        :param test_only:
        :param disabled:
        :param reference_boundary:
        :param metric_config:
        :param sel_channels: list of selected channels to compute metrics on
        """

        super().__init__( frequency=frequency, batch_size=batch_size,
                          num_inference_steps=num_inference_steps,
                          num_frames=num_frames,
                          trim_start=trim_start,
                          use_ema=use_ema,
                          log_first_step=log_first_step,
                          test_only=test_only,
                          disabled=disabled,
                          reference_boundary=reference_boundary,
                          metric_config=metric_config)
        self.sel_channels = sel_channels
        self.sel_channels_norm = sel_channels_norm
    @rank_zero_only
    def update_metrics_impl(self, trainer, pl_module, dataloader):
        #return
        metrics_torch = nn.ModuleList(get_metrics(
                self.metric_config)).to(pl_module.device)
        metrics_torch.eval()
        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)
        #get train dataloader to use mean and std of train
        #train_dataloader = trainer.datamodule.train_dataloader()
        for batch in dataloader:
            meanFunc = lambda y: dataloader.dataset.datasets[y].dataset.norm_strat_data.mean
            stdFunc = lambda y: dataloader.dataset.datasets[y].dataset.norm_strat_data.std
            #meanFunc = lambda y: train_dataloader.dataset.datasets[y].dataset.dataset.norm_strat_data.mean
            #stdFunc = lambda y: train_dataloader.dataset.datasets[y].dataset.dataset.norm_strat_data.std

            class_labels = batch["loading_metadata"]["dataset_idx"].flatten().to(pl_module.device)
            means = list(map(meanFunc, batch["loading_metadata"]["dataset_idx"].flatten().long().tolist()))
            stds = list(map(stdFunc, batch["loading_metadata"]["dataset_idx"].flatten().long().tolist()))
            #convertto torch tensor
            means = torch.tensor(means, device=pl_module.device, dtype=torch.float32).to(pl_module.device).unsqueeze(-1)
            stds = torch.tensor(stds, device=pl_module.device, dtype=torch.float32).to(pl_module.device).unsqueeze(-1)

            if not (self.sel_channels_norm is None or self.sel_channels_norm == "None"):
                means = means[:, self.sel_channels_norm]
                stds = stds[:, self.sel_channels_norm]
            else:
                means = 0
                stds = 1
            with torch.inference_mode():
                prediction, target = pl_module.predict(batch, pl_module.device,
                                                    num_frames=self.num_frames,
                                                    num_inference_steps=self.num_inference_steps,
                                                    generator=generator,
                                                    #reference_boundary=self.reference_boundary,
                                                    batch_dim=True,
                                                    output_probabilistic=True)
                prediction= prediction[:, self.trim_start:]
                target= target[:, self.trim_start:]
                if self.sel_channels is not None:
                    prediction = prediction[..., self.sel_channels, :, :]  # first element is the input frame
                    target = target[..., self.sel_channels, :, :]
                # first element is the input frame
                prediction = torch.from_numpy(prediction).to(pl_module.device)
                target = torch.from_numpy(target).to(pl_module.device)

                for metric in metrics_torch:
                        
                    metric.update(stds*prediction + means, stds*target + means, class_labels=class_labels)

        metric_dict = {"epoch": trainer.current_epoch}

        for metric in metrics_torch:

            metric_name = metric.__class__.__name__

            metric_value = metric.compute()

            if isinstance(metric_value, tuple):
                metric_dict[f"{metric_name}_mean"] = metric_value[0]
                metric_dict[f"{metric_name}_std"] = metric_value[1]

            else:
                metric_dict[f"{metric_name}"] = metric_value

            if trainer.global_rank == 0:
                if hasattr(trainer, "test_config_debug"):
                    logdir = trainer.test_config_debug["runtime"]["logdir"]
                else:
                    logdir = trainer.logger.experiment.config["runtime"]["logdir"]
                #logdir = trainer.config["runtime"]["logdir"]
                metric.save(logdir + '/metrics/', f"{metric_name}_{trainer.current_epoch}.csv")

        self.log_metrics(trainer, metric_dict)

