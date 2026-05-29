from typing import List, Optional, Union, Iterable

from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig, OmegaConf

import glob
import numpy as np
import os
import wandb
from PIL import Image
from tqdm import tqdm

import torch
import subprocess

import seaborn as sns
from matplotlib import pyplot as plt
from pdetransformer.visualization import render_trajectory, render_vape_3d

from pdetransformer.data.simulations_apebench.render import zigzag_alpha
from pdetransformer.data.multi_module import get_subdatasets_from_dataloader
from pdetransformer.utils import instantiate_from_config, get_pipeline

def prepareImage(prediction: np.ndarray, target: np.ndarray, clamp=False, rescale=True):
    '''
    Prepare images for logging by clamping, rescaling, and trimming.
    Additionally, appends prediction and target into same image alongside each other.
    in addition to returning them separately. and the difference between them.'''
    if clamp:
        prediction = np.clip(prediction, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

    if rescale:
        prediction = (prediction + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        prediction = np.clip(prediction, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

    diff = prediction - target
    diff = (diff + 1) / (2)
    #diff = np.clip((diff + 1) / 2, 0.0, 1.0)
    comparison = np.concatenate([target, prediction, diff], axis=-1)

    return prediction, target, diff, comparison
class MultiTaskImageLogger(Callback):
    def __init__(self, frequency: int,  max_images: int = 10, num_frames: int = 10, num_inference_steps: int = 100,
                 clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_first_step=False, test_only=False,
                 reference_boundary=False, trim:int=0, prepare_plots: Optional[DictConfig]=None,
                 randomize: bool = True,sel_denorm_channels = None, **kwargs):

        super().__init__()

        self.rescale = rescale
        self.frequency = frequency
        self.max_images = max_images
        self.test_only = test_only
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.reference_boundary = reference_boundary
        self.trim = trim
        self.randomize = randomize
        if not prepare_plots is None:
            self.prepare_plots = instantiate_from_config(prepare_plots)
        else:
            self.prepare_plots = torch.nn.Identity()

        self.log_steps = [2 ** n for n in range(int(np.log2(self.frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.frequency]
        self.sel_denorm_channels = sel_denorm_channels
        self.clamp = clamp
        self.disabled = disabled

        self.log_first_step = log_first_step

    @rank_zero_only
    def log_images(self, trainer, pl_module):
        pass
        try:
            current_epoch = trainer.current_epoch

            is_train = pl_module.training

            if is_train:
                pl_module.eval()

            test_dataloader = trainer.test_dataloaders
            subsets, dataset = get_subdatasets_from_dataloader(test_dataloader)

            logdir = trainer.logger.experiment.config["runtime"]["logdir"]
            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)
            images_per_task = max(1, self.max_images // len(subsets))

            for d in range(len(subsets)):
                images = []
                if self.randomize:
                    indices = np.random.choice(len(subsets[d]), images_per_task, replace=False)
                else:
                    indices = list(range(0, min(len(subsets[d]), images_per_task)))
                for dd in range(min(len(subsets[d]), images_per_task)):
                    sample = subsets[d][indices[dd]] # only first sample
                    simulation_name = subsets[d].dataset.dset_name

                    sample = dataset.process_sample(sample,add_batch_dim=False)

                    #frames, target, labels = pl_module.get_input(sample)
                    #predict
                    predictions, reference = pl_module.predict(sample, pl_module.device,
                                                                num_frames=self.num_frames,
                                                                num_inference_steps=self.num_inference_steps,
                                                                reference_boundary=self.reference_boundary,
                                                                generator=generator, batch_dim=False)

                #class_labels.append(labels)

                #class_labels = torch.tensor(class_labels, device=pl_module.device, dtype=torch.long)

                #images = self.prepare_plots(images_tensor)
                    '''
                    if self.rescale:
                        images = (images + 1.0) / 2.0
                        images = np.clip(images, 0.0, 1.0)
                    images = np.where(images>0, 1.0, 0.0)
                    '''
                    if not (self.sel_denorm_channels is None or self.sel_denorm_channels == None or self.sel_denorm_channels == 'None'):
                        mean = subsets[d].dataset.norm_strat_data.mean[self.sel_denorm_channels] #4
                        std = subsets[d].dataset.norm_strat_data.std[self.sel_denorm_channels]
                    else:
                        mean, std = 0.0, 1.0
                        #images = images.cpu().numpy()
                        #image_ = prepareImage(np.where(predictions[:,-1]>-0.0, 1.0, 0.0) ,np.where(reference[:,-1]>0, 1.0, 0.0), False, False )[-1][0]
                    # Use index 0 if only 1 sample, otherwise use index 1
                    idx = min(1, predictions.shape[0] - 1)
                    image_ = prepareImage(predictions[idx][None]*std + mean  ,reference[idx][None]*std +   mean , True, False)[-1][0]
                    images.append(image_)
                for c, image in enumerate(images):
                    savedir = f"{logdir}/images/{simulation_name}"
                    os.makedirs(savedir, exist_ok=True)
                    savename = f'e{current_epoch}_i{c}.png'
                    img_path = f"{savedir}/{savename}"
                    img = Image.fromarray((image[-1] * 255).astype(np.uint8))
                    #img = Image.fromarray((np.repeat(image,(3),axis = 0).transpose(2,1,0) * 255).astype(np.uint8))

                    img.save(img_path)
                    trainer.logger.experiment.log({
                        f'{simulation_name}_{c}': wandb.Image(img_path, caption=f"{simulation_name}")
                    })
        

            if is_train:
                pl_module.train()

        except Exception as e:
            rank_zero_warn(f'Image logging failed with exception {e}')  
            raise e

    def check_frequency(self, check_idx):
        if ((check_idx + 1) % self.frequency) == 0 or (check_idx in self.log_steps) or (check_idx == 0 and self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        if ((not self.disabled) and (not self.test_only) and self.log_first_step and
                pl_module.current_epoch == 0):
            self.log_images(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.setup_test_dataloader(trainer)
        if ((not self.disabled) and (not self.test_only) and (pl_module.current_epoch > 0) and
                self.check_frequency(pl_module.current_epoch) and (self.max_images > 0)):
            self.log_images(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        if not self.disabled:
            self.log_images(trainer, pl_module)

