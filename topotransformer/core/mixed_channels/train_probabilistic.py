from pdetransformer.core.mixed_channels import SingleStepDiffusion as SSD
import torch.nn as nn
import lightning
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
import numpy as np
from diffusers.utils.torch_utils import randn_tensor

class SingleStepDiffusion(SSD):

    def get_input(self, batch, batch_dim=True, trim:int=0):

        conditioning: torch.Tensor = batch["conditioning"]
        data: torch.Tensor = batch["data"]
        meta_data_loading: dict = batch["loading_metadata"]
        meta_data_physical: dict = batch["physical_metadata"]

        if batch_dim:
            x: torch.Tensor = conditioning[:, 0+trim]
            y: torch.Tensor = data[:, 0+trim:]

            task_idx = meta_data_physical['PDE'][:, 0]

        else:
            x: torch.Tensor = conditioning[0+trim]
            y: torch.Tensor = data[0+trim:]
            task_idx = meta_data_physical['PDE']

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)

            if not torch.is_tensor(task_idx):
                task_idx = torch.tensor(task_idx)

        if self.downsample_factor > 1:
            # downsample with average pooling
            x = nn.functional.avg_pool2d(x, self.downsample_factor)

            num_batches = y.shape[0]
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
            y = nn.functional.avg_pool2d(y, self.downsample_factor)
            y = y.reshape(num_batches, -1, y.shape[-3], y.shape[-2], y.shape[-1])

        return x,y, task_idx#(x*127-104 +135) /308, y, task_idx

    def predict_step(self, conditioning, target_channels=1,
                     num_inference_steps=100, generator=None, **kwargs):
        """Override parent to correctly separate noise (target) from conditioning channels."""
        B, _, H, W = conditioning.shape
        noise_shape = (B, target_channels, H, W)

        if self.device.type == "mps":
            x0 = randn_tensor(noise_shape, generator=generator)
            x0 = x0.to(self.device)
        else:
            x0 = randn_tensor(noise_shape, generator=generator, device=self.device)

        input = torch.cat([x0, conditioning], dim=1)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            model_output = self.forward(input, t, **kwargs).sample
            x0 = self.scheduler.step(model_output, t, x0, generator=generator).prev_sample
            input = torch.cat([x0, conditioning], dim=1)

        return x0

    def predict(self, batch, device, num_frames=20, generator=None,
                output_type='numpy', num_inference_steps=100, return_dict=True, batch_dim=True, **kwargs):

        with torch.no_grad():

            input_0, input_1, labels = self.get_input(batch, batch_dim=batch_dim)

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            labels = labels.to(device)

            target_channels = input_1.shape[-3]  # number of target channels

            if generator is None:
                generator = (torch.Generator(device=input_0.device)
                             .manual_seed(2024))

            frames = [input_0.cpu()[...,-1:,:,:]]
            conditioning = input_0

            for _ in tqdm(range(num_frames)):

                x0 = self.predict_step(conditioning, target_channels=target_channels,
                                       generator=generator, class_labels=labels,
                                       num_inference_steps=num_inference_steps)
                frames.append(x0.cpu())

            vid = np.array([frame[...,-1:,:,:].numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(torch.concat([frames[0].unsqueeze(1),
                                               input_1.cpu()[...,-1:,:,:]], dim=1))

            if not batch_dim:
                vid = vid[0]
                reference = reference[0]

        return vid, reference