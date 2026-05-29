from pdetransformer.core.mixed_channels import SingleStepSupervised
import torch.nn as nn
import lightning
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
import numpy as np

class SingleStepSupervised(SingleStepSupervised):

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

        return x, y, task_idx


    def predict(self, batch, device, num_frames=20, generator=None,
                output_type='numpy', num_inference_steps=100, return_dict=True,
                reference_boundary=False, batch_dim=True, trim:int=0, **kwargs):

        boundary_slice = 0

        with torch.no_grad():

            input_0, input_1, labels = self.get_input(batch, batch_dim=batch_dim, trim=trim)

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)[:, :num_frames-1]
            labels = labels.to(device)

            if generator is None:
                generator = (torch.Generator(device=input_0.device)
                             .manual_seed(2024))

            frames = [input_0.cpu()[...,-1:,:,:]]
            previous_frame = input_0

            for i in tqdm(range(num_frames-1)):

                x0 = self.predict_step(previous_frame, generator=generator, class_labels=labels,
                                       num_inference_steps=num_inference_steps)
                #x0 = self.postprocess_output(x0, batch['physical_metadata'])

                # fill boundaries with reference
                if reference_boundary:

                    if len(x0.shape) == 4: # 2D
                        x0[:, :, 0:boundary_slice, :] = input_1[:, i, :, 0:boundary_slice, :]
                        x0[:, :, -boundary_slice:, :] = input_1[:, i, :, -boundary_slice:, :]
                        x0[:, :, :, 0:boundary_slice] = input_1[:, i, :, :, 0:boundary_slice]
                        x0[:, :, :, -boundary_slice:] = input_1[:, i, :, :, -boundary_slice:]

                    if len(x0.shape) == 5: # 3D
                        x0[:, :, 0:boundary_slice, :, :] = input_1[:, i, :, 0:boundary_slice, :, :]
                        x0[:, :, -boundary_slice:, :, :] = input_1[:, i, :, -boundary_slice:, :, :]
                        x0[:, :, :, 0:boundary_slice, :] = input_1[:, i, :, :, 0:boundary_slice, :]
                        x0[:, :, :, -boundary_slice:, :] = input_1[:, i, :, :, -boundary_slice:, :]
                        x0[:, :, :, :, 0:boundary_slice] = input_1[:, i, :, :, :, 0:boundary_slice]
                        x0[:, :, :, :, -boundary_slice:] = input_1[:, i, :, :, :, -boundary_slice:]

                previous_frame = x0
                frames.append(x0.cpu())

            vid = np.array([frame[...,-1:,:,:].numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(torch.concat([frames[0].unsqueeze(1),
                                               input_1.cpu()[...,-1:,:,:]], dim=1))

            if not batch_dim:
                vid = vid[0]
                reference = reference[0]

        return vid, reference
