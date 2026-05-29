"""
Bernoulli Flow Matching Training Module.

Uses Fisher-Rao geodesics (α=0 geometry) for flow matching on binary data.
The network predicts Hellinger velocity w = d/dt √π, which is constant along geodesics.

Geodesic interpolation: π_t = ((1-t)√π₀ + t√π₁)²
Hellinger velocity: w = √π₁ - √π₀ (constant along geodesic)
"""
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional

import lightning
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from pdetransformer.sampler.scheduler import OdeEulerScheduler
from pdetransformer.utils import instantiate_from_config
import logging
log = logging.getLogger(__name__)



class BernoulliFlowDiffusion(lightning.LightningModule):
    """
    Bernoulli Flow Matching model using Fisher-Rao geometry.
    
    Key properties:
    1. Model predicts Hellinger velocity w = √π₁ - √π₀
    2. Velocity is constant along Fisher-Rao geodesics
    3. Sampling integrates: d√π/dt = w, then π = (√π)²
    4. Outputs are naturally in [0, 1]
    """
    
    def __init__(
        self,
        model,
        ckpt_path: Optional[str] = None,
        ignore_keys: Optional[List[str]] = None,
        image_key: int = 0,
        monitor: Optional[str] = None,
        downsample_factor: int = 1,
        optimizer: str = 'adamw',
        learning_rate: float = 1e-4,
        eps: float = 1e-6,
        prior_type: str = 'uniform',
        volume_fraction: float = 0.5,
        volume_fraction_std: float = 0.1,
        num_sampling_steps: int = 100,
        sampling_method: str = 'euler',
        threshold: Optional[float] = 2.5,
        denormalize_target: bool = False,
        target_channel_idx: int = 4,
        **kwargs,
    ):
        super().__init__()
        
        self.image_key = image_key
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.downsample_factor = downsample_factor
        self.num_sampling_steps = num_sampling_steps
        self.sampling_method = sampling_method
        self.threshold = threshold
        
        # Numerical stability
        self.eps = eps
        
        # Prior settings
        self.prior_type = prior_type
        self.volume_fraction = volume_fraction
        self.volume_fraction_std = volume_fraction_std
        
        # Denormalization settings
        self.denormalize_target = denormalize_target
        self.target_channel_idx = target_channel_idx
        self._norm_params_cache = {}
        
        # Model
        self.model: nn.Module = instantiate_from_config(model)
        
        # ODE scheduler
        self.scheduler = OdeEulerScheduler(num_train_timesteps=100)
        
        if monitor is not None:
            self.monitor = monitor
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def on_train_start(self):
        """Set up dataloader reference for dynamic norm lookup."""
        if self.denormalize_target and hasattr(self.trainer, 'train_dataloader'):
            dl = self.trainer.train_dataloader
            if hasattr(dl, 'loaders'):
                dl = dl.loaders
            self._cache_norm_params(dl)
    
    def _cache_norm_params(self, dataloader):
        """Cache normalization parameters for inference."""
        try:
            ds = dataloader.dataset.datasets[0].dataset
            if hasattr(ds, 'dataset'):
                ds = ds.dataset
            if hasattr(ds, 'norm_strat_data'):
                mean = float(ds.norm_strat_data.mean[self.target_channel_idx, 0, 0])
                std = float(ds.norm_strat_data.std[self.target_channel_idx, 0, 0])
                self._norm_params_cache[0] = (mean, std)
                log.debug(f"[BernoulliFlow] Cached norm params: mean={mean:.4f}, std={std:.4f}")
        except Exception as e:
            log.warning(f"[BernoulliFlow] Could not cache norm params: {e}")
    
    def _clamp_prob(self, p: torch.Tensor) -> torch.Tensor:
        """Clamp probabilities for numerical stability."""
        return p.clamp(self.eps, 1.0 - self.eps)
    
    def _sample_prior(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Sample from prior distribution in [0, 1].
        
        Prior types:
        - 'uniform': Deterministic 0.5 everywhere (not generative)
        - 'uniform_random': Random value in (0,1), same for all pixels (generative, spatially uniform)
        - 'random': Uniform(0,1) per pixel (generative, high diversity)
        - 'volume_fraction': Per-sample volume fraction with noise
        - 'beta': Beta(0.5, 0.5) per pixel (U-shaped, favors extremes)
        """
        if self.prior_type == 'uniform':
            # Deterministic prior - same output every time
            #return torch.full(shape, 0.5, device=device)
            return torch.bernoulli(torch.full(shape, 0.5, device=device))
        elif self.prior_type == 'uniform_random':
            # Random value per sample, but same across all pixels (spatially uniform)
            # Shape: (B, C, H, W) -> sample one value per batch item
            batch_size = shape[0]
            values = torch.rand(batch_size, 1, 1, 1, device=device).clamp(self.eps, 1.0 - self.eps)
            return values.expand(shape)
        elif self.prior_type == 'random':
            # Random prior per pixel - makes model generative
            return torch.rand(shape, device=device).clamp(self.eps, 1.0 - self.eps)
        elif self.prior_type == 'volume_fraction':
            # Sample volume fraction with noise
            vf = self.volume_fraction + self.volume_fraction_std * torch.randn(
                shape[0], 1, 1, 1, device=device
            )
            vf = vf.clamp(0.1, 0.9).expand(shape)
            return vf
        elif self.prior_type == 'beta':
            # Beta(0.5, 0.5) = arcsin distribution, U-shaped
            return torch.distributions.Beta(0.5, 0.5).sample(shape).to(device)
        else:
            # Fallback to random
            return torch.rand(shape, device=device)#.clamp(self.eps, 1.0 - self.eps)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, t, **kwargs)
    
    def get_pipeline_args(self):
        return {"unet": self.model}
    
    def get_input(self, batch, batch_dim=True, trim: int = 0):
        conditioning: torch.Tensor = batch["conditioning"]
        data: torch.Tensor = batch["data"]
        meta_data_physical: dict = batch["physical_metadata"]
        
        if batch_dim:
            x: torch.Tensor = conditioning[:, 0 + trim]
            y: torch.Tensor = data[:, 0 + trim:]
            task_idx = meta_data_physical['PDE'][:, 0]
        else:
            x: torch.Tensor = conditioning[0 + trim]
            y: torch.Tensor = data[0 + trim:]
            task_idx = meta_data_physical['PDE']
            
            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)
            
            if not torch.is_tensor(task_idx):
                task_idx = torch.tensor(task_idx)
        
        if self.downsample_factor > 1:
            x = nn.functional.avg_pool2d(x, self.downsample_factor)
            
            num_batches = y.shape[0]
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
            y = nn.functional.avg_pool2d(y, self.downsample_factor)
            y = y.reshape(num_batches, -1, y.shape[-3], y.shape[-2], y.shape[-1])
        
        return x, y, task_idx
    
    def init_from_ckpt(self, path: str, ignore_keys: Optional[List[str]] = None):
        if ignore_keys is None:
            ignore_keys = []
        
        if Path(path).is_dir():
            path = Path(path).joinpath("last.ckpt")
        else:
            path = Path(path)
        
        if path.is_file():
            sd = torch.load(path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        del sd[k]
            self.load_state_dict(sd, strict=False)
            log.info(f"Restored from {path}")
    
    def training_step(self, batch, batch_idx):
        """
        Training step using Fisher-Rao geodesic flow matching.
        
        1. Sample t ~ U(0, 1)
        2. Sample prior π₀ 
        3. Get target π₁ (binary topology)
        4. Compute geodesic interpolant: √π_t = (1-t)√π₀ + t√π₁
        5. Compute Hellinger velocity: w = √π₁ - √π₀
        6. Loss = MSE(w_pred, w_target)
        """
        input_0, input_1, labels = self.get_input(batch)
        #input_0 = input_0- 0.2
        # Get target (last channel of input_1)
        x1 = input_1[:, -1]
        if x1.dim() == 3:
            x1 = x1.unsqueeze(1)
        
        device = x1.device
        batch_size = x1.shape[0]
        
        # Sample timestep
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Sample prior and clamp target
        pi_0 = self._sample_prior(x1.shape, device)
        x1_clamped = self._clamp_prob(x1.float())
        
        # Compute in root-probability space
        sqrt_pi_0 = torch.sqrt(pi_0)
        sqrt_x1 = torch.sqrt(x1_clamped)
        
        # Geodesic interpolation in root space
        #sqrt_pi_t = (1 - t) * sqrt_pi_0 + t * sqrt_x1
        #pi_t = self._clamp_prob(sqrt_pi_t ** 2)
        pi_t = (1-t) * pi_0 + t*x1_clamped

        pi_t = torch.bernoulli(pi_t)
        # Hellinger velocity (constant along geodesic)
        #w_target = sqrt_x1 - sqrt_pi_0
        
        # Model prediction
        # Input: [pi_t, conditioning]
        input_tensor = torch.cat([pi_t, input_0.unsqueeze(1) if input_0.dim() == 3 else input_0], dim=1)
        output = self.forward(input_tensor, t.squeeze(), class_labels=labels)
        output = output.sample if hasattr(output, 'sample') else output
        #w_pred = output.sample if hasattr(output, 'sample') else output
        #w_pred = F.sigmoid(w_pred)
        # MSE loss on Hellinger velocity
        #loss = ((w_pred - w_target) ** 2).mean()
        #loss = F.binary_cross_entropy_with_logits(w_pred, w_target)
        loss = F.binary_cross_entropy_with_logits(output, x1.float())
        # Logging
        self.log('loss', loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log('train/w_pred_norm', w_pred.abs().mean().item())
        #self.log('train/w_target_norm', w_target.abs().mean().item())
        self.log('train/pi_t_mean', pi_t.mean().item())
        self.log('train/x1_mean', x1_clamped.mean().item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_0, input_1, labels = self.get_input(batch)
        #input_0 = input_0- 0.2
        # Get target (last channel of input_1)
        x1 = input_1[:, -1]
        if x1.dim() == 3:
            x1 = x1.unsqueeze(1)
        
        device = x1.device
        batch_size = x1.shape[0]
        
        # Sample timestep
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Sample prior and clamp target
        pi_0 = self._sample_prior(x1.shape, device)
        x1_clamped = self._clamp_prob(x1.float())
        
        # Compute in root-probability space
        sqrt_pi_0 = torch.sqrt(pi_0)
        sqrt_x1 = torch.sqrt(x1_clamped)
        
        # Geodesic interpolation in root space
        #sqrt_pi_t = (1 - t) * sqrt_pi_0 + t * sqrt_x1
        #pi_t = self._clamp_prob(sqrt_pi_t ** 2)
        pi_t = (1-t) * pi_0 + t*x1_clamped

        pi_t = torch.bernoulli(pi_t)
        # Hellinger velocity (constant along geodesic)
        #w_target = sqrt_x1 - sqrt_pi_0
        
        # Model prediction
        # Input: [pi_t, conditioning]
        input_tensor = torch.cat([pi_t, input_0.unsqueeze(1) if input_0.dim() == 3 else input_0], dim=1)
        output = self.forward(input_tensor, t.squeeze(), class_labels=labels)
        output = output.sample if hasattr(output, 'sample') else output
        #w_pred = output.sample if hasattr(output, 'sample') else output
        #w_pred = F.sigmoid(w_pred)
        # MSE loss on Hellinger velocity
        #loss = ((w_pred - w_target) ** 2).mean()
        #loss = F.binary_cross_entropy_with_logits(w_pred, w_target)
        loss = F.binary_cross_entropy_with_logits(output, x1.float())
        # Logging
        self.log('val/loss', loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        return 0, {}
    
    def predict_step(
        self,
        previous_frame: torch.Tensor,
        num_inference_steps: int = None,
        generator: Optional[torch.Generator] = None,
        volume_limit: Optional[float] = None,
        class_labels: Optional[torch.Tensor] = None,
        # --- NEW CONTROL ARGUMENTS ---
        temperature: float = 1.0,  # Logit temperature scaling
        sampling_mode: str = 'stochastic',  # Options: 'greedy', 'stochastic', 'hybrid'
        confidence_threshold: float = 0.2,  # Only used for 'hybrid' mode
        force_binary_last_step: bool = True, # "Last step thresholding" toggle
        output_probabilistic: bool = False, # If True, returns p_next at last step instead of binary sample
        volume_limit_mode: str = 'final',  # 'final' = enforce at end, 'progressive' = enforce each step
        **kwargs
    ) -> torch.Tensor:
        """
        Generate topology using Discrete Flow Matching with selectable sampling strategies.

        Args:
            sampling_mode (str):
                - 'greedy': Deterministic. Thresholds p > 0.5 at every step. (Best for stability)
                - 'stochastic': Standard BFM. Probabilistic flipping. (Best for diversity/texture)
                - 'hybrid': Deterministic for confident pixels, Stochastic for uncertain ones.
            confidence_threshold (float):
                Used in 'hybrid' mode. Pixels with p > threshold or p < (1-threshold) are treated deterministically.
            force_binary_last_step (bool):
                If True, forces a hard threshold (Argmax) at the very last step regardless of mode.
                Recommended True for Topology Optimization to ensure valid masks.
            output_probabilistic (bool):
                If True, returns the continuous probabilities (p_next) at the last step instead of
                performing a Bernoulli flip or thresholding.
            volume_limit_mode (str):
                - 'final': Enforce volume constraint only at the end (original behavior)
                - 'progressive': Enforce at each sampling step by removing low-logit material
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_sampling_steps
            
        B = previous_frame.shape[0]
        H, W = previous_frame.shape[2], previous_frame.shape[3]
        image_shape = (B, 1, H, W)
        device = previous_frame.device
        
        # 1. Start from Prior
        # Always uniform noise to ensure correct Hellinger interpolation start
        pi_0 = self._sample_prior(image_shape, device)
        # Ensure pi_0 is valid probability for interpolation
        pi_0 = self._clamp_prob(pi_0)
        
        # Initial State: 
        # Start stochastic to allow diverse initialization paths
        x_current = torch.bernoulli(pi_0) 
        
        # 2. Time setup
        dt = 1.0 / num_inference_steps
        timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
        
        if previous_frame.dim() == 3: 
            previous_frame = previous_frame.unsqueeze(1)
        #previous_frame = previous_frame- 0.3
        # 3. The Flow Loop
        for i in range(num_inference_steps):
            t = timesteps[i]
            t_next = t + dt
            is_last_step = (i == num_inference_steps - 1)
            
            # --- A. Model Prediction ---
            t_input = torch.full((B,), t.item(), device=device)
            input_tensor = torch.cat([x_current, previous_frame], dim=1)
            
            logits = self.forward(input_tensor, t_input, class_labels=class_labels)
            logits = logits.sample if hasattr(logits, 'sample') else logits
            
            # Apply Temperature Scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            pred_x1_probs = torch.sigmoid(logits)
            pred_x1_probs = self._clamp_prob(pred_x1_probs)  # Clamp for numerical stability
            
            # --- B. Compute Trajectory (Hellinger Path) ---
            pi_0_clamped = self._clamp_prob(pi_0)  # Ensure pi_0 is valid for sqrt
            sqrt_pi_0 = torch.sqrt(pi_0_clamped)
            sqrt_x1 = torch.sqrt(pred_x1_probs)
            
            # Current probability (needed for stochastic math)
            #sqrt_pt = (1 - t) * sqrt_pi_0 + t * sqrt_x1
            #pt = self._clamp_prob(sqrt_pt ** 2)
            pt = (1-t) * pi_0_clamped + t*pred_x1_probs
            pt = self._clamp_prob(pt)  # Ensure probabilities stay in valid range
            # Next target probability
            #sqrt_p_next = (1 - t_next) * sqrt_pi_0 + t_next * sqrt_x1
            #p_next = self._clamp_prob(sqrt_p_next ** 2)
            p_next = (1-t_next) * pi_0_clamped + t_next*pred_x1_probs
            p_next = self._clamp_prob(p_next)  # Ensure probabilities stay in valid range

            # --- C. Update Logic ---
            
            # 1. Check for "Last Step Override"
            if is_last_step:
                if output_probabilistic:
                    x_current = pred_x1_probs #p_next

                    continue
                elif force_binary_last_step:
                    x_current = (p_next > 0.5).float()
                    continue 
            
            # 2. Compute Updates based on Sampling Mode
            
            if sampling_mode == 'greedy':
                # Deterministic: Snap to the curve
                x_current = (p_next > 0.5).float()
                
            elif sampling_mode == 'stochastic':
                # Standard BFM: Full probabilistic transition
                x_current = self._apply_stochastic_transition(x_current, pt, p_next)
                
            elif sampling_mode == 'hybrid':
                # Hybrid: Mix of Greedy (Safe) and Stochastic (Unsafe)
                
                # Identify "Confident" pixels
                is_confident = (p_next > confidence_threshold) | (p_next < (1.0 - confidence_threshold))
                
                # Calculate both potential outcomes
                x_det = (p_next > 0.5).float()
                x_stoch = self._apply_stochastic_transition(x_current, pt, p_next)
                
                # Apply deterministic update to confident pixels, stochastic to others
                x_current = torch.where(is_confident, x_det, x_stoch)
                
            else:
                raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
            
            # --- D. Progressive Volume Constraint (optional) ---
            # Remove material at each step based on logit values
            if volume_limit is not None and volume_limit_mode == 'progressive':
                x_current = self._enforce_volume_constraint_progressive(
                    x_current, logits, volume_limit
                )

        # 4. Final Output
        output = x_current
        
        # Optional: Hard Volume Enforcement (final mode)
        if volume_limit is not None and volume_limit_mode == 'final':
             output = self._enforce_volume_constraint(pred_x1_probs, volume_limit)
             output = (output > 0.5).float()

        return output

    def _apply_stochastic_transition(self, x_curr, p_curr, p_next):
        """
        Helper method for Standard Bernoulli Flow Matching transition logic.
        Calculates P(x_next | x_curr) and samples the update.
        """
        # Case 1: Probability Increasing (flip 0 -> 1)
        # When p_curr ~ 1, denominator becomes ~0, causing issues
        denom_flip = (1 - p_curr).clamp(min=1e-6)
        flip_0_to_1 = (p_next - p_curr) / denom_flip
        flip_0_to_1 = torch.clamp(flip_0_to_1, 0.0, 1.0)
        # Replace any NaN with 0 (no flip)
        flip_0_to_1 = torch.where(torch.isnan(flip_0_to_1), torch.zeros_like(flip_0_to_1), flip_0_to_1)
        
        # Case 2: Probability Decreasing (keep 1)
        # When p_curr ~ 0, denominator becomes ~0, causing issues
        denom_keep = p_curr.clamp(min=1e-6)
        keep_1 = p_next / denom_keep
        keep_1 = torch.clamp(keep_1, 0.0, 1.0)
        # Replace any NaN with 1 (keep the 1)
        keep_1 = torch.where(torch.isnan(keep_1), torch.ones_like(keep_1), keep_1)
        
        # Sample masks
        mask_flip_up = torch.bernoulli(flip_0_to_1)
        mask_keep = torch.bernoulli(keep_1)
        
        # Apply Logic
        # If 0: flip if mask says so
        # If 1: keep if mask says so, else become 0
        x_new_from_0 = mask_flip_up
        x_new_from_1 = mask_keep
        
        return torch.where(x_curr == 0, x_new_from_0, x_new_from_1)   
    def _enforce_volume_constraint(
        self,
        rho: torch.Tensor,
        volume_limit: float
    ) -> torch.Tensor:
        """
        Enforce volume constraint at the end by thresholding based on density values.
        Keeps the top `volume_limit` fraction of pixels by density value.
        """
        B = rho.shape[0]
        result = rho.clone()
        
        for b in range(B):
            current_vol = rho[b].mean().item()
            if current_vol > volume_limit:
                threshold = torch.quantile(rho[b].flatten(), 1 - volume_limit)
                result[b] = (rho[b] > threshold).float()
        
        return result
    
    def _enforce_volume_constraint_progressive(
        self,
        x_binary: torch.Tensor,
        logits: torch.Tensor,
        volume_limit: float
    ) -> torch.Tensor:
        """
        Enforce volume constraint progressively at each sampling step.
        
        After Bernoulli sampling, if volume exceeds limit, remove material 
        pixels with the lowest logit values until we reach the target fraction.
        
        Pseudocode:
            1. Check current volume fraction (mean of binary mask)
            2. If volume > limit:
                a. Get logit values only where material exists (x == 1)
                b. Find threshold logit such that keeping pixels above it
                   gives exactly volume_limit fraction
                c. Remove material (set to 0) for pixels below threshold
            3. Return constrained binary mask
        
        Args:
            x_binary: Current binary state (0 or 1), shape (B, 1, H, W)
            logits: Raw logit predictions from model, shape (B, 1, H, W)
            volume_limit: Target maximum volume fraction (e.g., 0.3)
        
        Returns:
            Constrained binary mask with volume <= volume_limit
        """
        B = x_binary.shape[0]
        result = x_binary.clone()
        
        for b in range(B):
            # Current volume = fraction of pixels that are 1
            current_vol = x_binary[b].mean().item()
            
            if current_vol > volume_limit:
                # Get mask of where material currently exists
                material_mask = (x_binary[b] == 1)
                
                # Get logit values at material locations
                material_logits = logits[b][material_mask]
                
                # How many pixels to keep? 
                # We want final volume = volume_limit
                # total_pixels * volume_limit = num_to_keep
                total_pixels = x_binary[b].numel()
                num_to_keep = int(volume_limit * total_pixels)
                
                if num_to_keep > 0 and material_logits.numel() > num_to_keep:
                    # Find threshold: keep top num_to_keep by logit value
                    # Sort descending, take the num_to_keep-th value as threshold
                    sorted_logits, _ = torch.sort(material_logits, descending=True)
                    threshold = sorted_logits[num_to_keep - 1].item()
                    
                    # Keep only pixels where logit >= threshold
                    # (among the material pixels)
                    keep_mask = (logits[b] >= threshold) & material_mask
                    result[b] = keep_mask.float()
                elif num_to_keep == 0:
                    result[b] = torch.zeros_like(x_binary[b])
                # else: keep all material (shouldn't happen if current_vol > limit)
        
        return result
    
    def predict(
        self,
        batch,
        device,
        num_frames: int = 20,
        generator: Optional[torch.Generator] = None,
        output_type: str = 'numpy',
        num_inference_steps: int = 100,
        return_dict: bool = True,
        batch_dim: bool = True,
        volume_limit: Optional[float] = None,
        **kwargs
    ):
        """
        Generate predictions compatible with the pipeline.
        
        Returns:
            vid: Generated frames [B, num_frames+1, C, H, W]
            reference: Ground truth reference [B, num_frames+1, C, H, W]
        """
        with torch.no_grad():
            input_0, input_1, labels = self.get_input(batch, batch_dim=batch_dim)
            
            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            labels = labels.to(device)
            
            if generator is None:
                generator = torch.Generator(device=input_0.device).manual_seed(2024)
            
            frames = [input_0.cpu()[..., -1:, :, :]]
            previous_frame = input_0
            
            for _ in tqdm(range(num_frames), desc="Generating frames"):
                x0 = self.predict_step(
                    previous_frame,
                    generator=generator,
                    class_labels=labels,
                    num_inference_steps=num_inference_steps,
                    volume_limit=volume_limit,
                    **kwargs
                )
                previous_frame = x0
                frames.append(x0.cpu())
            
            vid = np.array([frame[..., -1:, :, :].numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)
            
            reference = np.array(torch.concat([
                frames[0].unsqueeze(1),
                input_1.cpu()[..., -1:, :, :]
            ], dim=1))
            
            if not batch_dim:
                vid = vid[0]
                reference = reference[0]
        
        return vid, reference
    
    def configure_optimizers(self):
        """Configure optimizer."""
        if self.optimizer_name == 'adamw':
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
            )
        elif self.optimizer_name == 'adam':
            opt = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
