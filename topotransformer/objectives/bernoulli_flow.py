"""
Bernoulli Flow Matching (α=0 / Fisher-Rao / Hellinger geometry)

This module implements flow matching for binary data using the Fisher-Rao metric
on Bernoulli distributions. The key insight is that Fisher-Rao geodesics on the
probability simplex correspond to great circles on the "root-probability" sphere.

Mathematical Background:
------------------------
For Bernoulli distributions with parameter π ∈ (0,1), the Fisher information metric is:
    g(π) = 1 / (π(1-π))

Under the transformation z = √π (embedding on the positive orthant of a sphere),
Fisher-Rao geodesics become great circles, and the geodesic between π₀ and π₁ is:
    π_t = ((1-t)√π₀ + t√π₁)²

The Hellinger velocity (velocity in root-probability space) is constant:
    w = d/dt √π_t = √π₁ - √π₀

This leads to a simple MSE loss on the Hellinger velocity, avoiding the numerical
instability of weighting by 1/(π(1-π)).

References:
-----------
- Atkinson et al., "On closed-form expressions for the Fisher-Rao distance" (2024)
- Amari, "Information Geometry and Its Applications" (2016)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import math


class BernoulliFlowObjective(nn.Module):
    """
    Bernoulli Flow Matching objective using Fisher-Rao (Hellinger) geometry.
    
    The network predicts the Hellinger velocity w = d/dt √π, which is constant
    along Fisher-Rao geodesics. This avoids numerical instability from the
    Fisher information weighting.
    
    Args:
        eps: Small constant for numerical stability (clamping probabilities)
        prior_type: Type of prior distribution ('uniform', 'beta', 'volume_fraction')
        prior_alpha: Alpha parameter for Beta prior (if prior_type='beta')
        prior_beta: Beta parameter for Beta prior (if prior_type='beta')
        volume_fraction: Target volume fraction for volume_fraction prior
        volume_fraction_std: Standard deviation for volume fraction prior
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
        prior_type: Literal['uniform', 'beta', 'volume_fraction'] = 'uniform',
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        volume_fraction: float = 0.5,
        volume_fraction_std: float = 0.1,
    ):
        super().__init__()
        self.eps = eps
        self.prior_type = prior_type
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.volume_fraction = volume_fraction
        self.volume_fraction_std = volume_fraction_std
        
    def _clamp_prob(self, p: torch.Tensor) -> torch.Tensor:
        """Clamp probabilities to [eps, 1-eps] for numerical stability."""
        return p.clamp(self.eps, 1.0 - self.eps)
    
    def _sample_prior(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Sample prior probability map π₀.
        
        Args:
            shape: Shape of the output tensor
            device: Device to create tensor on
            
        Returns:
            Prior probability map π₀ ∈ (0, 1)
        """
        if self.prior_type == 'uniform':
            # Uniform prior: π₀ = 0.5 everywhere
            pi_0 = torch.full(shape, 0.5, device=device)
            
        elif self.prior_type == 'beta':
            # Beta distribution prior
            beta_dist = torch.distributions.Beta(
                torch.tensor(self.prior_alpha, device=device),
                torch.tensor(self.prior_beta, device=device)
            )
            pi_0 = beta_dist.sample(shape)
            
        elif self.prior_type == 'volume_fraction':
            # Volume fraction based prior with per-sample variation
            batch_size = shape[0]
            # Sample a volume fraction for each sample in the batch
            vf = torch.normal(
                mean=self.volume_fraction,
                std=self.volume_fraction_std,
                size=(batch_size, 1, 1, 1),
                device=device
            ).clamp(0.1, 0.9)
            pi_0 = vf.expand(shape)
            
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
            
        return self._clamp_prob(pi_0)
    
    def _fisher_rao_geodesic(
        self, 
        pi_0: torch.Tensor, 
        pi_1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fisher-Rao geodesic interpolation.
        
        The geodesic between π₀ and π₁ under the Fisher-Rao metric is:
            π_t = ((1-t)√π₀ + t√π₁)²
            
        This corresponds to linear interpolation in the root-probability space
        (great circle on the sphere).
        
        Args:
            pi_0: Starting probability map
            pi_1: Target probability map (can be binary)
            t: Interpolation time ∈ [0, 1]
            
        Returns:
            Interpolated probability map π_t
        """
        sqrt_pi_0 = torch.sqrt(self._clamp_prob(pi_0))
        sqrt_pi_1 = torch.sqrt(self._clamp_prob(pi_1))
        
        # Linear interpolation in root space
        sqrt_pi_t = (1 - t) * sqrt_pi_0 + t * sqrt_pi_1
        
        # Square to get probability
        pi_t = sqrt_pi_t ** 2
        
        return self._clamp_prob(pi_t)
    
    def _hellinger_velocity(
        self, 
        pi_0: torch.Tensor, 
        pi_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Hellinger velocity (velocity in root-probability space).
        
        The Hellinger velocity is constant along the geodesic:
            w = d/dt √π_t = √π₁ - √π₀
            
        Args:
            pi_0: Starting probability map
            pi_1: Target probability map
            
        Returns:
            Hellinger velocity w
        """
        sqrt_pi_0 = torch.sqrt(self._clamp_prob(pi_0))
        sqrt_pi_1 = torch.sqrt(self._clamp_prob(pi_1))
        
        return sqrt_pi_1 - sqrt_pi_0
    
    def loss(
        self,
        model: nn.Module,
        batch: dict,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute the Bernoulli Flow Matching loss.
        
        The loss is simple MSE on the Hellinger velocity:
            L = E[||ŵ_θ(π_t, t, c) - w_target||²]
            
        where w_target = √x₁ - √π₀
        
        Args:
            model: Neural network that predicts Hellinger velocity
            batch: Dictionary containing:
                - 'target': Binary ground truth x₁ ∈ {0,1}^{B,1,H,W}
                - 'input': Conditioning tensor c ∈ R^{B,C,H,W}
            return_components: If True, return loss components dict
            
        Returns:
            Loss tensor (and optionally components dict)
        """
        # Extract data from batch
        x1 = batch['target']  # Binary ground truth [B, 1, H, W]
        condition = batch['input']  # Conditioning [B, C, H, W]
        
        # Remove loading_metadata if present
        batch_for_model = {k: v for k, v in batch.items() if k != 'loading_metadata'}
        
        device = x1.device
        batch_size = x1.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Sample prior π₀
        pi_0 = self._sample_prior(x1.shape, device)
        
        # Clamp x1 to valid probability range (in case of truly binary data)
        x1_clamped = self._clamp_prob(x1.float())
        
        # Compute interpolated state π_t along Fisher-Rao geodesic
        pi_t = self._fisher_rao_geodesic(pi_0, x1_clamped, t)
        
        # Compute target Hellinger velocity
        w_target = self._hellinger_velocity(pi_0, x1_clamped)
        
        # Prepare input for model
        # The model receives π_t as the "noisy" state and predicts w
        model_input = batch_for_model.copy()
        model_input['target'] = pi_t  # Current state
        model_input['time'] = t.squeeze()  # Time [B]
        
        # Model prediction
        w_pred = model(model_input)
        
        # MSE loss on Hellinger velocity
        loss = ((w_pred - w_target) ** 2).mean()
        
        if return_components:
            components = {
                'loss': loss,
                'w_pred_norm': w_pred.abs().mean(),
                'w_target_norm': w_target.abs().mean(),
                'pi_t_mean': pi_t.mean(),
            }
            return loss, components
        
        return loss


class BernoulliFlowMatcher:
    """
    Complete Bernoulli Flow Matching module for training and sampling.
    
    This class wraps the objective and provides sampling functionality.
    
    Example:
        >>> matcher = BernoulliFlowMatcher(eps=1e-6, prior_type='uniform')
        >>> 
        >>> # Training
        >>> loss = matcher.compute_loss(model, x1, condition)
        >>> 
        >>> # Sampling
        >>> samples = matcher.sample(model, shape=(16, 1, 64, 64), condition=cond, steps=50)
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
        prior_type: Literal['uniform', 'beta', 'volume_fraction'] = 'uniform',
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        volume_fraction: float = 0.5,
        volume_fraction_std: float = 0.1,
    ):
        self.eps = eps
        self.prior_type = prior_type
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.volume_fraction = volume_fraction
        self.volume_fraction_std = volume_fraction_std
        
        self.objective = BernoulliFlowObjective(
            eps=eps,
            prior_type=prior_type,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            volume_fraction=volume_fraction,
            volume_fraction_std=volume_fraction_std,
        )
    
    def _clamp_prob(self, p: torch.Tensor) -> torch.Tensor:
        """Clamp probabilities to [eps, 1-eps] for numerical stability."""
        return p.clamp(self.eps, 1.0 - self.eps)
    
    def _sample_prior(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample prior probability map."""
        return self.objective._sample_prior(shape, device)
    
    def compute_loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        condition: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Bernoulli Flow Matching loss.
        
        Args:
            model: Neural network that takes (pi_t, t, condition) and outputs w_pred
                   Expected signature: model(pi_t, t, condition) -> w_pred
            x1: Binary ground truth [B, 1, H, W]
            condition: Conditioning tensor [B, C, H, W]
            t: Optional time tensor [B, 1, 1, 1]. If None, sampled uniformly.
            
        Returns:
            MSE loss on Hellinger velocity
        """
        device = x1.device
        batch_size = x1.shape[0]
        
        # Sample time if not provided
        if t is None:
            t = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Sample prior π₀
        pi_0 = self._sample_prior(x1.shape, device)
        
        # Clamp x1 to valid probability range
        x1_clamped = self._clamp_prob(x1.float())
        
        # Compute interpolated state π_t
        sqrt_pi_0 = torch.sqrt(pi_0)
        sqrt_x1 = torch.sqrt(x1_clamped)
        sqrt_pi_t = (1 - t) * sqrt_pi_0 + t * sqrt_x1
        pi_t = self._clamp_prob(sqrt_pi_t ** 2)
        
        # Compute target Hellinger velocity
        w_target = sqrt_x1 - sqrt_pi_0
        
        # Model prediction
        w_pred = model(pi_t, t.squeeze(-1).squeeze(-1), condition)
        
        # MSE loss
        loss = ((w_pred - w_target) ** 2).mean()
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        steps: int = 50,
        return_trajectory: bool = False,
        threshold: Optional[float] = 0.5,
    ) -> torch.Tensor:
        """
        Sample from the learned Bernoulli flow using Euler integration.
        
        The ODE is: dπ/dt = 2√π · ŵ
        In root space: d√π/dt = ŵ
        
        Update rule: √π_{t+dt} = √π_t + dt · ŵ
                     π_{t+dt} = (√π_t + dt · ŵ)²
        
        Args:
            model: Trained model that predicts Hellinger velocity
            shape: Output shape [B, 1, H, W]
            condition: Conditioning tensor [B, C, H, W]
            steps: Number of Euler integration steps
            return_trajectory: If True, return all intermediate states
            threshold: If not None, threshold final output to binary
            
        Returns:
            Final probability map (or binary if threshold is set)
            If return_trajectory=True, also returns trajectory tensor
        """
        device = condition.device
        dt = 1.0 / steps
        
        # Initialize from prior
        pi = self._sample_prior(shape, device)
        sqrt_pi = torch.sqrt(pi)
        
        trajectory = [pi.clone()] if return_trajectory else None
        
        # Euler integration in root space
        for step in range(steps):
            t = torch.full((shape[0],), step * dt, device=device)
            
            # Predict Hellinger velocity
            w_pred = model(pi, t, condition)
            
            # Update in root space: √π_{t+dt} = √π_t + dt · ŵ
            sqrt_pi = sqrt_pi + dt * w_pred
            
            # Ensure non-negative (should be, but numerical safety)
            sqrt_pi = sqrt_pi.clamp(min=math.sqrt(self.eps))
            
            # Convert back to probability
            pi = self._clamp_prob(sqrt_pi ** 2)
            
            if return_trajectory:
                trajectory.append(pi.clone())
        
        # Optional thresholding to binary
        if threshold is not None:
            output = (pi > threshold).float()
        else:
            output = pi
            
        if return_trajectory:
            return output, torch.stack(trajectory, dim=1)
        return output
    
    @torch.no_grad()
    def sample_heun(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        steps: int = 25,
        threshold: Optional[float] = 0.5,
    ) -> torch.Tensor:
        """
        Sample using Heun's method (2nd order) for better accuracy.
        
        Args:
            model: Trained model
            shape: Output shape [B, 1, H, W]
            condition: Conditioning tensor [B, C, H, W]
            steps: Number of integration steps
            threshold: Threshold for binarization
            
        Returns:
            Final probability map or binary output
        """
        device = condition.device
        dt = 1.0 / steps
        
        # Initialize from prior
        pi = self._sample_prior(shape, device)
        sqrt_pi = torch.sqrt(pi)
        
        for step in range(steps):
            t = step * dt
            t_tensor = torch.full((shape[0],), t, device=device)
            t_next_tensor = torch.full((shape[0],), t + dt, device=device)
            
            # Euler prediction
            w1 = model(pi, t_tensor, condition)
            sqrt_pi_euler = sqrt_pi + dt * w1
            sqrt_pi_euler = sqrt_pi_euler.clamp(min=math.sqrt(self.eps))
            pi_euler = self._clamp_prob(sqrt_pi_euler ** 2)
            
            # Heun correction
            w2 = model(pi_euler, t_next_tensor, condition)
            sqrt_pi = sqrt_pi + 0.5 * dt * (w1 + w2)
            sqrt_pi = sqrt_pi.clamp(min=math.sqrt(self.eps))
            pi = self._clamp_prob(sqrt_pi ** 2)
        
        if threshold is not None:
            return (pi > threshold).float()
        return pi


class BernoulliFlowObjectiveV2(nn.Module):
    """
    Alternative Bernoulli Flow objective that predicts velocity in probability space.
    
    Instead of predicting w (Hellinger velocity), the network predicts:
        u = dπ/dt = 2√π · w
        
    This may be easier for certain architectures but requires more careful handling
    of the Fisher metric weighting in the loss.
    
    The loss uses inverse-variance weighting (Fisher information):
        L = E[1/(π(1-π)) · (v_pred - u_target)²]
        
    With soft clamping to avoid extreme weights.
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
        fisher_weight_min: float = 4.0,  # min weight = 1/(0.5*0.5) = 4
        fisher_weight_max: float = 1000.0,  # max weight clamp
        prior_type: Literal['uniform', 'beta', 'volume_fraction'] = 'uniform',
        volume_fraction: float = 0.5,
    ):
        super().__init__()
        self.eps = eps
        self.fisher_weight_min = fisher_weight_min
        self.fisher_weight_max = fisher_weight_max
        self.prior_type = prior_type
        self.volume_fraction = volume_fraction
        
    def _clamp_prob(self, p: torch.Tensor) -> torch.Tensor:
        return p.clamp(self.eps, 1.0 - self.eps)
    
    def _sample_prior(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self.prior_type == 'uniform':
            return torch.full(shape, 0.5, device=device)
        elif self.prior_type == 'volume_fraction':
            return torch.full(shape, self.volume_fraction, device=device)
        else:
            return torch.rand(shape, device=device).clamp(self.eps, 1.0 - self.eps)
    
    def _fisher_weight(self, pi: torch.Tensor) -> torch.Tensor:
        """Compute Fisher information weight 1/(π(1-π)) with clamping."""
        pi_clamped = self._clamp_prob(pi)
        weight = 1.0 / (pi_clamped * (1 - pi_clamped))
        return weight.clamp(self.fisher_weight_min, self.fisher_weight_max)
    
    def loss(
        self,
        model: nn.Module,
        batch: dict,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute Fisher-weighted MSE loss on probability velocity.
        """
        x1 = batch['target']
        device = x1.device
        batch_size = x1.shape[0]
        
        batch_for_model = {k: v for k, v in batch.items() if k != 'loading_metadata'}
        
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        pi_0 = self._sample_prior(x1.shape, device)
        x1_clamped = self._clamp_prob(x1.float())
        
        # Fisher-Rao geodesic
        sqrt_pi_0 = torch.sqrt(pi_0)
        sqrt_x1 = torch.sqrt(x1_clamped)
        sqrt_pi_t = (1 - t) * sqrt_pi_0 + t * sqrt_x1
        pi_t = self._clamp_prob(sqrt_pi_t ** 2)
        
        # Target velocity in probability space: u = dπ/dt = 2√π · (√x₁ - √π₀)
        u_target = 2 * sqrt_pi_t * (sqrt_x1 - sqrt_pi_0)
        
        # Model prediction
        model_input = batch_for_model.copy()
        model_input['target'] = pi_t
        model_input['time'] = t.squeeze()
        
        u_pred = model(model_input)
        
        # Fisher-weighted MSE
        fisher_weight = self._fisher_weight(pi_t)
        loss = (fisher_weight * (u_pred - u_target) ** 2).mean()
        
        if return_components:
            components = {
                'loss': loss,
                'unweighted_mse': ((u_pred - u_target) ** 2).mean(),
                'fisher_weight_mean': fisher_weight.mean(),
            }
            return loss, components
        
        return loss
