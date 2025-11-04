"""
degradation_control.py

Spatiotemporal degradation control for Go-with-the-Flow.
Provides flexible degradation patterns for motion control strength.
"""

import numpy as np
import torch
import cv2
import json
from pathlib import Path
from typing import Union, Optional, Literal, Tuple
from dataclasses import dataclass, asdict

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DegradationConfig:
    """
    Configuration for spatiotemporal degradation control.

    Modes:
        - 'scalar': Single value applied uniformly
        - 'temporal': Per-frame schedule
        - 'spatial': Per-pixel mask (constant across time)
        - 'spatiotemporal': Full per-pixel per-frame control
    """
    mode: Literal['scalar', 'temporal', 'spatial', 'spatiotemporal']
    scalar_value: Optional[float] = None
    temporal_schedule: Optional[np.ndarray] = None  # Shape: (T,)
    spatial_mask: Optional[np.ndarray] = None  # Shape: (H, W)
    spatiotemporal_mask: Optional[np.ndarray] = None  # Shape: (T, H, W)
    interpolation_method: str = 'bilinear'
    cache_interpolated: bool = True
    dtype: str = 'float32'

    def validate(self):
        """Validate configuration."""
        if self.mode == 'scalar':
            assert self.scalar_value is not None, "scalar_value required for scalar mode"
            assert 0 <= self.scalar_value <= 1, "scalar_value must be in [0, 1]"
        elif self.mode == 'temporal':
            assert self.temporal_schedule is not None, "temporal_schedule required"
        elif self.mode == 'spatial':
            assert self.spatial_mask is not None, "spatial_mask required"
        elif self.mode == 'spatiotemporal':
            assert self.spatiotemporal_mask is not None, "spatiotemporal_mask required"

# ============================================================================
# PROCESSOR
# ============================================================================

class DegradationProcessor:
    """Prepares degradation tensors with efficient broadcasting."""

    def __init__(
        self,
        config: DegradationConfig,
        target_shape: Tuple[int, int, int, int],  # (B, T, C, H, W) or (T, C, H, W)
    ):
        """
        Args:
            config: Degradation configuration
            target_shape: Target noise shape (B, T, C, H, W) or (T, C, H, W)
        """
        self.config = config
        self.config.validate()
        self.target_shape = target_shape

        # Handle both (B, T, C, H, W) and (T, C, H, W) shapes
        if len(target_shape) == 5:
            self.B, self.T, self.C, self.H, self.W = target_shape
            self.has_batch = True
        else:
            self.T, self.C, self.H, self.W = target_shape
            self.B = 1
            self.has_batch = False

        self._cache = None

    def prepare_degradation_tensor(self, device: str = 'cuda') -> torch.Tensor:
        """
        Prepare degradation tensor with minimal shape for broadcasting.

        Returns:
            Tensor with shape:
            - scalar: () → broadcasts to all
            - temporal: (B, T, 1, 1, 1) or (T, 1, 1, 1) → broadcasts to C, H, W
            - spatial: (B, 1, 1, H, W) or (1, 1, H, W) → broadcasts to T, C
            - spatiotemporal: (B, T, 1, H, W) or (T, 1, H, W) → broadcasts to C
        """
        if self.config.cache_interpolated and self._cache is not None:
            return self._cache.to(device)

        if self.config.mode == 'scalar':
            tensor = torch.tensor(
                self.config.scalar_value,
                dtype=getattr(torch, self.config.dtype),
                device=device
            )

        elif self.config.mode == 'temporal':
            schedule = self._interpolate_temporal(
                self.config.temporal_schedule,
                self.T
            )
            tensor = torch.from_numpy(schedule).to(device)
            if self.has_batch:
                tensor = tensor.view(1, self.T, 1, 1, 1)  # Add broadcast dimensions
            else:
                tensor = tensor.view(self.T, 1, 1, 1)

        elif self.config.mode == 'spatial':
            mask = self._interpolate_spatial(
                self.config.spatial_mask,
                self.H,
                self.W
            )
            tensor = torch.from_numpy(mask).to(device)
            if self.has_batch:
                tensor = tensor.view(1, 1, 1, self.H, self.W)  # Add broadcast dimensions
            else:
                tensor = tensor.view(1, 1, self.H, self.W)

        elif self.config.mode == 'spatiotemporal':
            mask = self._interpolate_spatiotemporal(
                self.config.spatiotemporal_mask,
                self.T,
                self.H,
                self.W
            )
            tensor = torch.from_numpy(mask).to(device)
            if self.has_batch:
                tensor = tensor.view(1, self.T, 1, self.H, self.W)  # Add channel broadcast
            else:
                tensor = tensor.view(self.T, 1, self.H, self.W)

        # Cache if requested
        if self.config.cache_interpolated:
            self._cache = tensor.cpu()

        return tensor

    def _interpolate_temporal(
        self,
        schedule: np.ndarray,
        target_frames: int
    ) -> np.ndarray:
        """Interpolate temporal schedule to target number of frames."""
        if len(schedule) == target_frames:
            return schedule.astype(self.config.dtype)

        from scipy.interpolate import interp1d

        x_orig = np.linspace(0, 1, len(schedule))
        x_new = np.linspace(0, 1, target_frames)

        kind = 'linear' if self.config.interpolation_method == 'bilinear' else 'nearest'
        interp_func = interp1d(x_orig, schedule, kind=kind)

        return interp_func(x_new).astype(self.config.dtype)

    def _interpolate_spatial(
        self,
        mask: np.ndarray,
        target_height: int,
        target_width: int
    ) -> np.ndarray:
        """Interpolate spatial mask to target dimensions."""
        if mask.shape == (target_height, target_width):
            return mask.astype(self.config.dtype)

        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC
        }

        return cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=interp_map.get(
                self.config.interpolation_method,
                cv2.INTER_LINEAR
            )
        ).astype(self.config.dtype)

    def _interpolate_spatiotemporal(
        self,
        mask: np.ndarray,
        target_frames: int,
        target_height: int,
        target_width: int
    ) -> np.ndarray:
        """Interpolate spatiotemporal mask to target dimensions."""
        current_frames, current_height, current_width = mask.shape

        # First interpolate spatially
        if (current_height, current_width) != (target_height, target_width):
            mask_resized = np.zeros(
                (current_frames, target_height, target_width),
                dtype=self.config.dtype
            )
            for t in range(current_frames):
                mask_resized[t] = self._interpolate_spatial(
                    mask[t],
                    target_height,
                    target_width
                )
            mask = mask_resized

        # Then interpolate temporally
        if current_frames != target_frames:
            mask_resampled = np.zeros(
                (target_frames, target_height, target_width),
                dtype=self.config.dtype
            )
            for h in range(target_height):
                for w in range(target_width):
                    mask_resampled[:, h, w] = self._interpolate_temporal(
                        mask[:, h, w],
                        target_frames
                    )
            mask = mask_resampled

        return mask

# ============================================================================
# MAIN APPLICATION FUNCTION
# ============================================================================

def apply_spatiotemporal_degradation(
    sample_noise: torch.Tensor,
    random_noise: torch.Tensor,
    degradation_config: DegradationConfig,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Apply spatiotemporal degradation to warped noise.

    Formula: output = (1 - λ) × warped + λ × random
    where λ can vary spatially and/or temporally

    Args:
        sample_noise: Warped noise, shape (B, T, C, H, W) or (T, C, H, W)
        random_noise: Random Gaussian noise, same shape as sample_noise
        degradation_config: Configuration object
        device: 'cuda' or 'cpu'

    Returns:
        Degraded noise, same shape as input
    """
    # Initialize processor
    processor = DegradationProcessor(
        config=degradation_config,
        target_shape=sample_noise.shape
    )

    # Get degradation tensor (efficiently broadcasts)
    degradation_tensor = processor.prepare_degradation_tensor(device=device)

    # Apply degradation formula with broadcasting
    output = (
        sample_noise * (1 - degradation_tensor) +
        random_noise * degradation_tensor
    )

    return output

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_temporal_schedule(
    num_frames: int,
    schedule_type: str = 'linear',
    start_value: float = 0.0,
    end_value: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Create temporal degradation schedules.

    Args:
        num_frames: Number of frames (typically 49 or 13)
        schedule_type: 'linear', 'exponential', 'cosine', 'constant', 'pulse', 'sinusoidal'
        start_value: Starting degradation [0, 1]
        end_value: Ending degradation [0, 1]

    Returns:
        1D array of shape (num_frames,)

    Examples:
        # Gradual increase
        schedule = create_temporal_schedule(49, 'linear', 0.0, 1.0)

        # Exponential growth
        schedule = create_temporal_schedule(49, 'exponential', 0.0, 1.0, rate=2.0)

        # Smooth S-curve
        schedule = create_temporal_schedule(49, 'cosine', 0.0, 1.0)

        # Pulse at specific frames
        schedule = create_temporal_schedule(49, 'pulse',
                                          pulse_frames=[10, 20, 30],
                                          pulse_value=1.0)
    """
    if schedule_type == 'linear':
        return np.linspace(start_value, end_value, num_frames, dtype=np.float32)

    elif schedule_type == 'exponential':
        rate = kwargs.get('rate', 2.0)
        x = np.linspace(0, 1, num_frames)
        schedule = start_value + (end_value - start_value) * (x ** rate)
        return schedule.astype(np.float32)

    elif schedule_type == 'cosine':
        # Smooth S-curve using cosine
        x = np.linspace(0, 1, num_frames)
        schedule = start_value + (end_value - start_value) * (1 - np.cos(x * np.pi)) / 2
        return schedule.astype(np.float32)

    elif schedule_type == 'constant':
        return np.full(num_frames, start_value, dtype=np.float32)

    elif schedule_type == 'pulse':
        pulse_frames = kwargs.get('pulse_frames', [num_frames // 2])
        pulse_value = kwargs.get('pulse_value', 1.0)
        pulse_width = kwargs.get('pulse_width', 1)

        schedule = np.full(num_frames, start_value, dtype=np.float32)
        for frame in pulse_frames:
            start_idx = max(0, frame - pulse_width // 2)
            end_idx = min(num_frames, frame + pulse_width // 2 + 1)
            schedule[start_idx:end_idx] = pulse_value

        return schedule

    elif schedule_type == 'sinusoidal':
        frequency = kwargs.get('frequency', 1.0)
        x = np.linspace(0, frequency * 2 * np.pi, num_frames)
        schedule = start_value + (end_value - start_value) * (np.sin(x) + 1) / 2
        return schedule.astype(np.float32)

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

def create_spatial_mask(
    height: int,
    width: int,
    mask_type: str = 'radial',
    **kwargs
) -> np.ndarray:
    """
    Generate common spatial mask patterns.

    Args:
        height: Mask height
        width: Mask width
        mask_type: 'uniform', 'radial', 'gradient', 'rectangle', 'ellipse'

    Returns:
        Spatial mask of shape (height, width)

    Examples:
        # Center-focused radial mask
        mask = create_spatial_mask(60, 90, 'radial',
                                   inner_value=1.0, outer_value=0.0)

        # Horizontal gradient
        mask = create_spatial_mask(60, 90, 'gradient',
                                   direction='horizontal')
    """
    if mask_type == 'uniform':
        value = kwargs.get('value', 0.5)
        return np.full((height, width), value, dtype=np.float32)

    elif mask_type == 'radial':
        center_x = kwargs.get('center_x', width / 2)
        center_y = kwargs.get('center_y', height / 2)
        max_radius = kwargs.get('max_radius', min(height, width) / 2)
        inner_value = kwargs.get('inner_value', 0.0)
        outer_value = kwargs.get('outer_value', 1.0)

        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        normalized = np.clip(distances / max_radius, 0, 1)
        mask = inner_value + (outer_value - inner_value) * normalized

        return mask.astype(np.float32)

    elif mask_type == 'gradient':
        direction = kwargs.get('direction', 'horizontal')
        start_value = kwargs.get('start_value', 0.0)
        end_value = kwargs.get('end_value', 1.0)

        if direction == 'horizontal':
            mask = np.linspace(start_value, end_value, width)[None, :]
            mask = np.repeat(mask, height, axis=0)
        else:  # vertical
            mask = np.linspace(start_value, end_value, height)[:, None]
            mask = np.repeat(mask, width, axis=1)

        return mask.astype(np.float32)

    elif mask_type == 'rectangle':
        x1 = kwargs.get('x1', width // 4)
        y1 = kwargs.get('y1', height // 4)
        x2 = kwargs.get('x2', 3 * width // 4)
        y2 = kwargs.get('y2', 3 * height // 4)
        inside_value = kwargs.get('inside_value', 1.0)
        outside_value = kwargs.get('outside_value', 0.0)

        mask = np.full((height, width), outside_value, dtype=np.float32)
        mask[y1:y2, x1:x2] = inside_value

        return mask

    elif mask_type == 'ellipse':
        center_x = kwargs.get('center_x', width / 2)
        center_y = kwargs.get('center_y', height / 2)
        radius_x = kwargs.get('radius_x', width / 4)
        radius_y = kwargs.get('radius_y', height / 4)
        inside_value = kwargs.get('inside_value', 1.0)
        outside_value = kwargs.get('outside_value', 0.0)

        y, x = np.ogrid[:height, :width]
        ellipse_mask = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1

        mask = np.full((height, width), outside_value, dtype=np.float32)
        mask[ellipse_mask] = inside_value

        return mask

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

def load_spatial_mask(
    mask_path: str,
    target_height: int,
    target_width: int,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Load and resize spatial degradation mask.

    Args:
        mask_path: Path to mask file (.npy, .npz, or image)
        target_height: Target height in latent space (e.g., 60)
        target_width: Target width in latent space (e.g., 90)
        interpolation: 'bilinear', 'nearest', 'cubic'

    Returns:
        Spatial mask array of shape (target_height, target_width)
    """
    # Load mask based on file type
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    elif mask_path.endswith('.npz'):
        mask = np.load(mask_path)['mask']
    else:  # Image file
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask from {mask_path}")
        mask = mask.astype(np.float32) / 255.0

    # Resize to target dimensions
    if mask.shape != (target_height, target_width):
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC
        }
        mask = cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=interp_map[interpolation]
        )

    # Ensure values in [0, 1]
    mask = np.clip(mask, 0.0, 1.0)

    return mask

def load_temporal_schedule(
    schedule_path: str,
    target_frames: int,
    interpolation: str = 'linear'
) -> np.ndarray:
    """
    Load and interpolate temporal schedule to target number of frames.

    Args:
        schedule_path: Path to .npy file containing schedule
        target_frames: Target number of frames (e.g., 49 or 13)
        interpolation: 'linear', 'nearest', or 'cubic'

    Returns:
        Temporal schedule of shape (target_frames,)
    """
    schedule = np.load(schedule_path)

    if len(schedule) != target_frames:
        # Interpolate to target length
        from scipy.interpolate import interp1d
        x_orig = np.linspace(0, 1, len(schedule))
        x_new = np.linspace(0, 1, target_frames)

        interp_func = interp1d(x_orig, schedule, kind=interpolation)
        schedule = interp_func(x_new)

    return schedule.astype(np.float32)

# ============================================================================
# I/O UTILITIES
# ============================================================================

class DegradationIO:
    """Save/load degradation configurations."""

    @staticmethod
    def save_config(config: DegradationConfig, path: str):
        """Save configuration to JSON + NPY/NPZ."""
        path = Path(path)

        # Save metadata as JSON
        metadata = {
            'mode': config.mode,
            'interpolation_method': config.interpolation_method,
            'dtype': config.dtype,
        }

        if config.mode == 'scalar':
            metadata['scalar_value'] = float(config.scalar_value)
        elif config.mode == 'temporal':
            schedule_path = path.with_suffix('.schedule.npy')
            np.save(schedule_path, config.temporal_schedule)
            metadata['temporal_schedule_path'] = str(schedule_path)
        elif config.mode == 'spatial':
            mask_path = path.with_suffix('.mask.npy')
            np.save(mask_path, config.spatial_mask)
            metadata['spatial_mask_path'] = str(mask_path)
        elif config.mode == 'spatiotemporal':
            mask_path = path.with_suffix('.stmask.npz')
            np.savez_compressed(mask_path, mask=config.spatiotemporal_mask)
            metadata['spatiotemporal_mask_path'] = str(mask_path)

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_config(path: str) -> DegradationConfig:
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            metadata = json.load(f)

        mode = metadata['mode']

        if mode == 'scalar':
            return DegradationConfig(
                mode='scalar',
                scalar_value=metadata['scalar_value']
            )
        elif mode == 'temporal':
            schedule = np.load(metadata['temporal_schedule_path'])
            return DegradationConfig(
                mode='temporal',
                temporal_schedule=schedule,
                interpolation_method=metadata.get('interpolation_method', 'bilinear')
            )
        elif mode == 'spatial':
            mask = np.load(metadata['spatial_mask_path'])
            return DegradationConfig(
                mode='spatial',
                spatial_mask=mask,
                interpolation_method=metadata.get('interpolation_method', 'bilinear')
            )
        elif mode == 'spatiotemporal':
            mask = np.load(metadata['spatiotemporal_mask_path'])['mask']
            return DegradationConfig(
                mode='spatiotemporal',
                spatiotemporal_mask=mask,
                interpolation_method=metadata.get('interpolation_method', 'bilinear')
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
