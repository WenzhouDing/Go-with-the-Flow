# Comprehensive Implementation Guide: Frequency-Decomposed Motion Transfer

## Overview
This implementation extends Go-with-the-Flow's motion transfer capabilities by adding frequency-domain decomposition of optical flow, enabling selective motion filtering (e.g., removing camera shake while preserving choreography). The approach leverages the existing RAFT integration and noise warping pipeline.

## Architecture Design

### Core Components

```python
# New file: CommonSource/frequency_motion_editor.py
class FrequencyMotionEditor:
    """
    Decomposes optical flow into frequency bands for selective motion editing.
    Uses 3D FFT (2D spatial + 1D temporal) for comprehensive motion analysis.
    """
    def __init__(self, flow_sequence, fps=30):
        self.flow_sequence = flow_sequence  # Shape: (T, 2, H, W)
        self.fps = fps
        self.freq_bands = None
        self.fft_flow = None
        
    def decompose_frequencies(self, num_bands=4):
        """Apply 3D FFT and segment into frequency bands"""
        
    def reconstruct_selective(self, band_weights):
        """Reconstruct flow using weighted frequency bands"""
```

## Implementation Steps

### Step 1: Extend Flow Extraction Pipeline

Modify `make_warped_noise.py` to store complete flow sequences:

```python
# In make_warped_noise.py, after line ~200 where flows are computed
def extract_full_flow_sequence(video_path, save_raw=True):
    """
    Extract and save complete optical flow sequence for frequency analysis.
    Returns both forward and backward flows.
    """
    # Load video and preprocess (existing code)
    frames = load_video(video_path)
    frames_resized = preprocess_frames(frames)
    
    # Extract flows using RAFT (existing)
    flows_forward = []
    flows_backward = []
    
    for i in range(len(frames_resized) - 1):
        # Use existing RAFT model
        with torch.no_grad():
            # Forward flow
            flow_f = model(frames_resized[i], frames_resized[i+1])[-1]
            flows_forward.append(flow_f)
            
            # Backward flow  
            flow_b = model(frames_resized[i+1], frames_resized[i])[-1]
            flows_backward.append(flow_b)
    
    # Stack into tensors
    flow_tensor = torch.stack(flows_forward)  # Shape: (T-1, 2, H, W)
    
    if save_raw:
        np.save('raw_flows_sequence.npy', flow_tensor.cpu().numpy())
    
    return flow_tensor
```

### Step 2: Implement Frequency Decomposition

Create `CommonSource/frequency_motion_editor.py`:

```python
import numpy as np
import torch
import torch.fft as fft
from scipy import signal
from typing import List, Tuple, Optional

class FrequencyMotionEditor:
    def __init__(self, flow_sequence: np.ndarray, fps: int = 30):
        """
        Initialize with optical flow sequence.
        
        Args:
            flow_sequence: Shape (T, 2, H, W) - T frames, 2 channels (u,v), H×W spatial
            fps: Video framerate for frequency calculations
        """
        self.flow_sequence = flow_sequence
        self.fps = fps
        self.T, _, self.H, self.W = flow_sequence.shape
        self.nyquist_freq = fps / 2.0
        
        # Compute FFT
        self.fft_flow = None
        self.freq_bands = []
        self.temporal_freqs = None
        
    def decompose_frequencies(self, num_bands: int = 4, 
                            band_type: str = 'logarithmic') -> List[np.ndarray]:
        """
        Decompose flow into frequency bands using 3D FFT.
        
        Args:
            num_bands: Number of frequency bands to create
            band_type: 'logarithmic' or 'linear' band spacing
            
        Returns:
            List of flow sequences for each frequency band
        """
        # Apply 3D FFT (temporal + 2D spatial)
        # Separate treatment for u,v components
        flow_u = self.flow_sequence[:, 0, :, :]  # Shape: (T, H, W)
        flow_v = self.flow_sequence[:, 1, :, :]
        
        # Temporal window to reduce edge artifacts
        window_t = signal.hann(self.T).reshape(-1, 1, 1)
        flow_u_windowed = flow_u * window_t
        flow_v_windowed = flow_v * window_t
        
        # 3D FFT
        fft_u = fft.rfftn(flow_u_windowed, axes=(0, 1, 2))
        fft_v = fft.rfftn(flow_v_windowed, axes=(0, 1, 2))
        
        self.fft_flow = np.stack([fft_u, fft_v], axis=1)
        
        # Create frequency bands
        temporal_freqs = fft.rfftfreq(self.T, d=1.0/self.fps)
        self.temporal_freqs = temporal_freqs
        
        # Define band boundaries
        if band_type == 'logarithmic':
            # Logarithmic spacing for better low-frequency resolution
            min_freq = 0.1  # Hz
            max_freq = self.nyquist_freq
            band_edges = np.logspace(np.log10(min_freq), 
                                    np.log10(max_freq), 
                                    num_bands + 1)
        else:
            # Linear spacing
            band_edges = np.linspace(0, self.nyquist_freq, num_bands + 1)
        
        # Extract bands
        self.freq_bands = []
        for i in range(num_bands):
            band_mask = self._create_band_mask(
                temporal_freqs, 
                band_edges[i], 
                band_edges[i+1]
            )
            
            # Apply mask to FFT
            band_fft_u = fft_u * band_mask[:, np.newaxis, np.newaxis]
            band_fft_v = fft_v * band_mask[:, np.newaxis, np.newaxis]
            
            # Inverse FFT to get band-limited flow
            band_flow_u = fft.irfftn(band_fft_u, s=(self.T, self.H, self.W))
            band_flow_v = fft.irfftn(band_fft_v, s=(self.T, self.H, self.W))
            
            band_flow = np.stack([band_flow_u.real, band_flow_v.real], axis=1)
            self.freq_bands.append(band_flow)
        
        return self.freq_bands
    
    def _create_band_mask(self, freqs: np.ndarray, 
                         low_freq: float, 
                         high_freq: float,
                         transition_width: float = 0.5) -> np.ndarray:
        """
        Create smooth band-pass filter mask with soft transitions.
        """
        mask = np.zeros_like(freqs)
        
        for i, f in enumerate(freqs):
            if low_freq <= f <= high_freq:
                mask[i] = 1.0
            elif f < low_freq:
                # Soft transition at low end
                if f > low_freq - transition_width:
                    mask[i] = 0.5 * (1 + np.cos(np.pi * (low_freq - f) / transition_width))
            elif f > high_freq:
                # Soft transition at high end
                if f < high_freq + transition_width:
                    mask[i] = 0.5 * (1 + np.cos(np.pi * (f - high_freq) / transition_width))
        
        return mask
    
    def reconstruct_selective(self, 
                            band_weights: List[float],
                            spatial_filter: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct flow using weighted combination of frequency bands.
        
        Args:
            band_weights: Weight for each frequency band (0.0 to 1.0)
            spatial_filter: Optional spatial mask (H, W) to apply local filtering
            
        Returns:
            Reconstructed flow sequence
        """
        assert len(band_weights) == len(self.freq_bands), \
            f"Need {len(self.freq_bands)} weights, got {len(band_weights)}"
        
        # Weighted sum of bands
        reconstructed = np.zeros_like(self.flow_sequence)
        for band, weight in zip(self.freq_bands, band_weights):
            if spatial_filter is not None:
                # Apply spatial filtering (e.g., to isolate dancer from background)
                band_filtered = band * spatial_filter[np.newaxis, np.newaxis, :, :]
                reconstructed += weight * band_filtered
            else:
                reconstructed += weight * band
        
        return reconstructed
    
    def analyze_motion_spectrum(self) -> dict:
        """
        Analyze motion characteristics in each frequency band.
        Returns statistics for intelligent band selection.
        """
        analysis = {}
        
        for i, band in enumerate(self.freq_bands):
            # Compute motion magnitude
            magnitude = np.sqrt(band[:, 0]**2 + band[:, 1]**2)
            
            analysis[f'band_{i}'] = {
                'mean_magnitude': float(np.mean(magnitude)),
                'std_magnitude': float(np.std(magnitude)),
                'max_magnitude': float(np.max(magnitude)),
                'temporal_variance': float(np.var(np.mean(magnitude, axis=(1, 2)))),
                'spatial_variance': float(np.var(np.mean(magnitude, axis=0))),
                'dominant_direction': self._compute_dominant_direction(band)
            }
        
        return analysis
    
    def _compute_dominant_direction(self, flow_band: np.ndarray) -> float:
        """Compute dominant motion direction in degrees."""
        mean_u = np.mean(flow_band[:, 0])
        mean_v = np.mean(flow_band[:, 1])
        angle = np.arctan2(mean_v, mean_u) * 180 / np.pi
        return float(angle)
```

### Step 3: Intelligent Frequency Band Selection

Create `CommonSource/motion_classifier.py`:

```python
import numpy as np
from typing import Dict, List, Tuple

class MotionClassifier:
    """
    Classify motion types based on frequency characteristics.
    Helps identify camera shake, body movement, micro-motion, etc.
    """
    
    @staticmethod
    def identify_camera_shake(freq_analysis: dict, 
                            band_idx: int) -> float:
        """
        Score likelihood that a band contains camera shake.
        Camera shake typically has high spatial uniformity and medium frequency.
        """
        band_stats = freq_analysis[f'band_{band_idx}']
        
        # High spatial uniformity (low spatial variance)
        spatial_uniformity = 1.0 / (1.0 + band_stats['spatial_variance'])
        
        # Medium to high temporal variance
        temporal_activity = band_stats['temporal_variance']
        
        # Moderate magnitude
        magnitude_score = 1.0 / (1.0 + abs(band_stats['mean_magnitude'] - 5.0))
        
        shake_score = spatial_uniformity * 0.5 + \
                     temporal_activity * 0.3 + \
                     magnitude_score * 0.2
        
        return shake_score
    
    @staticmethod
    def identify_body_movement(freq_analysis: dict, 
                              band_idx: int) -> float:
        """
        Score likelihood that a band contains primary body/object movement.
        Typically lower frequency with high spatial variance.
        """
        band_stats = freq_analysis[f'band_{band_idx}']
        
        # High spatial variance (localized movement)
        spatial_locality = band_stats['spatial_variance']
        
        # Consistent temporal pattern
        temporal_consistency = 1.0 / (1.0 + band_stats['temporal_variance'])
        
        # Higher magnitude
        magnitude_score = band_stats['mean_magnitude'] / 10.0
        
        movement_score = spatial_locality * 0.4 + \
                        temporal_consistency * 0.3 + \
                        magnitude_score * 0.3
        
        return min(movement_score, 1.0)
    
    @staticmethod
    def identify_micro_motion(freq_analysis: dict, 
                             band_idx: int) -> float:
        """
        Score likelihood that a band contains micro-motions (breathing, fabric).
        Typically highest frequency with low magnitude.
        """
        band_stats = freq_analysis[f'band_{band_idx}']
        
        # Low magnitude
        low_magnitude = 1.0 / (1.0 + band_stats['mean_magnitude'])
        
        # Low temporal variance (continuous)
        temporal_smoothness = 1.0 / (1.0 + band_stats['temporal_variance'])
        
        micro_score = low_magnitude * 0.6 + temporal_smoothness * 0.4
        
        return micro_score
    
    @staticmethod
    def recommend_band_weights(freq_analysis: dict,
                              remove_shake: bool = True,
                              preserve_micro: float = 0.5) -> List[float]:
        """
        Recommend band weights based on motion analysis.
        """
        num_bands = len(freq_analysis)
        weights = []
        
        for i in range(num_bands):
            shake_score = MotionClassifier.identify_camera_shake(freq_analysis, i)
            body_score = MotionClassifier.identify_body_movement(freq_analysis, i)
            micro_score = MotionClassifier.identify_micro_motion(freq_analysis, i)
            
            # Decision logic
            if remove_shake and shake_score > 0.7:
                weight = 0.0  # Remove camera shake
            elif body_score > 0.6:
                weight = 1.0  # Keep body movement
            elif micro_score > 0.5:
                weight = preserve_micro  # Partially keep micro-motion
            else:
                weight = 0.8  # Default: mostly keep
            
            weights.append(weight)
        
        return weights
```

### Step 4: Integration with Noise Warping Pipeline

Create `frequency_motion_transfer.py`:

```python
#!/usr/bin/env python3
"""
Frequency-decomposed motion transfer pipeline.
Extracts motion from video, decomposes into frequency bands,
selectively edits, and applies to new content.
"""

import argparse
import numpy as np
import torch
import os
from pathlib import Path

# Import from existing codebase
from make_warped_noise import extract_full_flow_sequence
from CommonSource.frequency_motion_editor import FrequencyMotionEditor
from CommonSource.motion_classifier import MotionClassifier
from CommonSource.noise_warp import warp_noise_iteratively

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_video', help='Input video with motion to extract')
    parser.add_argument('--target_image', help='Target image for I2V generation')
    parser.add_argument('--target_prompt', help='Target prompt for T2V generation')
    parser.add_argument('--output_dir', default='freq_motion_output')
    parser.add_argument('--num_bands', type=int, default=4, 
                       help='Number of frequency bands')
    parser.add_argument('--remove_shake', action='store_true',
                       help='Automatically remove camera shake')
    parser.add_argument('--micro_motion_scale', type=float, default=0.5,
                       help='Scale factor for micro-motions (0-1)')
    parser.add_argument('--custom_weights', type=str, 
                       help='Custom band weights as comma-separated values')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze motion, don\'t generate')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Step 1: Extracting optical flow from source video...")
    flow_sequence = extract_full_flow_sequence(args.source_video)
    print(f"  Extracted flow shape: {flow_sequence.shape}")
    
    print("\nStep 2: Decomposing into frequency bands...")
    editor = FrequencyMotionEditor(flow_sequence, fps=30)
    freq_bands = editor.decompose_frequencies(num_bands=args.num_bands)
    
    # Save individual bands for inspection
    for i, band in enumerate(freq_bands):
        np.save(output_dir / f'band_{i}_flow.npy', band)
        print(f"  Band {i}: Saved flow with shape {band.shape}")
    
    print("\nStep 3: Analyzing motion characteristics...")
    analysis = editor.analyze_motion_spectrum()
    
    # Print analysis
    for band_name, stats in analysis.items():
        print(f"\n  {band_name}:")
        print(f"    Mean magnitude: {stats['mean_magnitude']:.2f}")
        print(f"    Temporal variance: {stats['temporal_variance']:.3f}")
        print(f"    Spatial variance: {stats['spatial_variance']:.3f}")
        print(f"    Dominant direction: {stats['dominant_direction']:.1f}°")
    
    if args.analyze_only:
        print("\nAnalysis complete. Exiting without generation.")
        return
    
    print("\nStep 4: Selecting frequency bands...")
    
    if args.custom_weights:
        # Use custom weights
        weights = [float(w) for w in args.custom_weights.split(',')]
        print(f"  Using custom weights: {weights}")
    else:
        # Auto-recommend weights
        weights = MotionClassifier.recommend_band_weights(
            analysis,
            remove_shake=args.remove_shake,
            preserve_micro=args.micro_motion_scale
        )
        print(f"  Recommended weights: {weights}")
        
        # Show classification results
        for i in range(args.num_bands):
            shake = MotionClassifier.identify_camera_shake(analysis, i)
            body = MotionClassifier.identify_body_movement(analysis, i)
            micro = MotionClassifier.identify_micro_motion(analysis, i)
            print(f"    Band {i}: shake={shake:.2f}, body={body:.2f}, micro={micro:.2f}")
    
    print("\nStep 5: Reconstructing selective flow...")
    clean_flow = editor.reconstruct_selective(weights)
    np.save(output_dir / 'clean_flow.npy', clean_flow)
    print(f"  Reconstructed flow shape: {clean_flow.shape}")
    
    # Compare magnitudes
    original_magnitude = np.mean(np.sqrt(flow_sequence[:, 0]**2 + flow_sequence[:, 1]**2))
    clean_magnitude = np.mean(np.sqrt(clean_flow[:, 0]**2 + clean_flow[:, 1]**2))
    print(f"  Original mean magnitude: {original_magnitude:.2f}")
    print(f"  Cleaned mean magnitude: {clean_magnitude:.2f}")
    print(f"  Reduction: {(1 - clean_magnitude/original_magnitude)*100:.1f}%")
    
    print("\nStep 6: Warping noise with cleaned flow...")
    # Use existing noise warping function
    from CommonSource.noise_warp import warp_noise_iteratively
    
    # Initialize noise
    H, W = clean_flow.shape[2], clean_flow.shape[3]
    initial_noise = torch.randn(1, 4, H//8, W//8)  # Latent space dimensions
    
    # Warp noise following cleaned flow
    warped_noise = warp_noise_iteratively(
        initial_noise,
        clean_flow,
        mode='forward'
    )
    
    torch.save(warped_noise, output_dir / 'warped_noise.pt')
    print(f"  Warped noise shape: {warped_noise.shape}")
    
    print("\nStep 7: Generating video with cleaned motion...")
    # Call existing inference script
    import subprocess
    
    cmd = [
        'python', 'cut_and_drag_inference.py',
        str(output_dir),
        '--device', 'cuda',
        '--num_inference_steps', '30'
    ]
    
    if args.target_image:
        cmd.extend(['--image_path', args.target_image])
    if args.target_prompt:
        cmd.extend(['--prompt', args.target_prompt])
    
    subprocess.run(cmd)
    
    print("\nFrequency-decomposed motion transfer complete!")
    print(f"Output saved to: {output_dir}")

if __name__ == '__main__':
    main()
```

### Step 5: Advanced Features

Add spatial masking for local frequency editing:

```python
# In frequency_motion_editor.py, add method:
def apply_spatial_frequency_edit(self, 
                                mask: np.ndarray,
                                band_weights_fg: List[float],
                                band_weights_bg: List[float]) -> np.ndarray:
    """
    Apply different frequency weights to masked regions.
    Useful for removing shake from background while preserving subject motion.
    
    Args:
        mask: Binary or soft mask (H, W) where 1=foreground, 0=background
        band_weights_fg: Weights for foreground region
        band_weights_bg: Weights for background region
    """
    # Ensure mask is properly shaped
    mask = mask[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, H, W)
    
    # Reconstruct separately for foreground and background
    flow_fg = self.reconstruct_selective(band_weights_fg)
    flow_bg = self.reconstruct_selective(band_weights_bg)
    
    # Combine using mask
    combined_flow = flow_fg * mask + flow_bg * (1 - mask)
    
    return combined_flow
```

### Step 6: Testing and Validation

Create `test_frequency_decomposition.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

def visualize_frequency_bands(video_path: str, output_dir: str = 'viz'):
    """Visualize motion in different frequency bands."""
    
    # Extract flow
    flow = extract_full_flow_sequence(video_path)
    
    # Decompose
    editor = FrequencyMotionEditor(flow)
    bands = editor.decompose_frequencies(num_bands=4)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, band in enumerate(bands):
        # Plot temporal energy
        temporal_energy = np.sum(band**2, axis=(1, 2, 3))
        axes[0, i].plot(temporal_energy)
        axes[0, i].set_title(f'Band {i} Temporal Energy')
        axes[0, i].set_xlabel('Frame')
        
        # Plot spatial distribution (mean over time)
        spatial_mean = np.mean(np.sqrt(band[:, 0]**2 + band[:, 1]**2), axis=0)
        im = axes[1, i].imshow(spatial_mean, cmap='hot')
        axes[1, i].set_title(f'Band {i} Spatial Distribution')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frequency_bands_analysis.png')
    print(f"Saved visualization to {output_dir}/frequency_bands_analysis.png")

if __name__ == '__main__':
    import sys
    visualize_frequency_bands(sys.argv[1])
```

## Usage Examples

### Basic usage:
```bash
# Analyze motion frequencies
python frequency_motion_transfer.py shaky_dance.mp4 --analyze_only

# Remove camera shake, keep choreography
python frequency_motion_transfer.py shaky_dance.mp4 \
    --target_image cartoon.png \
    --remove_shake \
    --output_dir stabilized_motion

# Custom frequency selection
python frequency_motion_transfer.py dance.mp4 \
    --custom_weights 1.0,1.0,0.0,0.5 \
    --target_prompt "A robot dancing"
```

### Advanced usage with spatial masking:
```python
# In your script
from CommonSource.frequency_motion_editor import FrequencyMotionEditor
import cv2

# Load mask (e.g., from segmentation)
mask = cv2.imread('dancer_mask.png', 0) / 255.0

# Different frequency handling for subject vs background
editor = FrequencyMotionEditor(flow_sequence)
editor.decompose_frequencies()

# Keep all motion for dancer, remove shake from background
clean_flow = editor.apply_spatial_frequency_edit(
    mask=mask,
    band_weights_fg=[1.0, 1.0, 0.8, 0.5],  # Keep most motion
    band_weights_bg=[1.0, 1.0, 0.0, 0.0]   # Remove high frequencies
)
```

## Implementation Notes

1. **FFT Window Functions**: Use Hann or Hamming windows to reduce edge artifacts in FFT
2. **Frequency Resolution**: With 49 frames at 30fps, you get ~0.6Hz frequency resolution
3. **Memory Management**: FFT of large videos requires significant memory; process in chunks if needed
4. **GPU Acceleration**: Use CuPy or PyTorch's FFT for GPU acceleration on large videos
5. **Band Overlap**: Use soft transitions between bands to avoid sharp artifacts

## Testing Checklist

- [ ] Test with various shake intensities
- [ ] Verify frequency band separation is working
- [ ] Check reconstruction quality (should match original when all weights=1.0)
- [ ] Test with different video types (handheld, drone, tripod+subject)
- [ ] Validate motion classifier accuracy
- [ ] Benchmark performance on different video sizes
- [ ] Test edge cases (static video, extreme motion)

## Integration Points

1. **Make_warped_noise.py**: Extend to save full flow sequences
2. **CommonSource/noise_warp.py**: Feed cleaned flows instead of raw flows
3. **Cut_and_drag_inference.py**: No changes needed, works with modified noise
4. **Training**: Can train models on frequency-filtered motion for specialized effects

This implementation gives you fine-grained control over motion transfer, enabling professional-quality stabilization and selective motion editing while maintaining the efficiency of the original Go-with-the-Flow pipeline.