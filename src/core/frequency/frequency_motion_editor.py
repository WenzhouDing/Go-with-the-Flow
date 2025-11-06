"""
frequency_motion_editor.py

Decomposes optical flow into frequency bands for selective motion editing.
Uses 3D FFT (2D spatial + 1D temporal) for comprehensive motion analysis.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Literal


class FrequencyMotionEditor:
    """
    Decomposes optical flow into frequency bands for selective motion editing.
    Uses 3D FFT (2D spatial + 1D temporal) for comprehensive motion analysis.
    """

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

    def decompose_frequencies(self,
                            num_bands: int = 4,
                            band_type: Literal['logarithmic', 'linear'] = 'logarithmic') -> List[np.ndarray]:
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
        window_t = np.hanning(self.T).reshape(-1, 1, 1)
        flow_u_windowed = flow_u * window_t
        flow_v_windowed = flow_v * window_t

        # FFT along temporal dimension only (axis=0)
        # This gives us temporal frequency decomposition while preserving spatial structure
        fft_u = np.fft.rfft(flow_u_windowed, axis=0)
        fft_v = np.fft.rfft(flow_v_windowed, axis=0)

        self.fft_flow = np.stack([fft_u, fft_v], axis=1)

        # Create frequency bands
        temporal_freqs = np.fft.rfftfreq(self.T, d=1.0/self.fps)
        self.temporal_freqs = temporal_freqs

        # Define band boundaries
        if band_type == 'logarithmic':
            # Logarithmic spacing for better low-frequency resolution
            min_freq = 0.1  # Hz
            max_freq = self.nyquist_freq
            band_edges = np.logspace(
                np.log10(min_freq),
                np.log10(max_freq),
                num_bands + 1
            )
        else:
            # Linear spacing
            band_edges = np.linspace(0, self.nyquist_freq, num_bands + 1)

        # Store band edges for later use
        self.band_edges = band_edges
        self.band_freq_ranges = np.array([[band_edges[i], band_edges[i+1]] for i in range(num_bands)])

        # Extract bands
        self.freq_bands = []
        for i in range(num_bands):
            band_mask = self._create_band_mask(
                temporal_freqs,
                band_edges[i],
                band_edges[i+1]
            )

            # Apply mask to FFT along temporal dimension
            # FFT shape is (T_freq, H, W) where T_freq is temporal frequency bins
            # Broadcast mask along spatial dimensions
            band_mask_broadcast = band_mask[:, np.newaxis, np.newaxis]

            band_fft_u = fft_u * band_mask_broadcast
            band_fft_v = fft_v * band_mask_broadcast

            # Inverse FFT along temporal dimension
            band_flow_u = np.fft.irfft(band_fft_u, n=self.T, axis=0)
            band_flow_v = np.fft.irfft(band_fft_v, n=self.T, axis=0)

            band_flow = np.stack([band_flow_u.real, band_flow_v.real], axis=1)
            self.freq_bands.append(band_flow)

        return self.freq_bands

    def _create_band_mask(self,
                         freqs: np.ndarray,
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

        Returns:
            Combined flow with different frequency weights per region
        """
        # Ensure mask is properly shaped
        mask = mask[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, H, W)

        # Reconstruct separately for foreground and background
        flow_fg = self.reconstruct_selective(band_weights_fg)
        flow_bg = self.reconstruct_selective(band_weights_bg)

        # Combine using mask
        combined_flow = flow_fg * mask + flow_bg * (1 - mask)

        return combined_flow

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
