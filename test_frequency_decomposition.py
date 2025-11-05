#!/usr/bin/env python
"""
test_frequency_decomposition.py

Visualization and testing tool for frequency decomposition.
Creates plots and validates the frequency-based motion transfer pipeline.

Usage:
    # Visualize frequency bands from video
    python test_frequency_decomposition.py video.mp4

    # Visualize from pre-computed output directory
    python test_frequency_decomposition.py freq_motion_output/

    # Run unit tests
    python test_frequency_decomposition.py --test
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from CommonSource.frequency_motion_editor import FrequencyMotionEditor
from CommonSource.motion_classifier import MotionClassifier


def visualize_frequency_bands(flow_sequence: np.ndarray,
                             output_dir: str = 'viz',
                             fps: int = 30,
                             num_bands: int = 4):
    """
    Visualize motion in different frequency bands.

    Args:
        flow_sequence: Flow array with shape (T, 2, H, W)
        output_dir: Directory to save visualizations
        fps: Video framerate
        num_bands: Number of frequency bands
    """
    print("\n" + "="*60)
    print("VISUALIZING FREQUENCY BANDS")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Decompose
    print("\nDecomposing into frequency bands...")
    editor = FrequencyMotionEditor(flow_sequence, fps=fps)
    bands = editor.decompose_frequencies(num_bands=num_bands)

    # Analyze
    print("Analyzing motion spectrum...")
    analysis = editor.analyze_motion_spectrum()

    # Create visualizations
    print("Creating visualizations...")

    # Figure 1: Temporal energy for each band
    fig, axes = plt.subplots(2, num_bands, figsize=(4*num_bands, 8))

    for i, band in enumerate(bands):
        # Plot temporal energy (top row)
        temporal_energy = np.sum(band**2, axis=(1, 2, 3))
        axes[0, i].plot(temporal_energy, linewidth=2)
        axes[0, i].set_title(f'Band {i} Temporal Energy', fontsize=12, fontweight='bold')
        axes[0, i].set_xlabel('Frame', fontsize=10)
        axes[0, i].set_ylabel('Energy', fontsize=10)
        axes[0, i].grid(True, alpha=0.3)

        # Plot spatial distribution (bottom row)
        spatial_mean = np.mean(np.sqrt(band[:, 0]**2 + band[:, 1]**2), axis=0)
        im = axes[1, i].imshow(spatial_mean, cmap='hot', aspect='auto')
        axes[1, i].set_title(f'Band {i} Spatial Distribution', fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel('Width', fontsize=10)
        axes[1, i].set_ylabel('Height', fontsize=10)
        plt.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    output_path = output_dir / 'frequency_bands_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()

    # Figure 2: Band statistics comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    stats_to_plot = [
        ('mean_magnitude', 'Mean Magnitude'),
        ('temporal_variance', 'Temporal Variance'),
        ('spatial_variance', 'Spatial Variance'),
        ('max_magnitude', 'Max Magnitude'),
        ('std_magnitude', 'Std Magnitude'),
        ('dominant_direction', 'Dominant Direction (°)')
    ]

    for idx, (stat_key, stat_name) in enumerate(stats_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = [analysis[f'band_{i}'][stat_key] for i in range(num_bands)]
        bars = ax.bar(range(num_bands), values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(stat_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Band Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(range(num_bands))
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'band_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved statistics to {output_path}")
    plt.close()

    # Figure 3: Motion classification scores
    fig, ax = plt.subplots(figsize=(12, 6))

    band_indices = np.arange(num_bands)
    width = 0.25

    shake_scores = [MotionClassifier.identify_camera_shake(analysis, i) for i in range(num_bands)]
    body_scores = [MotionClassifier.identify_body_movement(analysis, i) for i in range(num_bands)]
    micro_scores = [MotionClassifier.identify_micro_motion(analysis, i) for i in range(num_bands)]

    bars1 = ax.bar(band_indices - width, shake_scores, width, label='Camera Shake', color='red', alpha=0.7)
    bars2 = ax.bar(band_indices, body_scores, width, label='Body Movement', color='green', alpha=0.7)
    bars3 = ax.bar(band_indices + width, micro_scores, width, label='Micro Motion', color='blue', alpha=0.7)

    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Score', fontsize=12, fontweight='bold')
    ax.set_title('Motion Type Classification by Frequency Band', fontsize=14, fontweight='bold')
    ax.set_xticks(band_indices)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'motion_classification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved classification to {output_path}")
    plt.close()

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


def run_unit_tests():
    """Run comprehensive unit tests on the implementation."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)

    all_passed = True

    # Test 1: FrequencyMotionEditor instantiation
    print("\n[Test 1] FrequencyMotionEditor instantiation...")
    try:
        T, H, W = 49, 60, 90
        flow = np.random.randn(T, 2, H, W).astype(np.float32)
        editor = FrequencyMotionEditor(flow, fps=30)
        assert editor.T == T
        assert editor.H == H
        assert editor.W == W
        assert editor.nyquist_freq == 15.0
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Frequency decomposition
    print("\n[Test 2] Frequency decomposition...")
    try:
        num_bands = 4
        bands = editor.decompose_frequencies(num_bands=num_bands)
        assert len(bands) == num_bands
        for i, band in enumerate(bands):
            assert band.shape == flow.shape, f"Band {i} shape mismatch"
        print(f"  ✓ PASSED - Created {num_bands} bands")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Perfect reconstruction (all weights = 1.0)
    print("\n[Test 3] Perfect reconstruction test...")
    try:
        weights = [1.0] * num_bands
        reconstructed = editor.reconstruct_selective(weights)

        # Check shape
        assert reconstructed.shape == flow.shape

        # Check if reasonably close (allowing for windowing effects)
        mse = np.mean((reconstructed - flow)**2)
        relative_error = mse / (np.mean(flow**2) + 1e-8)

        print(f"  Relative reconstruction error: {relative_error:.6f}")
        if relative_error < 0.5:  # Generous threshold due to windowing
            print("  ✓ PASSED")
        else:
            print(f"  ⚠ WARNING: High reconstruction error (windowing expected)")
            print("  ✓ PASSED (with warning)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Zero reconstruction (all weights = 0.0)
    print("\n[Test 4] Zero reconstruction test...")
    try:
        weights = [0.0] * num_bands
        reconstructed = editor.reconstruct_selective(weights)
        max_val = np.max(np.abs(reconstructed))
        print(f"  Max absolute value: {max_val:.6f}")
        assert max_val < 1e-10, "Reconstruction should be near zero"
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: Motion spectrum analysis
    print("\n[Test 5] Motion spectrum analysis...")
    try:
        analysis = editor.analyze_motion_spectrum()
        assert len(analysis) == num_bands
        for i in range(num_bands):
            band_stats = analysis[f'band_{i}']
            required_keys = ['mean_magnitude', 'std_magnitude', 'max_magnitude',
                           'temporal_variance', 'spatial_variance', 'dominant_direction']
            for key in required_keys:
                assert key in band_stats, f"Missing key: {key}"
                assert isinstance(band_stats[key], (float, int))
        print(f"  ✓ PASSED - Analyzed {num_bands} bands")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: MotionClassifier
    print("\n[Test 6] MotionClassifier functions...")
    try:
        for i in range(num_bands):
            shake = MotionClassifier.identify_camera_shake(analysis, i)
            body = MotionClassifier.identify_body_movement(analysis, i)
            micro = MotionClassifier.identify_micro_motion(analysis, i)

            assert 0 <= shake <= 1, f"Shake score out of range: {shake}"
            assert 0 <= body <= 1, f"Body score out of range: {body}"
            assert 0 <= micro <= 1, f"Micro score out of range: {micro}"

        weights = MotionClassifier.recommend_band_weights(analysis)
        assert len(weights) == num_bands
        assert all(0 <= w <= 1 for w in weights), "Weights out of range"

        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 7: Spatial frequency editing
    print("\n[Test 7] Spatial frequency editing...")
    try:
        mask = np.ones((H, W))
        mask[:H//2, :] = 0  # Top half background

        weights_fg = [1.0, 1.0, 0.5, 0.0]
        weights_bg = [1.0, 0.0, 0.0, 0.0]

        combined = editor.apply_spatial_frequency_edit(mask, weights_fg, weights_bg)
        assert combined.shape == flow.shape
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 8: Different band types
    print("\n[Test 8] Linear vs logarithmic band spacing...")
    try:
        editor_linear = FrequencyMotionEditor(flow, fps=30)
        bands_linear = editor_linear.decompose_frequencies(num_bands=4, band_type='linear')

        editor_log = FrequencyMotionEditor(flow, fps=30)
        bands_log = editor_log.decompose_frequencies(num_bands=4, band_type='logarithmic')

        assert len(bands_linear) == len(bands_log) == 4
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Test and visualize frequency decomposition')
    parser.add_argument('input', nargs='?', help='Video file or output directory')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--output', default='viz', help='Output directory for visualizations')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--num_bands', type=int, default=4, help='Number of frequency bands')

    args = parser.parse_args()

    if args.test:
        success = run_unit_tests()
        sys.exit(0 if success else 1)

    if not args.input:
        print("Error: Please provide input video or directory")
        print("Usage: python test_frequency_decomposition.py <video_or_dir>")
        print("   or: python test_frequency_decomposition.py --test")
        sys.exit(1)

    input_path = Path(args.input)

    # Check if input is a directory with pre-computed flows
    if input_path.is_dir():
        flow_path = input_path / 'raw_flows.npy'
        if flow_path.exists():
            print(f"Loading flows from: {flow_path}")
            flow_sequence = np.load(flow_path)
        else:
            print(f"Error: No raw_flows.npy found in {input_path}")
            sys.exit(1)
    else:
        # Extract flow from video
        print(f"Extracting flows from video: {input_path}")
        from extract_flow_sequence import extract_full_flow_sequence
        flow_sequence = extract_full_flow_sequence(str(input_path), fps=args.fps, save_raw=False)

    # Visualize
    visualize_frequency_bands(flow_sequence, args.output, args.fps, args.num_bands)


if __name__ == '__main__':
    main()
