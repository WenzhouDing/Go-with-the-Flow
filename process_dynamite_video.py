"""
Process dynamite_clip.mp4: Extract flow, apply FFT filtering, and visualize results
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Use rp imports like in make_warped_noise.py
import rp
rp.r._pip_import_autoyes = True
rp.git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw

# Add local CommonSource to path
sys.path.insert(0, str(Path(__file__).parent))
from CommonSource.frequency_motion_editor import FrequencyMotionEditor
from CommonSource.motion_classifier import MotionClassifier


def visualize_flow(flow, title="Optical Flow", save_path=None):
    """
    Visualize optical flow as RGB (hue=direction, saturation=magnitude)

    Args:
        flow: (T, 2, H, W) flow array
        title: Title for the plot
        save_path: Optional path to save visualization
    """
    T = flow.shape[0]

    # Select a few representative frames
    frame_indices = [0, T//4, T//2, 3*T//4, T-1]

    fig, axes = plt.subplots(1, len(frame_indices), figsize=(20, 4))
    fig.suptitle(title, fontsize=16)

    for idx, frame_idx in enumerate(frame_indices):
        flow_frame = flow[frame_idx]  # (2, H, W)

        # Convert to (H, W, 2) for visualization
        flow_uv = np.transpose(flow_frame, (1, 2, 0)).astype(np.float32)

        # Create HSV visualization
        h, w = flow_uv.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Magnitude and angle
        mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])

        # Hue = direction
        hsv[..., 0] = ang * 180 / np.pi / 2

        # Saturation = 255 (full)
        hsv[..., 1] = 255

        # Value = normalized magnitude
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        axes[idx].imshow(rgb)
        axes[idx].set_title(f'Frame {frame_idx}\nMag: {mag.mean():.2f}')
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")

    plt.close()


def create_flow_video(flow, output_path, fps=30):
    """
    Create a video visualizing the optical flow

    Args:
        flow: (T, 2, H, W) flow array
        output_path: Path to save video
        fps: Frames per second
    """
    T, _, H, W = flow.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in range(T):
        flow_frame = flow[t]  # (2, H, W)
        flow_uv = np.transpose(flow_frame, (1, 2, 0)).astype(np.float32)

        # Create HSV visualization
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(rgb)

    out.release()
    print(f"✓ Saved flow video: {output_path}")


def plot_motion_spectrum(flow, analysis, title="Motion Spectrum", save_path=None, fps=12):
    """
    Plot the motion spectrum showing magnitude per frequency band
    """
    num_bands = len([k for k in analysis.keys() if k.startswith('band_')])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # Compute frequency ranges (same logic as in FrequencyMotionEditor)
    nyquist = fps / 2.0
    min_freq = 0.1
    freq_edges = np.logspace(np.log10(min_freq), np.log10(nyquist), num_bands + 1)

    # Extract per-band stats
    bands_data = []
    for i in range(num_bands):
        band_stats = analysis[f'band_{i}']
        bands_data.append({
            'band': i,
            'magnitude': band_stats['mean_magnitude'],
            'temporal_var': band_stats['temporal_variance'],
            'spatial_var': band_stats['spatial_variance'],
            'freq_range': (freq_edges[i], freq_edges[i+1])
        })

    # Plot 1: Magnitude per band
    ax = axes[0, 0]
    magnitudes = [b['magnitude'] for b in bands_data]
    ax.bar(range(num_bands), magnitudes, color=['blue', 'green', 'orange', 'red'])
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title('Motion Magnitude per Band')
    ax.set_xticks(range(num_bands))
    for i, b in enumerate(bands_data):
        ax.text(i, magnitudes[i], f"{magnitudes[i]:.2f}", ha='center', va='bottom')

    # Plot 2: Temporal variance
    ax = axes[0, 1]
    temp_vars = [b['temporal_var'] for b in bands_data]
    ax.bar(range(num_bands), temp_vars, color=['blue', 'green', 'orange', 'red'])
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Temporal Variance')
    ax.set_title('Temporal Variance per Band')
    ax.set_xticks(range(num_bands))

    # Plot 3: Frequency ranges
    ax = axes[1, 0]
    freq_labels = [f"Band {i}\n{b['freq_range'][0]:.1f}-{b['freq_range'][1]:.1f}Hz"
                   for i, b in enumerate(bands_data)]
    ax.bar(range(num_bands), magnitudes, color=['blue', 'green', 'orange', 'red'])
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title('Frequency Ranges')
    ax.set_xticks(range(num_bands))
    ax.set_xticklabels(freq_labels, rotation=0, fontsize=9)

    # Plot 4: Per-frame magnitude over time
    ax = axes[1, 1]
    T = flow.shape[0]
    mag_per_frame = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2).mean(axis=(1, 2))
    ax.plot(mag_per_frame, linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title('Motion Magnitude Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved spectrum plot: {save_path}")

    plt.close()


def compare_flows(original, filtered, title="Flow Comparison", save_path=None):
    """
    Create side-by-side comparison of original vs filtered flows
    """
    T = original.shape[0]
    frame_indices = [0, T//4, T//2, 3*T//4, T-1]

    fig, axes = plt.subplots(2, len(frame_indices), figsize=(20, 8))
    fig.suptitle(title, fontsize=16)

    for idx, frame_idx in enumerate(frame_indices):
        # Original flow
        flow_orig = original[frame_idx]
        flow_uv = np.transpose(flow_orig, (1, 2, 0)).astype(np.float32)

        hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_orig = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        axes[0, idx].imshow(rgb_orig)
        axes[0, idx].set_title(f'Original\nFrame {frame_idx}\nMag: {mag.mean():.2f}')
        axes[0, idx].axis('off')

        # Filtered flow
        flow_filt = filtered[frame_idx]
        flow_uv = np.transpose(flow_filt, (1, 2, 0)).astype(np.float32)

        hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_filt = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        axes[1, idx].imshow(rgb_filt)
        axes[1, idx].set_title(f'Filtered\nFrame {frame_idx}\nMag: {mag.mean():.2f}')
        axes[1, idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison: {save_path}")

    plt.close()


def main():
    print("="*70)
    print("Processing dynamite_clip.mp4 with FFT Motion Filtering")
    print("="*70)

    # Create output directory
    output_dir = Path('dynamite_results')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")

    # Step 1: Extract optical flow
    print("\n[Step 1] Extracting optical flow...")
    video_path = 'test_video/dynamite_clip.mp4'

    video = rp.load_video(video_path)
    print(f"✓ Loaded video: {len(video)} frames")

    # Resize and crop
    video = rp.resize_list(video, length=49)
    video = rp.resize_images_to_hold(video, height=480, width=720)
    video = rp.crop_images(video, height=480, width=720, origin='center')
    video = rp.as_numpy_array(video)  # Convert to numpy array for noise_warp
    print(f"✓ Preprocessed: 49 frames, 480x720")

    # Extract flow using noise_warp
    output = nw.get_noise_from_video(
        video,
        remove_background=False,
        visualize=False,
        save_files=False,
        noise_channels=16,
        output_folder=None,
        resize_frames=1,
        resize_flow=1,
        downscale_factor=1,
    )

    flow = output.numpy_flows  # (T, 2, H, W)
    print(f"✓ Extracted flow: {flow.shape}")
    print(f"  Mean magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.2f}")

    # Save original flow
    np.save(output_dir / 'original_flow.npy', flow)
    print(f"✓ Saved: {output_dir}/original_flow.npy")

    # Step 2: Visualize original flow
    print("\n[Step 2] Visualizing original flow...")
    visualize_flow(flow, "Original Optical Flow", output_dir / 'original_flow.png')
    create_flow_video(flow, str(output_dir / 'original_flow.mp4'), fps=12)

    # Step 3: FFT decomposition and analysis
    print("\n[Step 3] Applying FFT decomposition...")
    editor = FrequencyMotionEditor(flow, fps=12)
    bands = editor.decompose_frequencies(num_bands=4)
    print(f"✓ Decomposed into {len(bands)} frequency bands")

    # Analyze
    analysis = editor.analyze_motion_spectrum()
    print("\n" + "="*70)
    MotionClassifier.print_analysis_report(analysis)
    print("="*70)

    # Save analysis
    import json
    with open(output_dir / 'motion_analysis.json', 'w') as f:
        # Convert numpy types to native Python types
        analysis_serializable = {}
        for k, v in analysis.items():
            if isinstance(v, dict):
                analysis_serializable[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else
                        [float(x) for x in vv] if isinstance(vv, (list, tuple, np.ndarray)) else vv
                    for kk, vv in v.items()
                }
            else:
                analysis_serializable[k] = v
        json.dump(analysis_serializable, f, indent=2)
    print(f"✓ Saved: {output_dir}/motion_analysis.json")

    # Plot spectrum
    plot_motion_spectrum(flow, analysis, "Motion Spectrum Analysis",
                        output_dir / 'motion_spectrum.png', fps=12)

    # Step 4: Apply different filtering presets
    print("\n[Step 4] Applying FFT filters...")

    presets = {
        'hands_emphasized': {
            'weights': [0.0, 0.3, 1.5, 2.0],
            'description': 'Emphasize fast motion (hands), remove body drift'
        },
        'ultra_smooth': {
            'weights': [1.0, 0.8, 0.0, 0.0],
            'description': 'Ultra smooth (remove all fast motion)'
        },
        'extreme_hands': {
            'weights': [0.0, 0.0, 2.0, 3.0],
            'description': 'Extreme hand emphasis (only fast motion)'
        },
        'remove_shake': {
            'weights': [1.0, 1.0, 0.8, 0.0],
            'description': 'Remove shake only (keep most motion)'
        }
    }

    for preset_name, preset_config in presets.items():
        print(f"\n  Processing: {preset_name}")
        print(f"    Weights: {preset_config['weights']}")
        print(f"    Description: {preset_config['description']}")

        # Apply filtering
        filtered_flow = editor.reconstruct_selective(preset_config['weights'])

        # Stats
        orig_mag = np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean()
        filt_mag = np.sqrt(filtered_flow[:,0]**2 + filtered_flow[:,1]**2).mean()
        change = (filt_mag/orig_mag - 1) * 100

        print(f"    Result: {orig_mag:.2f} → {filt_mag:.2f} ({change:+.1f}%)")

        # Save
        np.save(output_dir / f'{preset_name}_flow.npy', filtered_flow)
        print(f"    ✓ Saved: {output_dir}/{preset_name}_flow.npy")

        # Visualize
        visualize_flow(filtered_flow, f"Filtered: {preset_name}",
                      output_dir / f'{preset_name}_flow.png')
        create_flow_video(filtered_flow, str(output_dir / f'{preset_name}_flow.mp4'), fps=12)

        # Compare
        compare_flows(flow, filtered_flow, f"Comparison: {preset_name}",
                     output_dir / f'{preset_name}_comparison.png')

    # Step 5: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOriginal flow magnitude: {orig_mag:.2f}")
    print("\nFiltered versions:")
    for preset_name, preset_config in presets.items():
        filtered_flow = np.load(output_dir / f'{preset_name}_flow.npy')
        filt_mag = np.sqrt(filtered_flow[:,0]**2 + filtered_flow[:,1]**2).mean()
        change = (filt_mag/orig_mag - 1) * 100
        print(f"  {preset_name:20s}: {filt_mag:.2f} ({change:+.1f}%)")

    print(f"\n✓ All results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - original_flow.npy/png/mp4       (original optical flow)")
    print("  - motion_analysis.json             (frequency analysis)")
    print("  - motion_spectrum.png              (spectrum visualization)")
    print("  - *_flow.npy/png/mp4              (filtered flows)")
    print("  - *_comparison.png                 (before/after comparisons)")

    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)


if __name__ == '__main__':
    main()
