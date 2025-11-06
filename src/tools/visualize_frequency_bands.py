"""
Visualize individual frequency bands from FFT decomposition

Usage:
    python src/tools/visualize_frequency_bands.py <result_dir> [options]

Examples:
    # Create grid showing all bands
    python src/tools/visualize_frequency_bands.py results/fft_analysis/beat_it/

    # Create separate videos for each band
    python src/tools/visualize_frequency_bands.py results/fft_analysis/beat_it/ --separate

    # Custom output path
    python src/tools/visualize_frequency_bands.py results/fft_analysis/beat_it/ \
        --output results/visualizations/beat_it_bands.mp4
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json


def flow_to_hsv(flow_frame, white_background=True):
    """
    Convert optical flow to HSV color-coded image.
    """
    # Handle different input formats
    if flow_frame.shape[0] == 2:
        # (2, H, W) -> (H, W, 2)
        flow_uv = np.transpose(flow_frame, (1, 2, 0))
    else:
        flow_uv = flow_frame

    flow_uv = flow_uv.astype(np.float32)
    h, w = flow_uv.shape[:2]

    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])

    # Normalize magnitude
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if white_background:
        # Start with white background
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Create HSV visualization for areas with motion
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Hue: direction
        hsv[..., 1] = 255  # Saturation: full
        hsv[..., 2] = 255  # Value: full brightness

        # Convert to RGB
        flow_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Blend based on magnitude
        alpha = mag_norm[:, :, np.newaxis] / 255.0
        rgb = (alpha * flow_colored + (1 - alpha) * rgb).astype(np.uint8)
    else:
        # Original black background
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = mag_norm
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def draw_flow_vectors(image, flow_frame, arrow_spacing=20, arrow_scale=1.0,
                       color=(0, 0, 0), thickness=1, tip_length=0.3):
    """
    Draw arrow vectors on top of an image to show flow direction.
    """
    # Handle different input formats
    if flow_frame.shape[0] == 2:
        # (2, H, W) -> (H, W, 2)
        flow_uv = np.transpose(flow_frame, (1, 2, 0))
    else:
        flow_uv = flow_frame

    h, w = flow_uv.shape[:2]
    output = image.copy()

    # Draw arrows on a grid
    for y in range(arrow_spacing // 2, h, arrow_spacing):
        for x in range(arrow_spacing // 2, w, arrow_spacing):
            # Get flow at this point
            fx = flow_uv[y, x, 0] * arrow_scale
            fy = flow_uv[y, x, 1] * arrow_scale

            # Only draw if flow is significant (avoid cluttering with tiny arrows)
            magnitude = np.sqrt(fx**2 + fy**2)
            if magnitude > 0.5:  # Threshold to avoid noise
                # Start and end points
                start_point = (int(x), int(y))
                end_point = (int(x + fx), int(y + fy))

                # Draw arrow
                cv2.arrowedLine(output, start_point, end_point, color,
                               thickness=thickness, tipLength=tip_length)

    return output


def add_text_overlay(image, text, position='top', bg_color=(255, 255, 255),
                     text_color=(0, 0, 0), font_scale=0.35, thickness=1):
    """Add text overlay to image"""
    h, w = image.shape[:2]
    img_copy = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate position
    if position == 'top':
        x = (w - text_w) // 2
        y = 30
    elif position == 'bottom':
        x = (w - text_w) // 2
        y = h - 10
    else:
        x, y = position

    # Draw background rectangle
    padding = 5
    cv2.rectangle(img_copy,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding),
                  bg_color, -1)

    # Draw text
    cv2.putText(img_copy, text, (x, y), font, font_scale, text_color, thickness)

    return img_copy


def create_band_grid_video(band_flows, band_info, output_path, fps=12,
                           white_background=True, show_original=True,
                           arrow_spacing=20, arrow_scale=1.0):
    """
    Create a grid video showing all frequency bands together.

    Args:
        band_flows: Dict with keys 'band_0', 'band_1', etc. containing flow arrays
        band_info: Dict with band metadata (frequencies, etc.)
        output_path: Path to save video
        fps: Frames per second
        white_background: Use white background
        show_original: Include original flow in grid
        arrow_spacing: Spacing between arrow vectors
        arrow_scale: Scale factor for arrow length
    """
    # Determine grid layout
    n_bands = len([k for k in band_flows.keys() if k.startswith('band_')])
    n_videos = n_bands + (1 if show_original else 0)

    # Create 2x2 or 2x3 grid
    if n_videos <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = 2, 3

    # Get video dimensions
    sample_flow = band_flows[list(band_flows.keys())[0]]
    T, _, H, W = sample_flow.shape

    print(f"Creating {n_rows}x{n_cols} grid video with {n_videos} panels...")
    print(f"Video: {T} frames, {H}x{W} per panel")

    # Calculate grid dimensions with padding
    padding = 10
    panel_h = H
    panel_w = W
    grid_h = n_rows * panel_h + (n_rows + 1) * padding
    grid_w = n_cols * panel_w + (n_cols + 1) * padding

    # Setup video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))

    # Create frames
    for t in range(T):
        # Create grid background
        if white_background:
            grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # Light gray
        else:
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        panel_idx = 0

        # Add original flow if requested
        if show_original and 'original' in band_flows:
            row = panel_idx // n_cols
            col = panel_idx % n_cols
            y_start = padding + row * (panel_h + padding)
            x_start = padding + col * (panel_w + padding)

            flow_frame = band_flows['original'][t]
            vis = flow_to_hsv(flow_frame, white_background=white_background)

            # Add arrow vectors
            arrow_color = (0, 0, 0) if white_background else (255, 255, 255)
            vis = draw_flow_vectors(vis, flow_frame, arrow_spacing=arrow_spacing,
                                   arrow_scale=arrow_scale, color=arrow_color)

            # Calculate magnitude for label
            mag = np.sqrt(flow_frame[0]**2 + flow_frame[1]**2).mean()
            label = f"Original (mag: {mag:.2f})"
            vis = add_text_overlay(vis, label, position='top')

            grid[y_start:y_start+panel_h, x_start:x_start+panel_w] = vis
            panel_idx += 1

        # Add frequency bands
        for i in range(n_bands):
            band_key = f'band_{i}'
            if band_key not in band_flows:
                continue

            row = panel_idx // n_cols
            col = panel_idx % n_cols
            y_start = padding + row * (panel_h + padding)
            x_start = padding + col * (panel_w + padding)

            flow_frame = band_flows[band_key][t]
            vis = flow_to_hsv(flow_frame, white_background=white_background)

            # Add arrow vectors
            arrow_color = (0, 0, 0) if white_background else (255, 255, 255)
            vis = draw_flow_vectors(vis, flow_frame, arrow_spacing=arrow_spacing,
                                   arrow_scale=arrow_scale, color=arrow_color)

            # Get band info
            freq_range = band_info.get('frequency_ranges_hz', [[0, 0]])[i]
            mag = np.sqrt(flow_frame[0]**2 + flow_frame[1]**2).mean()

            label = f"Band {i}: {freq_range[0]:.2f}-{freq_range[1]:.2f}Hz (mag: {mag:.2f})"
            vis = add_text_overlay(vis, label, position='top')

            grid[y_start:y_start+panel_h, x_start:x_start+panel_w] = vis
            panel_idx += 1

        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"✓ Saved grid video to: {output_path}")


def create_separate_band_videos(band_flows, band_info, output_dir, fps=12,
                                white_background=True, arrow_spacing=20,
                                arrow_scale=1.0):
    """
    Create separate videos for each frequency band.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    n_bands = len([k for k in band_flows.keys() if k.startswith('band_')])

    print(f"Creating {n_bands} separate band videos...")

    for i in range(n_bands):
        band_key = f'band_{i}'
        if band_key not in band_flows:
            continue

        flow = band_flows[band_key]
        T, _, H, W = flow.shape

        # Get band info
        freq_range = band_info.get('frequency_ranges_hz', [[0, 0]])[i]

        # Setup video writer
        output_path = output_dir / f'band_{i}_{freq_range[0]:.2f}-{freq_range[1]:.2f}Hz.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        # Create frames
        for t in range(T):
            flow_frame = flow[t]
            vis = flow_to_hsv(flow_frame, white_background=white_background)

            # Add arrow vectors
            arrow_color = (0, 0, 0) if white_background else (255, 255, 255)
            vis = draw_flow_vectors(vis, flow_frame, arrow_spacing=arrow_spacing,
                                   arrow_scale=arrow_scale, color=arrow_color)

            # Add label
            mag = np.sqrt(flow_frame[0]**2 + flow_frame[1]**2).mean()
            label = f"Band {i}: {freq_range[0]:.2f}-{freq_range[1]:.2f}Hz (mag: {mag:.2f})"
            vis = add_text_overlay(vis, label, position='top')

            # Convert and write
            bgr_frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()
        print(f"✓ Saved band {i} to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize individual frequency bands from FFT decomposition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('result_dir', type=str,
                       help='Path to FFT analysis results directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid video (default: result_dir/bands_grid.mp4)')
    parser.add_argument('--separate', action='store_true',
                       help='Create separate videos for each band instead of grid')
    parser.add_argument('--fps', type=int, default=12,
                       help='Frames per second (default: 12)')
    parser.add_argument('--black_background', action='store_true',
                       help='Use black background instead of white')
    parser.add_argument('--no_original', action='store_true',
                       help='Do not include original flow in grid')
    parser.add_argument('--arrow_spacing', type=int, default=20,
                       help='Spacing between arrows in pixels (default: 20)')
    parser.add_argument('--arrow_scale', type=float, default=1.0,
                       help='Scale factor for arrow length (default: 1.0)')

    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    if not result_dir.exists():
        print(f"Error: Directory not found: {result_dir}")
        return

    # Load band info
    info_path = result_dir / 'band_info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            band_info = json.load(f)
        print(f"Loaded band info: {band_info.get('n_bands', 'unknown')} bands")
    else:
        print("Warning: band_info.json not found, using defaults")
        band_info = {}

    # Load all band flows
    print("\nLoading frequency band flows...")
    band_flows = {}

    # Load original flow
    original_path = result_dir / 'original_flow.npy'
    if original_path.exists():
        band_flows['original'] = np.load(original_path)
        print(f"  ✓ Original flow: {band_flows['original'].shape}")

    # Load individual bands
    band_idx = 0
    while True:
        band_path = result_dir / f'band_{band_idx}_flow.npy'
        if not band_path.exists():
            break
        band_flows[f'band_{band_idx}'] = np.load(band_path)
        print(f"  ✓ Band {band_idx}: {band_flows[f'band_{band_idx}'].shape}")
        band_idx += 1

    if band_idx == 0:
        print("Error: No band flow files found!")
        print("Expected files like: band_0_flow.npy, band_1_flow.npy, etc.")
        return

    # Set output path
    if args.output is None:
        if args.separate:
            args.output = str(result_dir / 'band_videos')
        else:
            args.output = str(result_dir / 'bands_grid.mp4')

    # Create visualizations
    if args.separate:
        create_separate_band_videos(
            band_flows,
            band_info,
            args.output,
            fps=args.fps,
            white_background=not args.black_background,
            arrow_spacing=args.arrow_spacing,
            arrow_scale=args.arrow_scale
        )
    else:
        create_band_grid_video(
            band_flows,
            band_info,
            args.output,
            fps=args.fps,
            white_background=not args.black_background,
            show_original=not args.no_original,
            arrow_spacing=args.arrow_spacing,
            arrow_scale=args.arrow_scale
        )

    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
