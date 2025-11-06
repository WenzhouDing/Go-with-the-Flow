"""
frequency_motion_transfer.py

Frequency-decomposed motion transfer pipeline.
Extracts motion from video, decomposes into frequency bands,
selectively edits, and applies to new content.

Usage:
    # Analysis only
    conda activate flow_warp
    python frequency_motion_transfer.py source_video.mp4 --analyze_only

    # Full pipeline with automatic shake removal
    python frequency_motion_transfer.py source_video.mp4 \
        --output_dir freq_output \
        --remove_shake \
        --num_bands 4

    # Custom band weights
    python frequency_motion_transfer.py source_video.mp4 \
        --custom_weights 1.0,1.0,0.0,0.5
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import frequency modules from new location
from src.core.frequency.frequency_motion_editor import FrequencyMotionEditor
from src.core.frequency.motion_classifier import MotionClassifier


def main():
    parser = argparse.ArgumentParser(
        description='Frequency-decomposed motion transfer pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('source_video', nargs='?', help='Input video with motion to extract (optional if --flow_path provided)')
    parser.add_argument('--output_dir', '--output', dest='output_dir', default='freq_motion_output',
                       help='Output directory for results')
    parser.add_argument('--num_bands', type=int, default=4,
                       help='Number of frequency bands (default: 4)')
    parser.add_argument('--band_type', choices=['linear', 'logarithmic'],
                       default='logarithmic',
                       help='Frequency band spacing type')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate (default: 30)')
    parser.add_argument('--remove_shake', action='store_true',
                       help='Automatically remove camera shake')
    parser.add_argument('--micro_motion_scale', type=float, default=0.5,
                       help='Scale factor for micro-motions 0-1 (default: 0.5)')
    parser.add_argument('--custom_weights', type=str,
                       help='Custom band weights as comma-separated values (e.g., "1.0,1.0,0.0,0.5")')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze motion, don\'t reconstruct')
    parser.add_argument('--flow_path', type=str,
                       help='Path to pre-extracted flow .npy file (skip extraction)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("FREQUENCY-DECOMPOSED MOTION TRANSFER PIPELINE")
    print("="*70)

    # ========================================================================
    # STEP 1: Extract optical flow from source video
    # ========================================================================
    print("\n[1/6] Extracting optical flow from source video...")
    print("-" * 70)

    if args.flow_path:
        print(f"  Loading pre-extracted flows from: {args.flow_path}")
        flow_sequence = np.load(args.flow_path)
    else:
        print("  Running flow extraction (requires flow_warp conda env)...")
        from src.tools.extract_flow_sequence import extract_full_flow_sequence

        flow_path = output_dir / 'raw_flows.npy'
        flow_sequence = extract_full_flow_sequence(
            args.source_video,
            output_path=str(flow_path),
            fps=args.fps
        )

    print(f"  ✓ Extracted flow shape: {flow_sequence.shape}")
    print(f"    Mean flow magnitude: {np.sqrt(flow_sequence[:,0]**2 + flow_sequence[:,1]**2).mean():.2f}")

    # ========================================================================
    # STEP 2: Decompose into frequency bands
    # ========================================================================
    print("\n[2/6] Decomposing into frequency bands...")
    print("-" * 70)

    editor = FrequencyMotionEditor(flow_sequence, fps=args.fps)
    freq_bands = editor.decompose_frequencies(
        num_bands=args.num_bands,
        band_type=args.band_type
    )

    # Save individual bands for inspection
    for i, band in enumerate(freq_bands):
        band_path = output_dir / f'band_{i}_flow.npy'
        np.save(band_path, band)
        magnitude = np.sqrt(band[:, 0]**2 + band[:, 1]**2).mean()
        print(f"  ✓ Band {i}: Saved flow with shape {band.shape}, magnitude {magnitude:.2f}")

    # Save band metadata
    import json
    band_info = {
        'n_bands': len(freq_bands),
        'fps': args.fps,
        'band_type': args.band_type,
        'frequency_ranges_hz': editor.band_freq_ranges.tolist() if hasattr(editor, 'band_freq_ranges') else []
    }
    band_info_path = output_dir / 'band_info.json'
    with open(band_info_path, 'w') as f:
        json.dump(band_info, f, indent=2)
    print(f"  ✓ Saved band metadata to: {band_info_path}")

    # Save original flow for comparison
    original_flow_path = output_dir / 'original_flow.npy'
    np.save(original_flow_path, flow_sequence)
    print(f"  ✓ Saved original flow for comparison")

    # ========================================================================
    # STEP 3: Analyze motion characteristics
    # ========================================================================
    print("\n[3/6] Analyzing motion characteristics...")
    print("-" * 70)

    analysis = editor.analyze_motion_spectrum()

    # Print detailed analysis
    MotionClassifier.print_analysis_report(analysis)

    # Save analysis to file
    import json
    analysis_path = output_dir / 'motion_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  ✓ Saved analysis to: {analysis_path}")

    if args.analyze_only:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE (--analyze_only mode)")
        print("="*70)
        return

    # ========================================================================
    # STEP 4: Select frequency bands
    # ========================================================================
    print("\n[4/6] Selecting frequency bands...")
    print("-" * 70)

    if args.custom_weights:
        # Use custom weights
        weights = [float(w) for w in args.custom_weights.split(',')]
        if len(weights) != args.num_bands:
            raise ValueError(f"Expected {args.num_bands} weights, got {len(weights)}")
        print(f"  Using custom weights: {weights}")
    else:
        # Auto-recommend weights
        weights = MotionClassifier.recommend_band_weights(
            analysis,
            remove_shake=args.remove_shake,
            preserve_micro=args.micro_motion_scale
        )
        print(f"  Recommended weights: {weights}")

        # Show reasoning
        print("\n  Band Selection Reasoning:")
        for i in range(args.num_bands):
            shake = MotionClassifier.identify_camera_shake(analysis, i)
            body = MotionClassifier.identify_body_movement(analysis, i)
            micro = MotionClassifier.identify_micro_motion(analysis, i)
            classification = MotionClassifier.classify_band(analysis, i)

            print(f"    Band {i} (weight={weights[i]:.2f}):")
            print(f"      - Shake score:  {shake:.3f}")
            print(f"      - Body score:   {body:.3f}")
            print(f"      - Micro score:  {micro:.3f}")
            print(f"      - Classified as: {classification}")

    # ========================================================================
    # STEP 5: Reconstruct selective flow
    # ========================================================================
    print("\n[5/6] Reconstructing selective flow...")
    print("-" * 70)

    clean_flow = editor.reconstruct_selective(weights)
    clean_flow_path = output_dir / 'clean_flow.npy'
    np.save(clean_flow_path, clean_flow)
    print(f"  ✓ Reconstructed flow shape: {clean_flow.shape}")

    # Compare magnitudes
    original_magnitude = np.sqrt(flow_sequence[:, 0]**2 + flow_sequence[:, 1]**2).mean()
    clean_magnitude = np.sqrt(clean_flow[:, 0]**2 + clean_flow[:, 1]**2).mean()

    print(f"\n  Motion Analysis:")
    print(f"    Original mean magnitude:  {original_magnitude:.2f}")
    print(f"    Cleaned mean magnitude:   {clean_magnitude:.2f}")
    print(f"    Motion reduction:         {(1 - clean_magnitude/original_magnitude)*100:.1f}%")

    print(f"\n  ✓ Saved clean flow to: {clean_flow_path}")

    # ========================================================================
    # STEP 6: Summary and next steps
    # ========================================================================
    print("\n[6/6] Pipeline Summary")
    print("-" * 70)

    print("\n  ✓ Frequency decomposition complete!")
    print(f"\n  Output files in: {output_dir}")
    print(f"    - raw_flows.npy         : Original optical flow")
    print(f"    - band_*_flow.npy       : Individual frequency bands")
    print(f"    - motion_analysis.json  : Detailed motion analysis")
    print(f"    - clean_flow.npy        : Reconstructed selective flow")

    print("\n  Next Steps:")
    print("    1. Use clean_flow.npy for noise warping:")
    print(f"       python make_warped_noise.py --flow_override {output_dir}/clean_flow.npy")
    print("\n    2. Or visualize the results:")
    print(f"       python test_frequency_decomposition.py {output_dir}")
    print("\n    3. For video generation, use cut_and_drag_inference.py")
    print("       with the warped noise from step 1")

    print("\n" + "="*70)
    print("FREQUENCY-DECOMPOSED MOTION TRANSFER COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
