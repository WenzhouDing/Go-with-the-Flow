"""
Integrate FFT-filtered flow into video generation pipeline

This script shows how to use frequency-filtered flows (e.g., hands emphasized)
in the actual video generation process.
"""
import numpy as np
import sys
from pathlib import Path

def integrate_filtered_flow(
    filtered_flow_path: str,
    target_folder: str,
    backup_original: bool = True
):
    """
    Replace the flow in a warped noise folder with filtered flow.

    Args:
        filtered_flow_path: Path to FFT-filtered flow (e.g., test_hands_emphasized.npy)
        target_folder: Folder containing flows_dxdy.npy (e.g., my_warped_noise/)
        backup_original: Whether to backup the original flow first
    """
    filtered_flow = np.load(filtered_flow_path)
    target_path = Path(target_folder) / 'flows_dxdy.npy'

    if not target_path.exists():
        print(f"❌ Error: {target_path} does not exist!")
        print(f"   Make sure {target_folder} contains flows_dxdy.npy")
        return False

    # Backup original
    if backup_original:
        backup_path = Path(target_folder) / 'flows_dxdy_original.npy'
        if not backup_path.exists():
            original = np.load(target_path)
            np.save(backup_path, original)
            print(f"✓ Backed up original flow to: {backup_path}")

    # Load original to compare shapes
    original = np.load(target_path)
    print(f"\nOriginal flow shape: {original.shape}")
    print(f"Filtered flow shape: {filtered_flow.shape}")

    if original.shape != filtered_flow.shape:
        print(f"❌ Error: Shape mismatch!")
        print(f"   Original: {original.shape}")
        print(f"   Filtered: {filtered_flow.shape}")
        return False

    # Compare magnitudes
    orig_mag = np.sqrt(original[:,0]**2 + original[:,1]**2).mean()
    filt_mag = np.sqrt(filtered_flow[:,0]**2 + filtered_flow[:,1]**2).mean()

    print(f"\nMagnitude comparison:")
    print(f"  Original: {orig_mag:.3f}")
    print(f"  Filtered: {filt_mag:.3f}")
    print(f"  Change: {(filt_mag/orig_mag - 1)*100:+.1f}%")

    # Save filtered flow
    np.save(target_path, filtered_flow)
    print(f"\n✓ Replaced {target_path} with filtered flow")
    print(f"\nNow you can re-generate video with:")
    print(f"  python cut_and_drag_inference.py \\")
    print(f"    --folder_path {target_folder} \\")
    print(f"    --degradation 0.0 \\")
    print(f"    --output_name hands_emphasized.mp4")

    return True


def compare_flows(flow1_path: str, flow2_path: str):
    """Compare two flow sequences visually."""
    flow1 = np.load(flow1_path)
    flow2 = np.load(flow2_path)

    print(f"\n{'='*60}")
    print(f"FLOW COMPARISON")
    print(f"{'='*60}")
    print(f"\nFlow 1: {flow1_path}")
    print(f"  Shape: {flow1.shape}")
    print(f"  Magnitude: {np.sqrt(flow1[:,0]**2 + flow1[:,1]**2).mean():.3f}")
    print(f"  Max magnitude: {np.sqrt(flow1[:,0]**2 + flow1[:,1]**2).max():.3f}")

    print(f"\nFlow 2: {flow2_path}")
    print(f"  Shape: {flow2.shape}")
    print(f"  Magnitude: {np.sqrt(flow2[:,0]**2 + flow2[:,1]**2).mean():.3f}")
    print(f"  Max magnitude: {np.sqrt(flow2[:,0]**2 + flow2[:,1]**2).max():.3f}")

    # Per-frame comparison
    mag1 = np.sqrt(flow1[:,0]**2 + flow1[:,1]**2).mean(axis=(1,2))
    mag2 = np.sqrt(flow2[:,0]**2 + flow2[:,1]**2).mean(axis=(1,2))

    print(f"\nPer-frame magnitude:")
    print(f"  Flow 1 range: [{mag1.min():.2f}, {mag1.max():.2f}]")
    print(f"  Flow 2 range: [{mag2.min():.2f}, {mag2.max():.2f}]")
    print(f"  Correlation: {np.corrcoef(mag1, mag2)[0,1]:.3f}")


def main():
    print("="*60)
    print("FFT-Filtered Flow Integration")
    print("="*60)

    # Example 1: Compare filtered flows
    print("\n[1] Comparing different filtered versions...")
    if Path('fft_filtered/raw_flows.npy').exists():
        if Path('test_hands_emphasized.npy').exists():
            compare_flows('fft_filtered/raw_flows.npy', 'test_hands_emphasized.npy')
        if Path('test_ultra_smooth.npy').exists():
            print("\n" + "-"*60)
            compare_flows('fft_filtered/raw_flows.npy', 'test_ultra_smooth.npy')

    # Example 2: Integrate into pipeline
    print("\n\n[2] Integration example:")
    print("\nTo use hands-emphasized flow in generation:")
    print("  python integrate_filtered_flow.py integrate \\")
    print("    test_hands_emphasized.npy my_warped_noise/")

    print("\n\n[3] Complete workflow:")
    print("  # Step 1: Analyze and filter")
    print("  python test_hand_emphasis.py")
    print()
    print("  # Step 2: Integrate filtered flow")
    print("  python integrate_filtered_flow.py integrate \\")
    print("    test_hands_emphasized.npy my_warped_noise/")
    print()
    print("  # Step 3: Generate video with filtered motion")
    print("  conda activate flow")
    print("  python cut_and_drag_inference.py \\")
    print("    --folder_path my_warped_noise \\")
    print("    --degradation 0.0 \\")
    print("    --output_name hands_emphasized.mp4")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'integrate':
        if len(sys.argv) < 4:
            print("Usage: python integrate_filtered_flow.py integrate <filtered_flow.npy> <target_folder>")
            sys.exit(1)

        filtered_flow_path = sys.argv[2]
        target_folder = sys.argv[3]

        integrate_filtered_flow(filtered_flow_path, target_folder)
    else:
        main()
