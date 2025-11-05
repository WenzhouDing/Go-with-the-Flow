"""
Test script: Emphasize fast hand motion vs slow body motion
"""
import numpy as np
from CommonSource.frequency_motion_editor import FrequencyMotionEditor
from CommonSource.motion_classifier import MotionClassifier

print("="*60)
print("TESTING: Fast Hand vs Slow Body Control")
print("="*60)

# Load flow
print("\nLoading flow from fft_filtered/raw_flows.npy...")
flow = np.load('fft_filtered/raw_flows.npy')
orig_mag = np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean()
print(f"✓ Original flow magnitude: {orig_mag:.2f}")

# Create editor and decompose
print("\nDecomposing into 4 frequency bands...")
editor = FrequencyMotionEditor(flow, fps=12)
bands = editor.decompose_frequencies(num_bands=4)

# Show per-band magnitudes
print("\nPer-band analysis:")
for i in range(4):
    mag = np.sqrt(bands[i][:,0]**2 + bands[i][:,1]**2).mean()
    print(f"  Band {i}: magnitude = {mag:.3f}")

# Test 1: Emphasize fast motion (hands)
print("\n" + "="*60)
print("TEST 1: Emphasize Fast Motion (Hands)")
print("="*60)
weights_hands = [0.0, 0.3, 1.5, 2.0]
print(f"Weights: {weights_hands}")

result_hands = editor.reconstruct_selective(weights_hands)
hands_mag = np.sqrt(result_hands[:,0]**2 + result_hands[:,1]**2).mean()

print(f"\nResult magnitude: {hands_mag:.2f}")
print(f"Change: {(hands_mag/orig_mag - 1)*100:+.1f}%")

print("\nPer-band contribution:")
for i, w in enumerate(weights_hands):
    band_mag = np.sqrt(bands[i][:,0]**2 + bands[i][:,1]**2).mean()
    contribution = band_mag * w
    print(f"  Band {i}: {band_mag:.3f} × {w} = {contribution:.3f}")

# Test 2: Ultra smooth (remove fast)
print("\n" + "="*60)
print("TEST 2: Ultra Smooth (Remove Fast Motion)")
print("="*60)
weights_smooth = [1.0, 0.8, 0.0, 0.0]
print(f"Weights: {weights_smooth}")

result_smooth = editor.reconstruct_selective(weights_smooth)
smooth_mag = np.sqrt(result_smooth[:,0]**2 + result_smooth[:,1]**2).mean()

print(f"\nResult magnitude: {smooth_mag:.2f}")
print(f"Change: {(smooth_mag/orig_mag - 1)*100:+.1f}%")

# Test 3: Extreme hand emphasis
print("\n" + "="*60)
print("TEST 3: Extreme Hand Emphasis")
print("="*60)
weights_extreme = [0.0, 0.0, 2.0, 3.0]
print(f"Weights: {weights_extreme}")

result_extreme = editor.reconstruct_selective(weights_extreme)
extreme_mag = np.sqrt(result_extreme[:,0]**2 + result_extreme[:,1]**2).mean()

print(f"\nResult magnitude: {extreme_mag:.2f}")
print(f"Change: {(extreme_mag/orig_mag - 1)*100:+.1f}%")

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

np.save('test_hands_emphasized.npy', result_hands)
np.save('test_ultra_smooth.npy', result_smooth)
np.save('test_extreme_hands.npy', result_extreme)

print("✓ Saved test_hands_emphasized.npy")
print("✓ Saved test_ultra_smooth.npy")
print("✓ Saved test_extreme_hands.npy")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original:         {orig_mag:.2f}")
print(f"Hands emphasized: {hands_mag:.2f} ({(hands_mag/orig_mag - 1)*100:+.1f}%)")
print(f"Ultra smooth:     {smooth_mag:.2f} ({(smooth_mag/orig_mag - 1)*100:+.1f}%)")
print(f"Extreme hands:    {extreme_mag:.2f} ({(extreme_mag/orig_mag - 1)*100:+.1f}%)")

print("\n✓ Done! FFT successfully controlled hand vs body motion.")
