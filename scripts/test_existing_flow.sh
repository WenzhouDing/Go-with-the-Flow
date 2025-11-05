#!/bin/bash
# Test FFT with existing flow data

conda activate flow_warp

echo "=== Testing FFT with Existing Flow Data ==="
echo ""
echo "Loading flow from: my_warped_noise/flows_dxdy.npy"
echo ""

python << 'PYTHON_EOF'
import numpy as np
from CommonSource.frequency_motion_editor import FrequencyMotionEditor
from CommonSource.motion_classifier import MotionClassifier

print("Loading existing flow data...")
flow = np.load('my_warped_noise/flows_dxdy.npy')
print(f"✓ Flow shape: {flow.shape}")
print(f"  Mean magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.2f}")
print(f"  Max magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).max():.2f}")
print(f"  Std magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).std():.2f}")

print("\nApplying FFT decomposition (4 bands)...")
editor = FrequencyMotionEditor(flow, fps=12)
bands = editor.decompose_frequencies(num_bands=4)
print(f"✓ Created {len(bands)} frequency bands")

print("\nAnalyzing motion spectrum...")
analysis = editor.analyze_motion_spectrum()

print("\n" + "="*60)
MotionClassifier.print_analysis_report(analysis)
print("="*60)

print("\nRecommending band weights...")
weights = MotionClassifier.recommend_band_weights(analysis, remove_shake=True)
print(f"Recommended weights: {weights}")

print("\nApplying selective reconstruction...")
clean_flow = editor.reconstruct_selective(weights)

orig_mag = np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean()
clean_mag = np.sqrt(clean_flow[:,0]**2 + clean_flow[:,1]**2).mean()
reduction = (1 - clean_mag/orig_mag) * 100

print(f"\nResults:")
print(f"  Original magnitude: {orig_mag:.2f}")
print(f"  Filtered magnitude: {clean_mag:.2f}")
print(f"  Motion reduction: {reduction:.1f}%")

np.save('fft_filtered_flow.npy', clean_flow)
print(f"\n✓ Filtered flow saved to: fft_filtered_flow.npy")
print("\nDone!")

PYTHON_EOF

