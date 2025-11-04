"""
Unit tests for degradation control module.

Run with: pytest tests/test_degradation_control.py -v
Or without pytest: python tests/test_degradation_control.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from degradation_control import *

def test_backwards_compatibility_scalar():
    """Ensure scalar mode works identically to original degradation."""
    print("Testing backwards compatibility (scalar mode)...")

    config = DegradationConfig(mode='scalar', scalar_value=0.5)
    sample = torch.randn(1, 13, 16, 60, 90)
    random = torch.randn(1, 13, 16, 60, 90)

    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    expected = sample * 0.5 + random * 0.5

    assert torch.allclose(result, expected, atol=1e-5), "Scalar mode not backwards compatible!"
    print("   ✓ Scalar mode backwards compatible")


def test_scalar_edge_cases():
    """Test scalar degradation with edge values."""
    print("Testing scalar edge cases...")

    sample = torch.randn(1, 13, 16, 60, 90)
    random = torch.randn(1, 13, 16, 60, 90)

    # Test degradation = 0 (full warped noise)
    config = DegradationConfig(mode='scalar', scalar_value=0.0)
    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    assert torch.allclose(result, sample, atol=1e-5), "Degradation=0 should return warped noise"
    print("   ✓ Degradation=0 works (full warped noise)")

    # Test degradation = 1 (full random noise)
    config = DegradationConfig(mode='scalar', scalar_value=1.0)
    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    assert torch.allclose(result, random, atol=1e-5), "Degradation=1 should return random noise"
    print("   ✓ Degradation=1 works (full random noise)")


def test_temporal_interpolation():
    """Test temporal schedule interpolation."""
    print("Testing temporal schedule interpolation...")

    schedule = np.array([0.0, 1.0])
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    processor = DegradationProcessor(config, (1, 13, 16, 60, 90))

    tensor = processor.prepare_degradation_tensor('cpu')
    assert tensor.shape == (1, 13, 1, 1, 1), f"Wrong shape: {tensor.shape}"
    assert tensor[0, 0].item() < 0.1, "First frame should be close to 0"
    assert tensor[0, -1].item() > 0.9, "Last frame should be close to 1"
    print(f"   ✓ Temporal interpolation works: {tensor.shape}")


def test_spatial_resizing():
    """Test spatial mask resizing."""
    print("Testing spatial mask resizing...")

    mask = np.random.rand(30, 45).astype(np.float32)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    processor = DegradationProcessor(config, (1, 13, 16, 60, 90))

    tensor = processor.prepare_degradation_tensor('cpu')
    assert tensor.shape == (1, 1, 1, 60, 90), f"Wrong shape: {tensor.shape}"
    print(f"   ✓ Spatial resizing works: {mask.shape} → {tensor.shape}")


def test_spatiotemporal_mask():
    """Test spatiotemporal mask processing."""
    print("Testing spatiotemporal mask...")

    mask = np.random.rand(13, 60, 90).astype(np.float32)
    config = DegradationConfig(mode='spatiotemporal', spatiotemporal_mask=mask)
    processor = DegradationProcessor(config, (1, 13, 16, 60, 90))

    tensor = processor.prepare_degradation_tensor('cpu')
    assert tensor.shape == (1, 13, 1, 60, 90), f"Wrong shape: {tensor.shape}"
    print(f"   ✓ Spatiotemporal mask works: {mask.shape} → {tensor.shape}")


def test_temporal_schedules():
    """Test different temporal schedule types."""
    print("Testing temporal schedule types...")

    num_frames = 13

    # Linear
    schedule = create_temporal_schedule(num_frames, 'linear', 0.0, 1.0)
    assert len(schedule) == num_frames
    assert schedule[0] < 0.01 and schedule[-1] > 0.99
    print("   ✓ Linear schedule works")

    # Exponential
    schedule = create_temporal_schedule(num_frames, 'exponential', 0.0, 1.0, rate=2.0)
    assert len(schedule) == num_frames
    print("   ✓ Exponential schedule works")

    # Cosine
    schedule = create_temporal_schedule(num_frames, 'cosine', 0.0, 1.0)
    assert len(schedule) == num_frames
    print("   ✓ Cosine schedule works")

    # Pulse
    schedule = create_temporal_schedule(num_frames, 'pulse', 0.0, 1.0,
                                       pulse_frames=[5], pulse_value=1.0)
    assert len(schedule) == num_frames
    assert schedule[5] == 1.0
    print("   ✓ Pulse schedule works")


def test_spatial_masks():
    """Test different spatial mask types."""
    print("Testing spatial mask types...")

    height, width = 60, 90

    # Uniform
    mask = create_spatial_mask(height, width, 'uniform', value=0.5)
    assert mask.shape == (height, width)
    assert np.allclose(mask, 0.5)
    print("   ✓ Uniform mask works")

    # Radial
    mask = create_spatial_mask(height, width, 'radial',
                              inner_value=0.0, outer_value=1.0)
    assert mask.shape == (height, width)
    center_val = mask[height//2, width//2]
    edge_val = mask[0, 0]
    assert center_val < edge_val, "Radial mask should be lower at center"
    print("   ✓ Radial mask works")

    # Gradient
    mask = create_spatial_mask(height, width, 'gradient',
                              direction='horizontal',
                              start_value=0.0, end_value=1.0)
    assert mask.shape == (height, width)
    assert mask[0, 0] < mask[0, -1], "Gradient should increase"
    print("   ✓ Gradient mask works")

    # Rectangle
    mask = create_spatial_mask(height, width, 'rectangle',
                              x1=10, y1=10, x2=50, y2=50,
                              inside_value=1.0, outside_value=0.0)
    assert mask.shape == (height, width)
    assert mask[20, 20] == 1.0, "Inside rectangle should be 1.0"
    assert mask[0, 0] == 0.0, "Outside rectangle should be 0.0"
    print("   ✓ Rectangle mask works")


def test_config_save_load():
    """Test saving and loading configurations."""
    print("Testing config save/load...")

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test scalar config
        config = DegradationConfig(mode='scalar', scalar_value=0.7)
        path = os.path.join(tmpdir, 'test_scalar.json')
        DegradationIO.save_config(config, path)
        loaded = DegradationIO.load_config(path)
        assert loaded.mode == 'scalar'
        assert loaded.scalar_value == 0.7
        print("   ✓ Scalar save/load works")

        # Test temporal config
        schedule = create_temporal_schedule(13, 'linear', 0.0, 1.0)
        config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
        path = os.path.join(tmpdir, 'test_temporal.json')
        DegradationIO.save_config(config, path)
        loaded = DegradationIO.load_config(path)
        assert loaded.mode == 'temporal'
        assert np.allclose(loaded.temporal_schedule, schedule)
        print("   ✓ Temporal save/load works")


def test_broadcasting():
    """Test that degradation tensors broadcast correctly."""
    print("Testing broadcasting...")

    sample = torch.randn(1, 13, 16, 60, 90)
    random = torch.randn(1, 13, 16, 60, 90)

    # Temporal broadcasting
    schedule = create_temporal_schedule(13, 'linear', 0.0, 1.0)
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    assert result.shape == sample.shape
    print("   ✓ Temporal broadcasting works")

    # Spatial broadcasting
    mask = create_spatial_mask(60, 90, 'uniform', value=0.5)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    assert result.shape == sample.shape
    print("   ✓ Spatial broadcasting works")


def test_shape_flexibility():
    """Test handling of different input shapes."""
    print("Testing shape flexibility...")

    # Test with batch dimension (B, T, C, H, W)
    sample = torch.randn(1, 13, 16, 60, 90)
    random = torch.randn(1, 13, 16, 60, 90)
    config = DegradationConfig(mode='scalar', scalar_value=0.5)
    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    assert result.shape == (1, 13, 16, 60, 90)
    print("   ✓ Shape (B, T, C, H, W) works")

    # Test without batch dimension (T, C, H, W)
    sample = torch.randn(13, 16, 60, 90)
    random = torch.randn(13, 16, 60, 90)
    config = DegradationConfig(mode='scalar', scalar_value=0.5)
    result = apply_spatiotemporal_degradation(sample, random, config, 'cpu')
    assert result.shape == (13, 16, 60, 90)
    print("   ✓ Shape (T, C, H, W) works")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Degradation Control Unit Tests")
    print("=" * 70)
    print()

    tests = [
        test_backwards_compatibility_scalar,
        test_scalar_edge_cases,
        test_temporal_interpolation,
        test_spatial_resizing,
        test_spatiotemporal_mask,
        test_temporal_schedules,
        test_spatial_masks,
        test_config_save_load,
        test_broadcasting,
        test_shape_flexibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"   ✗ FAILED: {e}\n")

    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
