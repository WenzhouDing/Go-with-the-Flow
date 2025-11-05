"""
Examples of creating degradation configurations for Go-with-the-Flow.

This script demonstrates how to create various temporal, spatial, and
spatiotemporal degradation patterns for controlling motion strength.

Run: python examples/create_degradation_configs.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degradation_control import *
import numpy as np

def create_all_examples():
    """Create all example configurations."""

    print("Creating example degradation configurations...\n")

    # ========================================================================
    # TEMPORAL EXAMPLES
    # ========================================================================

    print("1. Creating temporal schedule examples...")

    # Example 1a: Gradual temporal increase (linear)
    schedule = create_temporal_schedule(13, 'linear', 0.0, 1.0)
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    DegradationIO.save_config(config, 'configs/temporal_linear_increase.json')
    print("   ✓ Created: temporal_linear_increase.json")
    print(f"     Schedule: {schedule[:5]}... → ...{schedule[-5:]}")

    # Example 1b: Exponential temporal increase
    schedule = create_temporal_schedule(13, 'exponential', 0.0, 1.0, rate=2.0)
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    DegradationIO.save_config(config, 'configs/temporal_exponential.json')
    print("   ✓ Created: temporal_exponential.json")

    # Example 1c: Smooth S-curve (cosine)
    schedule = create_temporal_schedule(13, 'cosine', 0.0, 1.0)
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    DegradationIO.save_config(config, 'configs/temporal_cosine.json')
    print("   ✓ Created: temporal_cosine.json")

    # Example 1d: Pulse at specific frames
    schedule = create_temporal_schedule(13, 'pulse', 0.0, 1.0,
                                       pulse_frames=[3, 6, 9],
                                       pulse_value=1.0,
                                       pulse_width=1)
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    DegradationIO.save_config(config, 'configs/temporal_pulse.json')
    print("   ✓ Created: temporal_pulse.json")

    # Example 1e: Sinusoidal oscillation
    schedule = create_temporal_schedule(13, 'sinusoidal', 0.0, 1.0, frequency=2.0)
    config = DegradationConfig(mode='temporal', temporal_schedule=schedule)
    DegradationIO.save_config(config, 'configs/temporal_sinusoidal.json')
    print("   ✓ Created: temporal_sinusoidal.json\n")

    # ========================================================================
    # SPATIAL EXAMPLES
    # ========================================================================

    print("2. Creating spatial mask examples...")

    # Example 2a: Radial mask (center focused)
    mask = create_spatial_mask(60, 90, 'radial',
                              inner_value=0.0,
                              outer_value=1.0,
                              max_radius=40)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    DegradationIO.save_config(config, 'configs/spatial_radial_center.json')
    print("   ✓ Created: spatial_radial_center.json")
    print(f"     Mask shape: {mask.shape}, Range: [{mask.min():.2f}, {mask.max():.2f}]")

    # Example 2b: Radial mask (edge focused)
    mask = create_spatial_mask(60, 90, 'radial',
                              inner_value=1.0,
                              outer_value=0.0,
                              max_radius=40)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    DegradationIO.save_config(config, 'configs/spatial_radial_edge.json')
    print("   ✓ Created: spatial_radial_edge.json")

    # Example 2c: Horizontal gradient (left to right)
    mask = create_spatial_mask(60, 90, 'gradient',
                              direction='horizontal',
                              start_value=0.0,
                              end_value=1.0)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    DegradationIO.save_config(config, 'configs/spatial_gradient_horizontal.json')
    print("   ✓ Created: spatial_gradient_horizontal.json")

    # Example 2d: Vertical gradient (top to bottom)
    mask = create_spatial_mask(60, 90, 'gradient',
                              direction='vertical',
                              start_value=0.0,
                              end_value=1.0)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    DegradationIO.save_config(config, 'configs/spatial_gradient_vertical.json')
    print("   ✓ Created: spatial_gradient_vertical.json")

    # Example 2e: Rectangle region
    mask = create_spatial_mask(60, 90, 'rectangle',
                              x1=20, y1=15,
                              x2=70, y2=45,
                              inside_value=1.0,
                              outside_value=0.0)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    DegradationIO.save_config(config, 'configs/spatial_rectangle.json')
    print("   ✓ Created: spatial_rectangle.json")

    # Example 2f: Ellipse region
    mask = create_spatial_mask(60, 90, 'ellipse',
                              center_x=45, center_y=30,
                              radius_x=25, radius_y=15,
                              inside_value=1.0,
                              outside_value=0.0)
    config = DegradationConfig(mode='spatial', spatial_mask=mask)
    DegradationIO.save_config(config, 'configs/spatial_ellipse.json')
    print("   ✓ Created: spatial_ellipse.json\n")

    # ========================================================================
    # SPATIOTEMPORAL EXAMPLES
    # ========================================================================

    print("3. Creating spatiotemporal mask examples...")

    # Example 3a: Moving spotlight (left to right)
    num_frames, height, width = 13, 60, 90
    mask = np.zeros((num_frames, height, width), dtype=np.float32)
    for t in range(num_frames):
        center_x = int((t / (num_frames - 1)) * width)
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - center_x)**2 + (y - height/2)**2)
        mask[t] = np.clip(1.0 - dist / 20, 0, 1)

    config = DegradationConfig(mode='spatiotemporal', spatiotemporal_mask=mask)
    DegradationIO.save_config(config, 'configs/spatiotemporal_moving_spotlight.json')
    print("   ✓ Created: spatiotemporal_moving_spotlight.json")
    print(f"     Mask shape: {mask.shape}")

    # Example 3b: Expanding circle
    mask = np.zeros((num_frames, height, width), dtype=np.float32)
    for t in range(num_frames):
        radius = (t / (num_frames - 1)) * min(height, width) / 2
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask[t] = np.where(dist <= radius, 1.0, 0.0)

    config = DegradationConfig(mode='spatiotemporal', spatiotemporal_mask=mask)
    DegradationIO.save_config(config, 'configs/spatiotemporal_expanding_circle.json')
    print("   ✓ Created: spatiotemporal_expanding_circle.json")

    # Example 3c: Fading gradient (temporal + spatial combined)
    mask = np.zeros((num_frames, height, width), dtype=np.float32)
    for t in range(num_frames):
        temporal_factor = t / (num_frames - 1)
        # Horizontal gradient
        spatial_gradient = np.linspace(0, 1, width)[None, :]
        spatial_gradient = np.repeat(spatial_gradient, height, axis=0)
        # Combine temporal and spatial
        mask[t] = spatial_gradient * temporal_factor

    config = DegradationConfig(mode='spatiotemporal', spatiotemporal_mask=mask)
    DegradationIO.save_config(config, 'configs/spatiotemporal_fading_gradient.json')
    print("   ✓ Created: spatiotemporal_fading_gradient.json")

    # Example 3d: Alternating regions
    mask = np.zeros((num_frames, height, width), dtype=np.float32)
    for t in range(num_frames):
        if t % 2 == 0:
            # Left half
            mask[t, :, :width//2] = 1.0
        else:
            # Right half
            mask[t, :, width//2:] = 1.0

    config = DegradationConfig(mode='spatiotemporal', spatiotemporal_mask=mask)
    DegradationIO.save_config(config, 'configs/spatiotemporal_alternating_regions.json')
    print("   ✓ Created: spatiotemporal_alternating_regions.json\n")

    # ========================================================================
    # SCALAR EXAMPLES (for completeness)
    # ========================================================================

    print("4. Creating scalar examples...")

    # Example 4a: Low degradation (strong motion control)
    config = DegradationConfig(mode='scalar', scalar_value=0.1)
    DegradationIO.save_config(config, 'configs/scalar_low_0.1.json')
    print("   ✓ Created: scalar_low_0.1.json")

    # Example 4b: Medium degradation
    config = DegradationConfig(mode='scalar', scalar_value=0.5)
    DegradationIO.save_config(config, 'configs/scalar_medium_0.5.json')
    print("   ✓ Created: scalar_medium_0.5.json")

    # Example 4c: High degradation (weak motion control)
    config = DegradationConfig(mode='scalar', scalar_value=0.9)
    DegradationIO.save_config(config, 'configs/scalar_high_0.9.json')
    print("   ✓ Created: scalar_high_0.9.json\n")

    print("=" * 70)
    print("All example configurations created successfully!")
    print("Total configs created: 20")
    print("\nTo use these configs, pass them to cut_and_drag_inference.py:")
    print("  python cut_and_drag_inference.py noise_output/ \\")
    print("    --degradation configs/temporal_linear_increase.json")
    print("=" * 70)


if __name__ == '__main__':
    create_all_examples()
