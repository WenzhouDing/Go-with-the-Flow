"""
Create horizontal 3-way comparison video (no text labels).
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def create_three_way_comparison(video1_path, video2_path, video3_path, output_path):
    """
    Create horizontal 3-way comparison video without text labels.

    Args:
        video1_path: Path to first video (left)
        video2_path: Path to second video (center)
        video3_path: Path to third video (right)
        output_path: Path to save comparison video
    """
    print(f"Creating 3-way horizontal comparison...")
    print(f"  Left:   {video1_path}")
    print(f"  Center: {video2_path}")
    print(f"  Right:  {video3_path}")

    # Open all three videos
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    cap3 = cv2.VideoCapture(str(video3_path))

    if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
        raise ValueError("Could not open one or more videos")

    # Get video properties from all three videos
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width3 = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH))
    height3 = int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video 1: {width1}x{height1} @ {fps1} fps")
    print(f"  Video 2: {width2}x{height2}")
    print(f"  Video 3: {width3}x{height3}")

    # Use the maximum dimensions as target (or use video 2's dimensions)
    target_height = max(height1, height2, height3)
    target_width = int(target_height * width2 / height2)  # Preserve aspect ratio of center video

    print(f"  Target resolution for all panels: {target_width}x{target_height}")

    # Setup output
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    padding = 10
    output_width = target_width * 3 + padding * 4
    output_height = target_height + padding * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps1, (output_width, output_height))

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not ret1 or not ret2 or not ret3:
            break

        # Resize all frames to target resolution
        frame1 = cv2.resize(frame1, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame3 = cv2.resize(frame3, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Create white canvas
        canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        # Place three frames horizontally with padding
        y = padding

        # Left video
        x = padding
        canvas[y:y+target_height, x:x+target_width] = frame1

        # Center video
        x = padding * 2 + target_width
        canvas[y:y+target_height, x:x+target_width] = frame2

        # Right video
        x = padding * 3 + target_width * 2
        canvas[y:y+target_height, x:x+target_width] = frame3

        out.write(canvas)
        frame_count += 1

    cap1.release()
    cap2.release()
    cap3.release()
    out.release()

    print(f"✓ Created 3-way comparison video: {output_path}")
    print(f"  Frames: {frame_count}")
    print(f"  Output size: {output_width}x{output_height}")


def main():
    parser = argparse.ArgumentParser(
        description='Create horizontal 3-way video comparison without text labels'
    )
    parser.add_argument('video1', help='First video path (left)')
    parser.add_argument('video2', help='Second video path (center)')
    parser.add_argument('video3', help='Third video path (right)')
    parser.add_argument('output', help='Output video path')

    args = parser.parse_args()

    create_three_way_comparison(
        args.video1,
        args.video2,
        args.video3,
        args.output
    )


if __name__ == '__main__':
    main()
