"""
motion_classifier.py

Classify motion types based on frequency characteristics.
Helps identify camera shake, body movement, micro-motion, etc.
"""

import numpy as np
from typing import Dict, List, Tuple


class MotionClassifier:
    """
    Classify motion types based on frequency characteristics.
    Helps identify camera shake, body movement, micro-motion, etc.
    """

    @staticmethod
    def identify_camera_shake(freq_analysis: dict,
                            band_idx: int) -> float:
        """
        Score likelihood that a band contains camera shake.
        Camera shake typically has high spatial uniformity and medium frequency.

        Args:
            freq_analysis: Motion spectrum analysis from FrequencyMotionEditor
            band_idx: Index of the frequency band to analyze

        Returns:
            Score between 0.0 and 1.0 indicating likelihood of camera shake
        """
        band_stats = freq_analysis[f'band_{band_idx}']

        # High spatial uniformity (low spatial variance)
        spatial_uniformity = 1.0 / (1.0 + band_stats['spatial_variance'])

        # Medium to high temporal variance
        temporal_activity = band_stats['temporal_variance']

        # Moderate magnitude
        magnitude_score = 1.0 / (1.0 + abs(band_stats['mean_magnitude'] - 5.0))

        shake_score = spatial_uniformity * 0.5 + \
                     temporal_activity * 0.3 + \
                     magnitude_score * 0.2

        return min(shake_score, 1.0)  # Clamp to [0, 1]

    @staticmethod
    def identify_body_movement(freq_analysis: dict,
                              band_idx: int) -> float:
        """
        Score likelihood that a band contains primary body/object movement.
        Typically lower frequency with high spatial variance.

        Args:
            freq_analysis: Motion spectrum analysis from FrequencyMotionEditor
            band_idx: Index of the frequency band to analyze

        Returns:
            Score between 0.0 and 1.0 indicating likelihood of body movement
        """
        band_stats = freq_analysis[f'band_{band_idx}']

        # High spatial variance (localized movement)
        spatial_locality = band_stats['spatial_variance']

        # Consistent temporal pattern
        temporal_consistency = 1.0 / (1.0 + band_stats['temporal_variance'])

        # Higher magnitude
        magnitude_score = band_stats['mean_magnitude'] / 10.0

        movement_score = spatial_locality * 0.4 + \
                        temporal_consistency * 0.3 + \
                        magnitude_score * 0.3

        return min(movement_score, 1.0)

    @staticmethod
    def identify_micro_motion(freq_analysis: dict,
                             band_idx: int) -> float:
        """
        Score likelihood that a band contains micro-motions (breathing, fabric).
        Typically highest frequency with low magnitude.

        Args:
            freq_analysis: Motion spectrum analysis from FrequencyMotionEditor
            band_idx: Index of the frequency band to analyze

        Returns:
            Score between 0.0 and 1.0 indicating likelihood of micro-motion
        """
        band_stats = freq_analysis[f'band_{band_idx}']

        # Low magnitude
        low_magnitude = 1.0 / (1.0 + band_stats['mean_magnitude'])

        # Low temporal variance (continuous)
        temporal_smoothness = 1.0 / (1.0 + band_stats['temporal_variance'])

        micro_score = low_magnitude * 0.6 + temporal_smoothness * 0.4

        return micro_score

    @staticmethod
    def recommend_band_weights(freq_analysis: dict,
                              remove_shake: bool = True,
                              preserve_micro: float = 0.5) -> List[float]:
        """
        Recommend band weights based on motion analysis.

        Args:
            freq_analysis: Motion spectrum analysis from FrequencyMotionEditor
            remove_shake: Whether to remove camera shake
            preserve_micro: Scale factor for micro-motions (0-1)

        Returns:
            List of recommended weights for each frequency band
        """
        num_bands = len(freq_analysis)
        weights = []

        for i in range(num_bands):
            shake_score = MotionClassifier.identify_camera_shake(freq_analysis, i)
            body_score = MotionClassifier.identify_body_movement(freq_analysis, i)
            micro_score = MotionClassifier.identify_micro_motion(freq_analysis, i)

            # Decision logic
            if remove_shake and shake_score > 0.7:
                weight = 0.0  # Remove camera shake
            elif body_score > 0.6:
                weight = 1.0  # Keep body movement
            elif micro_score > 0.5:
                weight = preserve_micro  # Partially keep micro-motion
            else:
                weight = 0.8  # Default: mostly keep

            weights.append(weight)

        return weights

    @staticmethod
    def classify_band(freq_analysis: dict, band_idx: int) -> str:
        """
        Classify a frequency band into a motion type category.

        Args:
            freq_analysis: Motion spectrum analysis
            band_idx: Index of the frequency band

        Returns:
            String describing the dominant motion type in this band
        """
        shake_score = MotionClassifier.identify_camera_shake(freq_analysis, band_idx)
        body_score = MotionClassifier.identify_body_movement(freq_analysis, band_idx)
        micro_score = MotionClassifier.identify_micro_motion(freq_analysis, band_idx)

        scores = {
            'camera_shake': shake_score,
            'body_movement': body_score,
            'micro_motion': micro_score
        }

        # Return the category with highest score
        dominant_type = max(scores.items(), key=lambda x: x[1])

        return dominant_type[0]

    @staticmethod
    def print_analysis_report(freq_analysis: dict) -> None:
        """
        Print a detailed analysis report of all frequency bands.

        Args:
            freq_analysis: Motion spectrum analysis
        """
        print("\n" + "="*60)
        print("FREQUENCY BAND ANALYSIS REPORT")
        print("="*60)

        num_bands = len(freq_analysis)

        for i in range(num_bands):
            print(f"\n--- Band {i} ---")
            band_stats = freq_analysis[f'band_{i}']

            print(f"  Mean Magnitude:      {band_stats['mean_magnitude']:.2f}")
            print(f"  Std Magnitude:       {band_stats['std_magnitude']:.2f}")
            print(f"  Max Magnitude:       {band_stats['max_magnitude']:.2f}")
            print(f"  Temporal Variance:   {band_stats['temporal_variance']:.4f}")
            print(f"  Spatial Variance:    {band_stats['spatial_variance']:.4f}")
            print(f"  Dominant Direction:  {band_stats['dominant_direction']:.1f}°")

            # Classification scores
            shake_score = MotionClassifier.identify_camera_shake(freq_analysis, i)
            body_score = MotionClassifier.identify_body_movement(freq_analysis, i)
            micro_score = MotionClassifier.identify_micro_motion(freq_analysis, i)

            print(f"\n  Classification Scores:")
            print(f"    Camera Shake:      {shake_score:.3f}")
            print(f"    Body Movement:     {body_score:.3f}")
            print(f"    Micro Motion:      {micro_score:.3f}")

            classification = MotionClassifier.classify_band(freq_analysis, i)
            print(f"  Dominant Type: {classification.upper()}")

        print("\n" + "="*60)
