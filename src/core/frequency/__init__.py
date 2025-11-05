"""
CommonSource - Frequency-decomposed motion transfer components

This package contains modules for frequency-domain analysis and editing
of optical flow for selective motion transfer.
"""

from .frequency_motion_editor import FrequencyMotionEditor
from .motion_classifier import MotionClassifier

__all__ = ['FrequencyMotionEditor', 'MotionClassifier']
