import cv2
from typing import List, Tuple
import numpy as np


def detect_color(img : np.ndarray, colors : List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]], threshold : int = 5000):
    """
    Detects the predominant color in an image from a list of HSV color ranges.

    Parameters:
    -----------
    img : np.ndarray
        Input image in BGR format (as read by OpenCV).
    colors : List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
        List of HSV color ranges to detect. Each element is a tuple: (lower_HSV, upper_HSV).
    threshold : int, optional (default=5000)
        Minimum number of pixels required for a color to be considered detected.

    Returns:
    --------
    int or None
        The index of the detected color in the `colors` list if any color exceeds the threshold;
        otherwise, returns None.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    masks = [cv2.inRange(hsv_img, color[0], color[1]) for color in colors]

    kernel = np.ones((5,5), np.uint8)
    masks = [cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) for mask in masks]
    masks = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in masks]

    area_for_each_color = [cv2.countNonZero(mask) for mask in masks]
    max_area = max(area_for_each_color)
    if max_area > threshold:   # Threshold for a true detection
        detected_color = area_for_each_color.index(max_area)
        return detected_color
    else:
        return None

