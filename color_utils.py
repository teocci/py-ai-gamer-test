"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-28
"""
import cv2
import numpy as np


def is_red_area(bgr_image: np.ndarray, pixel_threshold=1000) -> bool:
    """
    Return True if there's a large enough cluster of red pixels in bgr_image,
    based on HSV color space detection.
    """
    # Convert BGR -> HSV
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Red in HSV can wrap around hue=0/180, so we define two ranges
    lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 120, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Combine two red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Count how many nonzero (red) pixels
    red_pixels = np.count_nonzero(mask)
    return red_pixels > pixel_threshold


def is_green_area(bgr_image: np.ndarray, pixel_threshold=500) -> bool:
    """
    Check if the image region contains a significant area of green.
    Condition: green channel > 150 and blue & red channels < 100.
    """
    mask = (
            (bgr_image[:, :, 1] > 150) &
            (bgr_image[:, :, 0] < 100) &
            (bgr_image[:, :, 2] < 100)
    )
    return np.sum(mask) > pixel_threshold


def is_yellow_area(bgr_image: np.ndarray, pixel_threshold=200) -> bool:
    """
    Check if the image region contains a significant area of yellow.
    Condition: blue channel < 100 and green & red channels > 150.
    """
    mask = (
            (bgr_image[:, :, 0] < 100) &
            (bgr_image[:, :, 1] > 150) &
            (bgr_image[:, :, 2] > 150)
    )
    return np.sum(mask) > pixel_threshold


def is_purple_area(bgr_image: np.ndarray, pixel_threshold=100) -> bool:
    """
    Check if the image region contains a significant area of purple.
    Condition: blue and red channels > 100 and green channel < 100.
    """
    mask = (
            (bgr_image[:, :, 0] > 100) &
            (bgr_image[:, :, 2] > 100) &
            (bgr_image[:, :, 1] < 100)
    )
    return np.sum(mask) > pixel_threshold

def is_purple_area_hsv(bgr_image: np.ndarray, pixel_threshold=1000) -> bool:
    """
    Check if the image region contains a significant area of purple.
    Uses HSV color space detection to identify purple pixels.
    """
    # Convert BGR -> HSV
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for purple
    lower_purple = np.array([120, 50, 50], dtype=np.uint8)  # Adjust as needed
    upper_purple = np.array([160, 255, 255], dtype=np.uint8)

    # Create a mask for purple color
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Count how many purple pixels exist
    purple_pixels = np.count_nonzero(mask)
    return purple_pixels > pixel_threshold

def is_dark_area(bgr_image: np.ndarray, color: tuple, pixel_threshold=1000) -> bool:
    """
    Check if the image region contains a significant area of a specific dark color.
    The color is provided as (B, G, R), and we allow a small tolerance.
    """
    tolerance = 10  # Allow slight variations in the color
    lower_bound = np.array([max(c - tolerance, 0) for c in color], dtype=np.uint8)
    upper_bound = np.array([min(c + tolerance, 255) for c in color], dtype=np.uint8)

    # Create a mask for the specified color
    mask = cv2.inRange(bgr_image, lower_bound, upper_bound)

    # Count how many pixels match the color
    matching_pixels = np.count_nonzero(mask)
    return matching_pixels > pixel_threshold


def is_white_area(bgr_image: np.ndarray, pixel_threshold=50) -> bool:
    """
    Check if the image region contains a significant area of white.
    Condition: all channels > 200.
    """
    mask = (
            (bgr_image[:, :, 0] > 200) &
            (bgr_image[:, :, 1] > 200) &
            (bgr_image[:, :, 2] > 200)
    )
    return np.sum(mask) > pixel_threshold
