"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-27
Description: Improved interface for interacting with Monanimal Mayhem game on dual monitor setup.
             Optimized for reinforcement learning with better state representation and screen detection.
"""
import logging
import os
import random
import time
import tkinter as tk
from enum import Enum
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from screeninfo import get_monitors

from agent import GameLearningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='game_interface.log'
)
logger = logging.getLogger(__name__)

# Constant for the game window handle - can be updated at runtime
GAME_HWND = 14623178

# Ensure PyAutoGUI doesn't move the mouse too fast
pyautogui.PAUSE = 0.1  # Add small pause between PyAutoGUI commands

# Create directories if they don't exist
SCREENS_DIR = "screens"
os.makedirs(SCREENS_DIR, exist_ok=True)

# Game UI element locations (based on ratios relative to window size)
UI_ELEMENTS = {
    "play_button": (0.5, 0.82),
    "normal_mode": (0.5, 0.48),
    "restart_button": (0.5, 0.83),
    "level_up_option_1": (0.3, 0.55),
    "level_up_option_2": (0.5, 0.55),
    "level_up_option_3": (0.7, 0.55),
    "pause_button": (0.82, 0.15),
    "shop_button": (0.5, 0.55),
}


# Game screen types
class GameScreen(Enum):
    MAIN_MENU = "main_menu"
    GAME_MODE = "game_mode"
    GAMEPLAY = "gameplay"
    LEVEL_UP = "level_up"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


class GameInterface:
    """Class for interfacing with the Monanimal Mayhem game, optimized for reinforcement learning."""

    def __init__(self, hwnd=None):
        """Initialize the game interface with optional window handle."""
        self.hwnd = hwnd or GAME_HWND
        self.window = None
        self.left_monitor = None
        self.current_screen = None
        self.previous_screenshot = None
        self.frame_buffer = []  # Store recent frames for state representation
        self.frame_buffer_size = 4  # Number of frames to keep for temporal information

        # Game state for RL
        self.game_state = {
            "level": 1,
            "health": 100,
            "coins": 0,
            "score": 0,
            "abilities": [],
            "time_alive": 0,
            "monsters_avoided": 0,
            "monsters_killed": 0,
            "game_over": False,
            "reward": 0
        }

        # Initialize monitor info
        self._initialize_monitor_info()

        # Get the game window directly by handle
        self.window = self.get_window_by_hwnd(self.hwnd)
        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")

        # Move window to left monitor if needed
        if not self.is_window_on_left_monitor(self.window):
            logger.info("Game window not on left monitor. Moving it...")
            self.move_window_to_left_monitor(self.window)

        # Debug info
        logger.info(f"Left monitor: x={self.left_monitor.x}, y={self.left_monitor.y}, "
                    f"width={self.left_monitor.width}, height={self.left_monitor.height}")
        logger.info(f"Game window: title='{self.window.title}', handle={self.window._hWnd}, "
                    f"position=({self.window.left}, {self.window.top}), "
                    f"size={self.window.width}x{self.window.height}")

        # Initialize reference features for screen detection
        self._initialize_screen_detection()

        # For RL, prefill frame buffer with initial state
        self._prefill_frame_buffer()

    def _initialize_monitor_info(self):
        """Initialize information about monitors."""
        monitors = get_monitors()
        if not monitors:
            raise Exception("No monitors found.")

        # Find the leftmost monitor (smallest x-coordinate)
        self.left_monitor = min(monitors, key=lambda m: m.x)
        logger.info(f"Detected {len(monitors)} monitors. Using leftmost monitor: {self.left_monitor}")

    def _initialize_screen_detection(self):
        """Initialize reference features for screen detection based on UI elements rather than template matching."""
        # Instead of using full screenshot templates, we'll use color histograms and UI element detection
        self.screen_features = {
            GameScreen.MAIN_MENU: {
                "ui_elements": ["play_button"],
                "color_signature": None,  # Will be populated when first encountered
            },
            GameScreen.GAME_MODE: {
                "ui_elements": ["normal_mode"],
                "color_signature": None,
            },
            GameScreen.GAMEPLAY: {
                "ui_elements": ["pause_button"],
                "color_signature": None,
            },
            GameScreen.LEVEL_UP: {
                "ui_elements": ["level_up_option_1", "level_up_option_2", "level_up_option_3"],
                "color_signature": None,
            },
            GameScreen.GAME_OVER: {
                "ui_elements": ["restart_button"],
                "color_signature": None,
            }
        }

    def _prefill_frame_buffer(self):
        """Initialize frame buffer with copies of the current state for RL."""
        initial_frame = self.get_game_state_representation()
        for _ in range(self.frame_buffer_size):
            self.frame_buffer.append(initial_frame.copy())

    def get_window_by_hwnd(self, hwnd) -> Optional[gw.Win32Window]:
        """
        Get a window by its window handle (_hWnd).
        Returns the window if found, None otherwise.
        """
        for window in gw.getAllWindows():
            if window._hWnd == hwnd:
                return window

        logger.warning(f"No window found with handle {hwnd}")
        return None

    def is_window_on_left_monitor(self, window: gw.Win32Window) -> bool:
        """
        Check if the window is primarily on the left monitor.
        Returns True if the window is on the left monitor, False otherwise.
        """
        # Calculate window center
        window_center_x = window.left + (window.width // 2)

        # Check if window center is within left monitor bounds
        return (self.left_monitor.x <= window_center_x <
                (self.left_monitor.x + self.left_monitor.width))

    def move_window_to_left_monitor(self, window: gw.Win32Window) -> None:
        """
        Move the window to the left monitor with a small offset from top-left.
        """
        # Calculate new position (centered on left monitor)
        new_x = self.left_monitor.x + (self.left_monitor.width - window.width) // 2
        new_y = self.left_monitor.y + (self.left_monitor.height - window.height) // 2

        # Ensure position is not negative
        new_x = max(self.left_monitor.x, new_x)
        new_y = max(self.left_monitor.y, new_y)

        # Move the window
        try:
            window.moveTo(new_x, new_y)
            time.sleep(0.5)  # Allow time for the window to move
            logger.info(f"Moved window to left monitor: ({new_x}, {new_y})")
        except Exception as e:
            logger.error(f"Failed to move window: {e}")

    def highlight_box_in_window(self, box, duration=2) -> None:
        """
        Draw a red box for the specified duration.
        """
        if not self.window:
            logger.warning("No window to highlight")
            return

        try:
            # Get window position and size
            left, top, width, height = box
            left, top = left + self.window.left, top + self.window.top

            # Create a transparent window with a red border
            highlight = tk.Tk()
            highlight.attributes("-topmost", True)
            highlight.attributes("-alpha", 0.3)
            highlight.overrideredirect(True)

            # Position and size the highlight
            highlight.geometry(f"{width}x{height}+{left}+{top}")

            # Create a red border frame
            frame = tk.Frame(highlight, borderwidth=3, bg="red")
            frame.pack(fill=tk.BOTH, expand=True)

            # Show the highlight
            highlight.update()

            # Wait for specified duration
            time.sleep(duration)

            # Destroy the highlight window
            highlight.destroy()

        except Exception as e:
            logger.error(f"Error highlighting box: {e}")

    def highlight_window(self, duration=2) -> None:
        """
        Draw a red box around the game window for the specified duration.
        """
        if not self.window:
            logger.warning("No window to highlight")
            return

        try:
            # Get window position and size
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height

            # Create a transparent window with a red border
            highlight = tk.Tk()
            highlight.attributes("-topmost", True)
            highlight.attributes("-alpha", 0.3)
            highlight.overrideredirect(True)

            # Position and size the highlight
            highlight.geometry(f"{width}x{height}+{left}+{top}")

            # Create a red border frame
            frame = tk.Frame(highlight, borderwidth=3, bg="red")
            frame.pack(fill=tk.BOTH, expand=True)

            # Show the highlight
            highlight.update()

            logger.info(f"Highlighting window: '{self.window.title}' for {duration} seconds")

            # Wait for specified duration
            time.sleep(duration)

            # Destroy the highlight window
            highlight.destroy()

        except Exception as e:
            logger.error(f"Error highlighting window: {e}")

    def get_window_coordinates(self) -> Tuple[int, int]:
        """
        Get the X and Y coordinates of the game window.
        Returns (x, y) coordinates of the top-left corner.
        """
        if not self.window:
            self.refresh_window()

        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")

        return self.window.left, self.window.top

    def get_window_region(self) -> Tuple[int, int, int, int]:
        """
        Get the region (left, top, width, height) of the game window.
        Returns (left, top, width, height) defining the window region.
        """
        if not self.window:
            self.refresh_window()

        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")

        return self.window.left, self.window.top, self.window.width, self.window.height

    def capture_screen(self, region=None):
        """
        Capture the screen within the game window region.
        Returns a screenshot of the game window.
        """
        if region is None:
            region = self.get_window_region()

        return pyautogui.screenshot(region=region)

    def detect_current_screen(self) -> GameScreen:
        """
        Improved screen detection that uses multiple techniques:
        1. UI element presence detection
        2. Color histogram analysis
        3. Frame difference analysis for motion detection

        Returns the detected GameScreen enum.
        """
        # Capture current screen
        current_screenshot = self.capture_screen()
        current_screenshot_path = os.path.join(SCREENS_DIR, "current.png")
        current_screenshot.save(current_screenshot_path)

        # Convert to OpenCV format
        current_img = cv2.imread(current_screenshot_path)
        if current_img is None:
            logger.error("Failed to load current screenshot for screen detection")
            return GameScreen.UNKNOWN

        # First check: Are there significant UI elements present?
        detected_screen = self._detect_screen_by_ui_elements(current_img)
        if detected_screen != GameScreen.UNKNOWN:
            self.current_screen = detected_screen
            return detected_screen

        # Second check: Color histogram analysis
        detected_screen = self._detect_screen_by_color(current_img)
        if detected_screen != GameScreen.UNKNOWN:
            self.current_screen = detected_screen
            return detected_screen

        # If still unknown, check if we're in gameplay by analyzing movement
        if self.previous_screenshot is not None:
            # If there's significant movement in the game area, it's likely gameplay
            prev_img = np.array(self.previous_screenshot)
            if self._detect_gameplay_by_motion(prev_img, current_img):
                self.current_screen = GameScreen.GAMEPLAY
                return GameScreen.GAMEPLAY

        # Store current screenshot for next comparison
        self.previous_screenshot = current_screenshot

        # If we reach here, we couldn't confidently identify the screen
        logger.warning("Could not confidently identify current screen")
        return GameScreen.UNKNOWN

    def _detect_screen_by_ui_elements(self, img) -> GameScreen:
        """
        Detect screen type by looking for specific UI elements.
        Regions for detection are defined as constant boxes.
        """
        h, w = img.shape[:2]

        # Define constant regions as (x, y, width, height)
        MAIN_MENU_BOX = (int(w * 0.3), int(h * 0.8), int(w * 0.4), int(h * 0.2))
        GAME_MODE_HEADER = (int(w * 0.3), int(h * 0.35), int(w * 0.4), int(h * 0.05))
        GAME_MODE_BUTTON = (int(w * 0.2), int(h * 0.47), int(w * 0.6), int(h * 0.05))
        GAME_OVER_TEXT = (int(w * 0.2), int(h * 0.3), int(w * 0.6), int(h * 0.2))
        GAME_OVER_BUTTON = (int(w * 0.3), int(h * 0.6), int(w * 0.4), int(h * 0.1))
        LEVEL_UP_TEXT = (int(w * 0.3), int(h * 0.25), int(w * 0.4), int(h * 0.1))
        UPGRADE_PANEL = (0, int(h * 0.5), w, int(h * 0.3))
        GAMEPLAY_LEVEL = (int(w * 0.4), 0, int(w * 0.2), int(h * 0.1))
        GAMEPLAY_TIMER = (int(w * 0.45), 0, int(w * 0.1), int(h * 0.1))

        # Utility function to extract a region from img
        def region(img, box):
            x, y, box_w, box_h = box
            return img[y:y + box_h, x:x + box_w]

        # 1. MAIN_MENU: look for a large green PLAY button in the bottom area
        self.highlight_box_in_window(MAIN_MENU_BOX)
        main_menu_area = region(img, MAIN_MENU_BOX)
        green_mask = (
                (main_menu_area[:, :, 1] > 150) &
                (main_menu_area[:, :, 0] < 100) &
                (main_menu_area[:, :, 2] < 100)
        )
        if np.sum(green_mask) > 500:
            logger.info("Detected MAIN_MENU screen (green PLAY button)")
            return GameScreen.MAIN_MENU

        # 2. GAME_MODE: look for yellow header text in the middle-top area
        self.highlight_box_in_window(GAME_MODE_HEADER)
        header_area = region(img, GAME_MODE_HEADER)
        yellow_mask = (
                (header_area[:, :, 0] < 100) &
                (header_area[:, :, 1] > 150) &
                (header_area[:, :, 2] > 150)
        )
        if np.sum(yellow_mask) > 200:
            self.highlight_box_in_window(GAME_MODE_BUTTON)
            button_area = region(img, GAME_MODE_BUTTON)
            green_button_mask = (button_area[:, :, 1] > 150) & (button_area[:, :, 0] < 100)
            if np.sum(green_button_mask) > 200:
                logger.info("Detected GAME_MODE screen (yellow header with green/purple buttons)")
                return GameScreen.GAME_MODE

        # 3. GAME_OVER: look for large red "GAME OVER" text in the center
        self.highlight_box_in_window(GAME_OVER_TEXT)
        game_over_area = region(img, GAME_OVER_TEXT)
        red_mask = (
                (game_over_area[:, :, 2] > 200) &
                (game_over_area[:, :, 0] < 100) &
                (game_over_area[:, :, 1] < 100)
        )
        if np.sum(red_mask) > 1000:
            self.highlight_box_in_window(GAME_OVER_BUTTON)
            button_area = region(img, GAME_OVER_BUTTON)
            purple_mask = (
                    (button_area[:, :, 0] > 100) &
                    (button_area[:, :, 2] > 100) &
                    (button_area[:, :, 1] < 100)
            )
            if np.sum(purple_mask) > 100:
                logger.info("Detected GAME_OVER screen (red GAME OVER text)")
                return GameScreen.GAME_OVER

        # 4. LEVEL_UP: look for yellow "LEVEL UP!" text with upgrade options
        self.highlight_box_in_window(LEVEL_UP_TEXT)
        level_up_area = region(img, LEVEL_UP_TEXT)
        yellow_mask = (
                (level_up_area[:, :, 0] < 100) &
                (level_up_area[:, :, 1] > 150) &
                (level_up_area[:, :, 2] > 150)
        )
        if np.sum(yellow_mask) > 200:
            self.highlight_box_in_window(UPGRADE_PANEL)
            upgrade_area = region(img, UPGRADE_PANEL)
            purple_mask = (
                    (upgrade_area[:, :, 0] > 80) &
                    (upgrade_area[:, :, 2] > 80) &
                    (upgrade_area[:, :, 1] < 80)
            )
            if np.sum(purple_mask) > 2000:
                logger.info("Detected LEVEL_UP screen (yellow LEVEL UP text with upgrade options)")
                return GameScreen.LEVEL_UP

        # 5. GAMEPLAY: check for level indicator and timer in the top bar
        self.highlight_box_in_window(GAMEPLAY_LEVEL)
        gameplay_level_area = region(img, GAMEPLAY_LEVEL)
        # The level indicator is expected to be white text within the top bar area
        # Adjust the region for level text based on GAMEPLAY_LEVEL's position
        level_text_x = int(w * 0.4) - GAMEPLAY_LEVEL[0]
        level_text_w = int(w * 0.2)
        level_text_area = gameplay_level_area[:, level_text_x: level_text_x + level_text_w]
        white_mask = (
                (level_text_area[:, :, 0] > 200) &
                (level_text_area[:, :, 1] > 200) &
                (level_text_area[:, :, 2] > 200)
        )
        if np.sum(white_mask) > 50:
            self.highlight_box_in_window(GAMEPLAY_TIMER)
            gameplay_timer_area = region(img, GAMEPLAY_TIMER)
            timer_mask = (
                    (gameplay_timer_area[:, :, 0] > 200) &
                    (gameplay_timer_area[:, :, 1] > 200) &
                    (gameplay_timer_area[:, :, 2] > 200)
            )
            # Ensure the center area of the screen is not overly bright
            center_area = img[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
            middle_area_bright = np.sum(center_area > 200) / center_area.size
            if np.sum(timer_mask) > 20 and middle_area_bright < 0.2:
                logger.info("Detected GAMEPLAY screen (level indicator and timer)")
                return GameScreen.GAMEPLAY

        logger.warning("Could not identify screen, marking as UNKNOWN")
        return GameScreen.UNKNOWN

    def _detect_screen_by_color(self, img) -> GameScreen:
        """Detect screen type by analyzing color distribution."""
        # Calculate color histogram
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # If we have color signatures for screens, compare with them
        # This would be populated after the first encounters with each screen type

        # For demonstration, check if predominant color is purple (gameplay background)
        b, g, r = cv2.split(img)
        if (np.mean(b) > np.mean(r) and np.mean(b) > np.mean(g) and
                np.mean(b) > 100 and np.mean(r) > 50):
            # Purple-ish background suggests gameplay
            return GameScreen.GAMEPLAY

        return GameScreen.UNKNOWN

    def _detect_gameplay_by_motion(self, prev_img, curr_img) -> bool:
        """
        Detect if we're in gameplay by checking for motion in the game area.
        This is useful since gameplay has moving monsters and player character.
        """
        if prev_img.shape != curr_img.shape:
            return False

        # Calculate absolute difference between frames
        diff = cv2.absdiff(prev_img, curr_img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # If there's significant movement, it's likely gameplay
        motion_pixels = np.sum(thresh > 0)
        motion_percentage = motion_pixels / (thresh.shape[0] * thresh.shape[1])

        logger.info(f"Motion percentage: {motion_percentage}")

        # Return true if motion detected exceeds threshold
        return motion_percentage > 0.01  # 1% of pixels changed

    def click_at_relative_position(self, x_ratio, y_ratio):
        """
        Click at a position relative to the game window size.
        x_ratio and y_ratio should be between 0 and 1.
        """
        if not self.window:
            self.refresh_window()

        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")

        # Ensure window size is valid
        if self.window.width <= 0 or self.window.height <= 0:
            logger.warning("Invalid window dimensions. Trying to refresh window information.")
            self.refresh_window()
            if self.window.width <= 0 or self.window.height <= 0:
                raise Exception("Window has invalid dimensions")

        # Calculate absolute position
        x = self.window.left + int(self.window.width * x_ratio)
        y = self.window.top + int(self.window.height * y_ratio)

        # Ensure the click is within the left monitor
        if (self.left_monitor.x <= x < self.left_monitor.x + self.left_monitor.width and
                self.left_monitor.y <= y < self.left_monitor.y + self.left_monitor.height):

            # Log the click position
            logger.info(
                f"Clicking at position: ({x}, {y}), relative to window at ({self.window.left}, {self.window.top})")

            # Focus the window first
            try:
                self.window.activate()
                time.sleep(0.2)  # Wait for window to gain focus
            except Exception as e:
                logger.warning(f"Failed to activate window: {e}")

            # Perform the click
            current_pos = pyautogui.position()
            logger.info(f"Moving mouse from {current_pos} to ({x}, {y})")

            # Use moveTo first to ensure we're at the right position
            pyautogui.moveTo(x, y)
            time.sleep(0.1)  # Small delay to ensure mouse is at position
            pyautogui.click(x, y)

            return True
        else:
            logger.warning(f"Click position ({x}, {y}) is outside the left monitor boundaries")
            # Adjust to be within monitor bounds
            adjusted_x = max(min(x, self.left_monitor.x + self.left_monitor.width - 1), self.left_monitor.x)
            adjusted_y = max(min(y, self.left_monitor.y + self.left_monitor.height - 1), self.left_monitor.y)
            logger.info(f"Adjusting click to stay within left monitor: ({adjusted_x}, {adjusted_y})")

            # Focus the window first
            try:
                self.window.activate()
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Failed to activate window: {e}")

            # Use the adjusted coordinates
            pyautogui.moveTo(adjusted_x, adjusted_y)
            time.sleep(0.1)
            pyautogui.click(adjusted_x, adjusted_y)
            return True

    # UI Interaction Methods
    def click_ui_element(self, element_name):
        """
        Click on a predefined UI element by name.
        Returns True if the element was clicked, False otherwise.
        """
        if element_name in UI_ELEMENTS:
            x_ratio, y_ratio = UI_ELEMENTS[element_name]
            return self.click_at_relative_position(x_ratio, y_ratio)
        else:
            logger.warning(f"Unknown UI element: {element_name}")
            return False

    def click_play_button(self):
        """Click the 'PLAY' button in the game window."""
        return self.click_ui_element("play_button")

    def click_shop_button(self):
        """Click the Shop button in the game UI."""
        return self.click_ui_element("shop_button")

    def click_normal_mode(self):
        """Click the Normal Mode button on the game mode selection screen."""
        return self.click_ui_element("normal_mode")

    def click_restart(self):
        """Click the Restart button on the game over screen."""
        return self.click_ui_element("restart_button")

    def select_upgrade(self, option=2):
        """
        Select an upgrade on the level up screen.
        Options: 1, 2, or 3 (left, middle, right)
        """
        option_element = f"level_up_option_{option}"
        result = self.click_ui_element(option_element)

        # Update game state
        if result:
            self.game_state["level"] += 1
            logger.info(f"Selected upgrade option {option}. New level: {self.game_state['level']}")

        return result

    def press_key(self, key, duration=0.05):
        """
        Press a key for the specified duration.
        Ensures the game window is active first.
        """
        if not self.window:
            self.refresh_window()

        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")

        try:
            # Activate the window first
            logger.info("Activating window...")
            self.window.activate()
            time.sleep(0.2)  # Wait for window to gain focus

            # Verify window is still valid (sometimes activate can fail silently)
            self.refresh_window()
            logger.info(f"Window activated: '{self.window.title}'")

            # Press the key
            logger.info(f"Pressing key: {key} for {duration}s")
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            return True
        except Exception as e:
            logger.error(f"Error pressing key {key}: {e}")
            return False

    def simulate_movement(self, direction, duration=0.05):
        """
        Simulate keyboard inputs for game movement.
        Direction must be one of 'left', 'right', 'up', 'down'.
        """
        valid_directions = {'left', 'right', 'up', 'down'}
        if direction not in valid_directions:
            logger.warning(f"Invalid direction: {direction}. Using 'left' instead.")
            direction = 'left'

        return self.press_key(direction, duration)

    # Reinforcement Learning Methods
    def get_game_state_representation(self):
        """
        Get a numerical representation of the game state for reinforcement learning.
        Returns a processed image or feature vector suitable for RL.
        """
        # Capture the current screen
        ss = self.capture_screen()

        # Convert to a format suitable for RL (grayscale, resized, normalized)
        img = np.array(ss)

        # Convert into grayscale to reduce dimensionality
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to a smaller dimension to reduce state space
        resized = cv2.resize(gray, (84, 84))  # Common size for RL

        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0

        return normalized

    def update_frame_buffer(self):
        """
        Update the frame buffer with the current state for temporal information.
        This is useful for RL algorithms that need sequence information.
        """
        current_state = self.get_game_state_representation()

        # Add current state to buffer and remove oldest if full
        self.frame_buffer.append(current_state)
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)

    def get_rl_state(self):
        """
        Get the current state representation for RL agent.
        Returns a stacked frame buffer for temporal information.
        """
        # Update buffer with latest frame
        self.update_frame_buffer()

        # Stack frames into a single array
        stacked_frames = np.stack(self.frame_buffer, axis=0)

        return stacked_frames

    def take_action(self, action_index):
        """
        Execute an action in the game environment based on the action index.
        Returns the new state, reward, and whether the episode is done.

        Action mapping:
        0: Do nothing
        1: Move up
        2: Move down
        3: Move left
        4: Move right
        """
        # Map action index to game control
        action_map = {
            0: None,  # No action
            1: "up",
            2: "down",
            3: "left",
            4: "right"
        }

        # Execute the selected action
        selected_action = action_map.get(action_index)
        if selected_action:
            self.simulate_movement(selected_action)

        # Small delay to let the game update
        time.sleep(0.1)

        # Capture new state
        new_state = self.get_rl_state()

        # Check if game state changed (e.g., level up, game over)
        self.detect_current_screen()

        # Calculate reward based on game state changes
        reward = self._calculate_reward()

        # Check if episode is done (game over)
        done = self.current_screen == GameScreen.GAME_OVER

        # Update game state
        self.game_state["reward"] = reward
        self.game_state["game_over"] = done

        return new_state, reward, done

    def _calculate_reward(self):
        """
        Calculate the reward based on game state changes.
        This is a simple implementation and should be expanded based on game mechanics.
        """
        reward = 0

        # Positive rewards
        if self.current_screen == GameScreen.LEVEL_UP:
            # Big reward for leveling up
            reward += 10

        # Small positive reward for surviving (encourages staying alive)
        if self.current_screen == GameScreen.GAMEPLAY:
            reward += 0.1

        # Large negative reward for game over
        if self.current_screen == GameScreen.GAME_OVER:
            reward -= 20

        return reward

    def reset_environment(self):
        """
        Reset the game environment to start a new episode.
        Returns the initial state.
        """
        # If game is over, click restart
        if self.current_screen == GameScreen.GAME_OVER:
            self.click_restart()
            time.sleep(1)

        # If at main menu, start a new game
        if self.current_screen == GameScreen.MAIN_MENU:
            self.click_play_button()
            time.sleep(1)

        # If at game mode selection, select normal mode
        if self.current_screen == GameScreen.GAME_MODE:
            self.click_normal_mode()
            time.sleep(1)

        # Reset game state
        self.game_state = {
            "level": 1,
            "health": 100,
            "coins": 0,
            "score": 0,
            "abilities": [],
            "time_alive": 0,
            "monsters_avoided": 0,
            "monsters_killed": 0,
            "game_over": False,
            "reward": 0
        }

        # Reset frame buffer
        self._prefill_frame_buffer()

        # Return initial state
        return self.get_rl_state()

    def play_game(self, max_runs=10):
        """
        Main game playing loop. Will play multiple runs of the game.
        Args:
            max_runs: Maximum number of game runs to play
        """
        run_count = 0

        while run_count < max_runs:
            # Detect current screen
            current_screen = self.detect_current_screen()

            if current_screen == GameScreen.UNKNOWN:
                logger.warning("Could not detect current screen. Taking screenshot for debug...")
                self.capture_screen().save(os.path.join(SCREENS_DIR, f"unknown_{int(time.time())}.png"))
                time.sleep(1)
                continue

            # Handle each screen type
            if current_screen == GameScreen.MAIN_MENU:
                logger.info("Main menu detected - clicking Play button")
                self.click_play_button()

            elif current_screen == GameScreen.GAME_MODE:
                logger.info("Game mode selection screen detected - selecting Normal Mode")
                self.click_normal_mode()

            elif current_screen == GameScreen.GAMEPLAY:
                logger.info("Gameplay screen detected - playing the game")
                self.play_game_loop()

            elif current_screen == GameScreen.LEVEL_UP:
                logger.info("Level up screen detected - selecting upgrade")
                self.select_upgrade()

            elif current_screen == GameScreen.GAME_OVER:
                logger.info("Game over screen detected - restarting game")
                self.click_restart()
                run_count += 1
                logger.info(f"Completed run {run_count}/{max_runs}")

            # Brief pause to allow screen transitions
            time.sleep(0.5)

    def play_game_loop(self, max_time=30):
        """
        Play the actual gameplay, controlling the character to avoid/eliminate monsters.
        """
        start_time = time.time()
        move_interval = 0.1  # Time between movement inputs
        last_move_time = 0

        logger.info(f"Starting gameplay loop for up to {max_time} seconds")

        while time.time() - start_time < max_time:
            # Check if we're still in gameplay screen
            current_time = time.time()

            # Make a move every interval
            if current_time - last_move_time > move_interval:
                # Choose a random direction to move
                self.random_movement()
                last_move_time = current_time

            # Brief pause to prevent excessive CPU usage
            time.sleep(0.01)

            # Every few seconds, check if the screen has changed
            if int(current_time) % 3 == 0:
                new_screen = self.detect_current_screen()
                if new_screen != GameScreen.GAMEPLAY:
                    logger.info(f"Screen changed during gameplay to {new_screen}")
                    return

    def random_movement(self):
        """Execute a random movement input."""
        directions = ["up", "down", "left", "right"]
        direction = random.choice(directions)
        self.simulate_movement(direction, duration=random.uniform(0.05, 0.2))

    def refresh_window(self) -> bool:
        """
        Refresh the game window handle in case it moved or was recreated.
        Returns True if window was found, False otherwise.
        """
        self.window = self.get_window_by_hwnd(self.hwnd)
        if self.window:
            logger.info(f"Found game window by handle: {self.hwnd}")

            # Check if window is on left monitor, if not, move it
            if not self.is_window_on_left_monitor(self.window):
                logger.info("Game window not on left monitor. Moving it...")
                self.move_window_to_left_monitor(self.window)

            return True

        logger.error(f"Game window with handle {self.hwnd} not found. Make sure the game is running.")
        return False


# Example usage
if __name__ == "__main__":
    try:
        print("Initializing Game Interface...")

        # List all windows with their handles for reference
        print("\nAvailable Windows (for reference):")
        for window in gw.getAllWindows():
            if window.visible and not window.isMinimized and window.width > 0 and window.height > 0:
                print(f"Handle: {window._hWnd}, Title: '{window.title}', Size: {window.width}x{window.height}")

        # Create game interface with the specified handle
        print(f"\nAttempting to connect to window with handle: {GAME_HWND}")
        game = GameInterface()

        # Display window information
        print(f"Successfully connected to: '{game.window.title}'")
        print(f"Window position: ({game.window.left}, {game.window.top})")
        print(f"Window size: {game.window.width}x{game.window.height}")

        # Highlight the game window
        print("Highlighting window...")
        game.highlight_window(duration=2)

        # Capture and save a screenshot
        print("Taking screenshot...")
        screenshot = game.capture_screen()
        screenshot_path = os.path.join(SCREENS_DIR, "initial_screenshot.png")
        screenshot.save(screenshot_path)
        print(f"Screenshot saved to {os.path.abspath(screenshot_path)}")

        # Detect current screen
        print("Detecting current screen...")
        current_screen = game.detect_current_screen()
        print(f"Current screen detected as: {current_screen.name if current_screen else 'Unknown'}")

        # Ask user what to do
        print("\nOptions:")
        print("1. Let the AI play the game using random actions")
        print("2. Train reinforcement learning agent")
        print("3. Play with trained agent")
        print("4. Run manual tests")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            print("\nStarting gameplay with random actions. Press Ctrl+C to stop...")
            game.play_game(max_runs=3)

        elif choice == '2':
            print("\nInitializing and training RL agent...")
            agent = GameLearningAgent(game)
            agent.train(num_episodes=50)

        elif choice == '3':
            print("\nPlaying with trained agent...")
            agent = GameLearningAgent(game)
            model_file = input("Enter model filename (leave empty for default): ")

            if not model_file:
                model_file = "model_episode_40.pkl"

            if agent.load_model(model_file):
                agent.play(num_episodes=3)
            else:
                print("Failed to load model. Using random play instead.")
                game.play_game(max_runs=3)

        elif choice == '4':
            # Manual testing section
            print("\nRunning manual tests...")

            # Test UI interactions with confirmation
            input("\nPress Enter to test clicking the 'PLAY' button...")
            game.click_play_button()
            print("Clicked the 'PLAY' button")

            # Detect current screen after click
            time.sleep(1)
            current_screen = game.detect_current_screen()
            print(f"Screen after clicking PLAY: {current_screen.name if current_screen else 'Unknown'}")
            game.click_normal_mode()
            print("Clicked the 'Normal Mode' button")


            # Test movement with confirmation
            input("\nPress Enter to test movement controls...")
            print("Testing movement: LEFT")
            game.simulate_movement('left')
            time.sleep(0.5)

            print("Testing movement: RIGHT")
            game.simulate_movement('right')
            time.sleep(0.5)

            print("Testing movement: UP")
            game.simulate_movement('up')
            time.sleep(0.5)

            print("Testing movement: DOWN")
            game.simulate_movement('down')

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in main: {e}", exc_info=True)
