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

import pyautogui
import pygetwindow as gw
from screeninfo import get_monitors

from agent import GameLearningAgent
from color_utils import *
from controller import GameController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='interface.log'
)
logger = logging.getLogger(__name__)


GAME_HWND = 14623178
pyautogui.PAUSE = 0.1
SCREENS_DIR = "screens"
os.makedirs(SCREENS_DIR, exist_ok=True)

# Game UI element locations (based on ratios relative to window size)
UI_ELEMENTS = {
    "shop_button": (0.5, 0.55),
    "play_button": (0.5, 0.82),
    "normal_mode": (0.5, 0.49),
    "endless_mode": (0.5, 0.55),
    "cancel_mode": (0.5, 0.66),
    "go_restart": (0.5, 0.66),
    "go_quit": (0.5, 0.73),
    "goc_restart": (0.5, 0.74),
    "goc_quit": (0.5, 0.80),
    "pause_restart": (0.5, 0.52),
    "pause_quit": (0.5, 0.58),
    "pause_continue": (0.5, 0.68),
    "level_up_left": (0.17, 0.55),
    "level_up_middle": (0.5, 0.55),
    "level_up_right": (0.83, 0.55),
    "game_pause": (0.92, 0.17),
}

# Game screen types
class GameScreen(Enum):
    MAIN_MENU = "main_menu"
    GAME_MODE = "game_mode"
    GAME_OVER = "game_over"
    PAUSE_MENU = "pause_menu"
    LEVEL_UP = "level_up"
    GAMEPLAY = "gameplay"
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
        self.controller = None
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
        self.window = self.window_by_hwnd(self.hwnd)
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

        # Initialize the game controller (from controller.py)
        self.controller = GameController(self)

    def _initialize_monitor_info(self):
        """Initialize information about monitors."""
        monitors = get_monitors()
        if not monitors:
            raise Exception("No monitors found.")
        self.left_monitor = min(monitors, key=lambda m: m.x)
        logger.info(f"Detected {len(monitors)} monitors. Using leftmost monitor: {self.left_monitor}")

    def _initialize_screen_detection(self):
        """Initialize reference features for screen detection based on UI elements."""
        self.screen_features = {
            GameScreen.MAIN_MENU: {
                "ui_elements": ["shop_button", "play_button"],
                "color_signature": None,
            },
            GameScreen.GAME_MODE: {
                "ui_elements": ["normal_mode", "endless_mode", "cancel_mode"],
                "color_signature": None,
            },
            GameScreen.GAME_OVER: {
                "ui_elements": ["go_restart", "go_quit", "goc_restart", "goc_quit"],
                "color_signature": None,
            },
            GameScreen.PAUSE_MENU: {
                "ui_elements": ["pause_restart", "pause_quit", "pause_continue"],
                "color_signature": None,
            },
            GameScreen.LEVEL_UP: {
                "ui_elements": ["level_up_left", "level_up_middle", "level_up_right"],
                "color_signature": None,
            },
            GameScreen.GAMEPLAY: {
                "ui_elements": ["game_pause"],
                "color_signature": None,
            },
        }

    def _prefill_frame_buffer(self):
        """Initialize frame buffer with copies of the current state for RL."""
        initial_frame = self.get_game_state_representation()
        for _ in range(self.frame_buffer_size):
            self.frame_buffer.append(initial_frame.copy())

    def window_by_hwnd(self, hwnd) -> Optional[gw.Win32Window]:
        """
        Get a window by its window handle (_hWnd).
        Returns the window if found, None otherwise.
        """
        for w in gw.getAllWindows():
            if w._hWnd == hwnd:
                return w
        logger.warning(f"No window found with handle {hwnd}")
        return None

    def bring_window_to_front_by_hwnd(self, hwnd) -> bool:
        """Bring the window with the given handle to the front."""
        w = self.window_by_hwnd(hwnd)
        return self.bring_window_to_front(w)

    def bring_window_to_front(self, w) -> bool:
        """Bring the window to the front."""
        if not w:
            logger.warning("No window to bring to front")
            return False
        if w.isMinimized:
            w.restore()
            time.sleep(0.2)
        w.activate()
        time.sleep(0.2)
        return True

    def bring_to_front(self) -> bool:
        """Bring the game window to the front."""
        if not self.window:
            self.refresh_window()
        if not self.window:
            logger.warning("No window to bring to front")
            return False
        try:
            return self.bring_window_to_front(self.window)
        except Exception as e:
            logger.error(f"Failed to bring window to front: {e}")
            return False

    def is_window_on_left_monitor(self, w: gw.Win32Window) -> bool:
        """
        Check if the window is primarily on the left monitor.
        """
        window_center_x = w.left + (w.width // 2)
        return (self.left_monitor.x <= window_center_x <
                (self.left_monitor.x + self.left_monitor.width))

    def move_window_to_left_monitor(self, w: gw.Win32Window) -> None:
        """
        Move the window to the left monitor with a small offset from top-left.
        """
        new_x = self.left_monitor.x + (self.left_monitor.width - w.width) // 2
        new_y = self.left_monitor.y + (self.left_monitor.height - w.height) // 2
        new_x = max(self.left_monitor.x, new_x)
        new_y = max(self.left_monitor.y, new_y)
        try:
            w.moveTo(new_x, new_y)
            time.sleep(0.5)
            logger.info(f"Moved window to left monitor: ({new_x}, {new_y})")
        except Exception as e:
            logger.error(f"Failed to move window: {e}")

    def highlight_box_in_window(self, box, duration=2) -> None:
        """Draw a red box for the specified duration."""
        if not self.window:
            logger.warning("No window to highlight")
            return
        try:
            left, top, width, height = box
            left, top = left + self.window.left, top + self.window.top
            highlight = tk.Tk()
            highlight.attributes("-topmost", True)
            highlight.attributes("-alpha", 0.3)
            highlight.overrideredirect(True)
            highlight.geometry(f"{width}x{height}+{left}+{top}")
            frame = tk.Frame(highlight, borderwidth=3, bg="red")
            frame.pack(fill=tk.BOTH, expand=True)
            highlight.update()
            time.sleep(duration)
            highlight.destroy()
        except Exception as e:
            logger.error(f"Error highlighting box: {e}")

    def highlight_window(self, duration=2) -> None:
        """Draw a red box around the game window for the specified duration."""
        if not self.window:
            logger.warning("No window to highlight")
            return
        try:
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height
            highlight = tk.Tk()
            highlight.attributes("-topmost", True)
            highlight.attributes("-alpha", 0.3)
            highlight.overrideredirect(True)
            highlight.geometry(f"{width}x{height}+{left}+{top}")
            frame = tk.Frame(highlight, borderwidth=3, bg="red")
            frame.pack(fill=tk.BOTH, expand=True)
            highlight.update()
            logger.info(f"Highlighting window: '{self.window.title}' for {duration} seconds")
            time.sleep(duration)
            highlight.destroy()
        except Exception as e:
            logger.error(f"Error highlighting window: {e}")

    def get_window_coordinates(self) -> Tuple[int, int]:
        """
        Get the X and Y coordinates of the game window.
        """
        if not self.window:
            self.refresh_window()
        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")
        return self.window.left, self.window.top

    def get_window_region(self) -> Tuple[int, int, int, int]:
        """
        Get the region (left, top, width, height) of the game window.
        """
        if not self.window:
            self.refresh_window()
        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")
        return self.window.left, self.window.top, self.window.width, self.window.height

    def capture_screen(self, region=None):
        """
        Capture the screen within the game window region.
        """
        if region is None:
            region = self.get_window_region()
        return pyautogui.screenshot(region=region)

    def detect_current_screen(self) -> GameScreen:
        """
        Detect current screen using UI element detection, color histogram analysis,
        and motion detection.
        """
        current_screenshot = self.capture_screen()
        current_screenshot_path = os.path.join(SCREENS_DIR, "current.png")
        current_screenshot.save(current_screenshot_path)
        current_img = cv2.imread(current_screenshot_path)
        if current_img is None:
            logger.error("Failed to load current screenshot for screen detection")
            return GameScreen.UNKNOWN

        detected_screen = self._detect_screen_by_ui_elements(current_img)
        if detected_screen != GameScreen.UNKNOWN:
            self.current_screen = detected_screen
            return detected_screen

        detected_screen = self._detect_screen_by_color(current_img)
        if detected_screen != GameScreen.UNKNOWN:
            self.current_screen = detected_screen
            return detected_screen

        if self.previous_screenshot is not None:
            prev_img = np.array(self.previous_screenshot)
            if self._detect_gameplay_by_motion(prev_img, current_img):
                self.current_screen = GameScreen.GAMEPLAY
                return GameScreen.GAMEPLAY

        self.previous_screenshot = current_screenshot
        logger.warning("Could not confidently identify current screen")
        return GameScreen.UNKNOWN

    def _detect_screen_by_ui_elements(self, img) -> GameScreen:
        """
        Detect screen type by looking for specific UI elements.
        """
        h, w = img.shape[:2]
        MAIN_MENU_BOX = (int(w * 0.3), int(h * 0.8), int(w * 0.4), int(h * 0.2))
        GAME_MODE_HEADER = (int(w * 0.25), int(h * 0.35), int(w * 0.5), int(h * 0.05))
        GAME_NORMAL_MODE_BUTTON = (int(w * 0.25), int(h * 0.47), int(w * 0.5), int(h * 0.05))
        GO_GAME_OVER_TEXT = (int(w * 0.07), int(h * 0.31), int(w * 0.86), int(h * 0.09))
        GOC_GAME_OVER_TEXT = (int(w * 0.07), int(h * 0.25), int(w * 0.86), int(h * 0.09))
        GO_RESTART_BUTTON = (int(w * 0.32), int(h * 0.64), int(w * 0.36), int(h * 0.05))
        GOC_RESTART_BUTTON = (int(w * 0.32), int(h * 0.71), int(w * 0.36), int(h * 0.05))
        LEVEL_UP_TEXT = (int(w * 0.35), int(h * 0.31), int(w * 0.29), int(h * 0.02))
        UPGRADE_PANEL = (int(w * 0.04), int(h * 0.46), int(w * 0.92), int(h * 0.23))
        GAMEPLAY_LEVEL = (int(w * 0.05), int(h * 0.11), int(w * 0.9), int(h * 0.03))
        GAMEPLAY_TIMER = (int(w * 0.43), int(h * 0.16), int(w * 0.13), int(h * 0.02))
        PAUSE_HEADER = (int(w * 0.25), int(h * 0.33), int(w * 0.49), int(h * 0.05))
        CONTINUE_BUTTON = (int(w * 0.34), int(h * 0.65), int(w * 0.32), int(h * 0.05))

        def region(img, box):
            x, y, box_w, box_h = box
            return img[y:y + box_h, x:x + box_w]

        main_menu_area = region(img, MAIN_MENU_BOX)
        if is_green_area(main_menu_area, pixel_threshold=500):
            logger.info("Detected MAIN_MENU screen (green PLAY button)")
            return GameScreen.MAIN_MENU

        header_area = region(img, GAME_MODE_HEADER)
        if is_yellow_area(header_area, pixel_threshold=200):
            button_area = region(img, GAME_NORMAL_MODE_BUTTON)
            if is_green_area(button_area, pixel_threshold=200):
                logger.info("Detected GAME_MODE screen (yellow header with green/purple buttons)")
                return GameScreen.GAME_MODE

        game_over_area = region(img, GO_GAME_OVER_TEXT)
        if is_red_area(game_over_area, pixel_threshold=500):
            button_area = region(img, GO_RESTART_BUTTON)
            if is_purple_area(button_area, pixel_threshold=100):
                logger.info("Detected GAME_OVER screen (red GAME OVER text, no collectibles)")
                return GameScreen.GAME_OVER

        game_over_area = region(img, GOC_GAME_OVER_TEXT)
        if is_red_area(game_over_area, pixel_threshold=500):
            button_area = region(img, GOC_RESTART_BUTTON)
            if is_purple_area(button_area, pixel_threshold=100):
                logger.info("Detected GAME_OVER screen (red GAME OVER text, found collectibles)")
                return GameScreen.GAME_OVER

        level_up_area = region(img, LEVEL_UP_TEXT)
        if is_yellow_area(level_up_area, pixel_threshold=200):
            upgrade_area = region(img, UPGRADE_PANEL)
            if is_purple_area_hsv(upgrade_area, pixel_threshold=2000):
                logger.info("Detected LEVEL_UP screen (yellow LEVEL UP text with upgrade options)")
                return GameScreen.LEVEL_UP

        gameplay_level_area = region(img, GAMEPLAY_LEVEL)
        if is_dark_area(gameplay_level_area, color=(27, 29, 51), pixel_threshold=2000):
            gameplay_timer_area = region(img, GAMEPLAY_TIMER)
            if is_white_area(gameplay_timer_area, pixel_threshold=20):
                return GameScreen.GAMEPLAY

        header_area = region(img, PAUSE_HEADER)
        if is_yellow_area(header_area, pixel_threshold=200):
            button_area = region(img, CONTINUE_BUTTON)
            if is_red_area(button_area, pixel_threshold=500):
                logger.info("Detected PAUSE_MENU screen (yellow header with red button)")
                return GameScreen.PAUSE_MENU

        logger.warning("Could not identify screen, marking as UNKNOWN")
        return GameScreen.UNKNOWN

    def _detect_screen_by_color(self, img) -> GameScreen:
        """Detect screen type by analyzing color distribution."""
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        b, g, r = cv2.split(img)
        if (np.mean(b) > np.mean(r) and np.mean(b) > np.mean(g) and
                np.mean(b) > 100 and np.mean(r) > 50):
            return GameScreen.GAMEPLAY
        return GameScreen.UNKNOWN

    def _detect_gameplay_by_motion(self, prev_img, curr_img) -> bool:
        """
        Detect if we're in gameplay by checking for motion in the game area.
        """
        if prev_img.shape != curr_img.shape:
            return False
        diff = cv2.absdiff(prev_img, curr_img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh > 0)
        motion_percentage = motion_pixels / (thresh.shape[0] * thresh.shape[1])
        logger.info(f"Motion percentage: {motion_percentage}")
        return motion_percentage > 0.01

    def click_at_relative_position(self, x_ratio, y_ratio) -> bool:
        """
        Click at a position relative to the game window size.
        """
        if not self.window:
            self.refresh_window()
        if not self.window:
            raise Exception(f"Game window with handle {self.hwnd} not found")
        if self.window.width <= 0 or self.window.height <= 0:
            logger.warning("Invalid window dimensions. Trying to refresh window information.")
            self.refresh_window()
            if self.window.width <= 0 or self.window.height <= 0:
                raise Exception("Window has invalid dimensions")
        x = self.window.left + int(self.window.width * x_ratio)
        y = self.window.top + int(self.window.height * y_ratio)
        if (self.left_monitor.x <= x < self.left_monitor.x + self.left_monitor.width and
                self.left_monitor.y <= y < self.left_monitor.y + self.left_monitor.height):
            logger.info(
                f"Clicking at position: ({x}, {y}), relative to window at ({self.window.left}, {self.window.top})")
            try:
                self.window.activate()
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Failed to activate window: {e}")
            current_pos = pyautogui.position()
            logger.info(f"Moving mouse from {current_pos} to ({x}, {y})")
            pyautogui.moveTo(x, y)
            time.sleep(0.1)
            pyautogui.click(x, y)
            return True
        else:
            logger.warning(f"Click position ({x}, {y}) is outside the left monitor boundaries")
            adjusted_x = max(min(x, self.left_monitor.x + self.left_monitor.width - 1), self.left_monitor.x)
            adjusted_y = max(min(y, self.left_monitor.y + self.left_monitor.height - 1), self.left_monitor.y)
            logger.info(f"Adjusting click to stay within left monitor: ({adjusted_x}, {adjusted_y})")
            try:
                self.window.activate()
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Failed to activate window: {e}")
            pyautogui.moveTo(adjusted_x, adjusted_y)
            time.sleep(0.1)
            pyautogui.click(adjusted_x, adjusted_y)
            return True

    def click_ui_element(self, element_name) -> bool:
        """
        Click on a predefined UI element by name.
        """
        if element_name in UI_ELEMENTS:
            x_ratio, y_ratio = UI_ELEMENTS[element_name]
            return self.click_at_relative_position(x_ratio, y_ratio)
        else:
            logger.warning(f"Unknown UI element: {element_name}")
            return False

    def click_shop_button(self):
        """Click the Shop button in the game UI."""
        return self.click_ui_element("shop_button")

    def click_play_button(self):
        """Click the 'PLAY' button in the game window."""
        return self.click_ui_element("play_button")

    def click_normal_mode(self):
        """Click the Normal Mode button on the game mode selection screen."""
        return self.click_ui_element("normal_mode")

    def click_endless_mode(self):
        """Click the Endless Mode button on the game mode selection screen."""
        return self.click_ui_element("endless_mode")

    def click_cancel_mode(self):
        """Click the Cancel button on the game mode selection screen."""
        return self.click_ui_element("cancel_mode")

    def click_game_over_restart(self):
        """Click the Restart button on the pause screen."""
        return self.click_ui_element("go_restart")

    def click_go_restart(self):
        """Click the Restart button on the game over screen."""
        return self.click_ui_element("go_restart")

    def click_go_quit(self):
        """Click the Quit button on the game over screen."""
        return self.click_ui_element("go_quit")

    def click_goc_restart(self):
        """Click the Restart button on the game over screen."""
        return self.click_ui_element("goc_restart")

    def click_goc_quit(self):
        """Click the Quit button on the game over screen."""
        return self.click_ui_element("goc_quit")

    def click_pause_restart(self):
        """Click the Restart button on the pause screen."""
        return self.click_ui_element("pause_restart")

    def click_quit(self):
        """Click the Quit button on the pause screen."""
        return self.click_ui_element("pause_quit")

    def click_continue(self):
        """Click the Continue button on the pause screen."""
        return self.click_ui_element("pause_continue")

    def select_upgrade(self, option='middle'):
        """
        Select an upgrade on the level up screen.
        Options: left, middle, or right.
        """
        option_element = f"level_up_{option}"
        if option_element in ["level_up_left", "level_up_right"]:
            activated = self.click_ui_element(option_element)
            if not activated:
                return False
            option_element = "level_up_middle"
        result = self.click_ui_element(option_element)
        if result:
            self.game_state["level"] += 1
            logger.info(f"Selected upgrade option {option}. New level: {self.game_state['level']}")
        return result

    def click_game_pause(self):
        """Click the Pause button during gameplay."""
        return self.click_ui_element("game_pause")

    def refresh_window(self) -> bool:
        """
        Refresh the game window handle in case it moved or was recreated.
        """
        self.window = self.window_by_hwnd(self.hwnd)
        if self.window:
            logger.info(f"Found game window by handle: {self.hwnd}")
            if not self.is_window_on_left_monitor(self.window):
                logger.info("Game window not on left monitor. Moving it...")
                self.move_window_to_left_monitor(self.window)
            return True
        logger.error(f"Game window with handle {self.hwnd} not found. Make sure the game is running.")
        return False

    def get_game_state_representation(self):
        """
        Get a numerical representation of the game state for reinforcement learning.
        """
        ss = self.capture_screen()
        img = np.array(ss)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        normalized = resized / 255.0
        return normalized

    def update_frame_buffer(self):
        """
        Update the frame buffer with the current state for temporal information.
        """
        current_state = self.get_game_state_representation()
        self.frame_buffer.append(current_state)
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)

    def get_rl_state(self):
        """
        Get the current state representation for the RL agent.
        """
        self.update_frame_buffer()
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        return stacked_frames

    def take_action(self, action_index):
        """
        Execute an action in the game environment based on the action index.
        Action mapping:
        0: Do nothing
        1: Move up
        2: Move down
        3: Move left
        4: Move right
        """
        action_map = {
            0: None,  # No action
            1: "up",
            2: "down",
            3: "left",
            4: "right"
        }
        selected_action = action_map.get(action_index)
        if selected_action:
            self.controller.simulate_movement(selected_action)
        time.sleep(0.1)
        new_state = self.get_rl_state()
        self.detect_current_screen()
        reward = self._calculate_reward()
        done = self.current_screen == GameScreen.GAME_OVER
        self.game_state["reward"] = reward
        self.game_state["game_over"] = done
        return new_state, reward, done

    def _calculate_reward(self):
        """
        Calculate the reward based on game state changes.
        """
        reward = 0
        if self.current_screen == GameScreen.LEVEL_UP:
            reward += 10
        if self.current_screen == GameScreen.GAMEPLAY:
            reward += 0.1
        if self.current_screen == GameScreen.GAME_OVER:
            reward -= 20
        return reward

    def reset_environment(self):
        """
        Reset the game environment to start a new episode.
        """
        if self.current_screen == GameScreen.GAME_OVER:
            self.click_game_over_restart()
            time.sleep(1)
        if self.current_screen == GameScreen.MAIN_MENU:
            self.click_play_button()
            time.sleep(1)
        if self.current_screen == GameScreen.GAME_MODE:
            self.click_normal_mode()
            time.sleep(1)
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
        self._prefill_frame_buffer()
        return self.get_rl_state()

    def play_game(self, max_runs=10):
        """
        Main game playing loop. Will play multiple runs of the game.
        """
        run_count = 0
        while run_count < max_runs:
            cs = self.detect_current_screen()
            if cs == GameScreen.UNKNOWN:
                logger.warning("Could not detect current screen. Taking screenshot for debug...")
                self.capture_screen().save(os.path.join(SCREENS_DIR, f"unknown_{int(time.time())}.png"))
                time.sleep(1)
                continue

            if cs == GameScreen.MAIN_MENU:
                logger.info("Main menu detected - clicking Play button")
                self.click_play_button()
            elif cs == GameScreen.GAME_MODE:
                logger.info("Game mode selection screen detected - selecting Normal Mode")
                self.click_normal_mode()
            elif cs == GameScreen.GAMEPLAY:
                logger.info("Gameplay screen detected - playing the game")
                self.play_game_loop()
            elif cs == GameScreen.LEVEL_UP:
                logger.info("Level up screen detected - selecting upgrade")
                self.select_upgrade()
            elif cs == GameScreen.GAME_OVER:
                logger.info("Game over screen detected - restarting game")
                self.click_game_over_restart()
                run_count += 1
                logger.info(f"Completed run {run_count}/{max_runs}")
            time.sleep(0.5)

    def play_game_loop(self, max_time=30):
        """
        Play the actual gameplay, controlling the character.
        """
        start_time = time.time()
        move_interval = 0.1
        last_move_time = 0
        logger.info(f"Starting gameplay loop for up to {max_time} seconds")
        while time.time() - start_time < max_time:
            current_time = time.time()
            if current_time - last_move_time > move_interval:
                self.controller.random_movement()
                last_move_time = current_time
            time.sleep(0.01)
            if int(current_time) % 3 == 0:
                new_screen = self.detect_current_screen()
                if new_screen != GameScreen.GAMEPLAY:
                    logger.info(f"Screen changed during gameplay to {new_screen}")
                    return

    def random_selection(self):
        """Select a random upgrade option."""
        options = ["left", "middle", "right"]
        option = random.choice(options)
        self.select_upgrade(option)

# Example usage
if __name__ == "__main__":
    try:
        print("Initializing Game Interface...")
        print("\nAvailable Windows (for reference):")
        for window in gw.getAllWindows():
            if window.visible and not window.isMinimized and window.width > 0 and window.height > 0:
                print(f"Handle: {window._hWnd}, Title: '{window.title}', Size: {window.width}x{window.height}")
        print(f"\nAttempting to connect to window with handle: {GAME_HWND}")
        game = GameInterface()
        game.bring_to_front()
        print(f"Successfully connected to: '{game.window.title}'")
        print(f"Window position: ({game.window.left}, {game.window.top})")
        print(f"Window size: {game.window.width}x{game.window.height}")
        print("Highlighting window...")
        game.highlight_window(duration=2)
        print("Taking screenshot...")
        screenshot = game.capture_screen()
        screenshot_path = os.path.join(SCREENS_DIR, "initial_screenshot.png")
        screenshot.save(screenshot_path)
        print(f"Screenshot saved to {os.path.abspath(screenshot_path)}")
        print("Detecting current screen...")
        current_screen = game.detect_current_screen()
        print(f"Current screen detected as: {current_screen.name if current_screen else 'Unknown'}")
        print("\nOptions:")
        print("1. Let the AI play the game using random actions")
        print("2. Train reinforcement learning agent")
        print("3. Play with trained agent")
        print("4. Run manual tests")
        print("5. Test UI Screen Detection")
        choice = input("Enter your choice (1-5): ")
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
            print("\nRunning manual tests...")
            input("\nPress Enter to test clicking the 'PLAY' button...")
            game.bring_to_front()
            game.click_play_button()
            print("Clicked the 'PLAY' button")
            time.sleep(1)
            current_screen = game.detect_current_screen()
            print(f"Screen after clicking PLAY: {current_screen.name if current_screen else 'Unknown'}")
            game.click_normal_mode()
            print("Clicked the 'Normal Mode' button")
            time.sleep(1)
            current_screen = game.detect_current_screen()
            print(f"Screen after clicking Normal Mode: {current_screen.value if current_screen else 'Unknown'}")
            print("Testing movement: LEFT")
            game.controller.simulate_movement('left')
            time.sleep(0.5)
            print("Testing movement: RIGHT")
            game.controller.simulate_movement('right')
            time.sleep(0.5)
            print("Testing movement: UP")
            game.controller.simulate_movement('up')
            time.sleep(0.5)
            print("Testing movement: DOWN")
            game.controller.simulate_movement('down')
        elif choice == '5':
            print("\nTesting UI screen detection...")
            moves = 0
            max_moves = 6
            while True:
                game.bring_to_front()
                current_screen = game.detect_current_screen()
                if current_screen == GameScreen.MAIN_MENU:
                    print("Detected screen: MAIN_MENU")
                    input("\nPress Enter to test clicking the 'PLAY' button...")
                    game.bring_to_front()
                    game.click_play_button()
                    print("Clicked the 'PLAY' button")
                elif current_screen == GameScreen.GAME_MODE:
                    print("Detected screen: GAME_MODE")
                    input("\nPress Enter to test clicking the 'Normal Mode' button...")
                    game.bring_to_front()
                    game.click_normal_mode()
                    print("Clicked the 'Normal Mode' button")
                elif current_screen == GameScreen.GAME_OVER:
                    print("Detected screen: GAME_OVER")
                    input("\nPress Enter to test clicking the 'Restart' button...")
                    game.bring_to_front()
                    game.click_game_over_restart()
                    print("Clicked the 'Restart' button")
                elif current_screen == GameScreen.LEVEL_UP:
                    print("Detected screen: LEVEL_UP")
                    game.random_selection()
                    print("Selected an upgrade")
                elif current_screen == GameScreen.GAMEPLAY:
                    game.bring_to_front()
                    print("Detected screen: GAMEPLAY")
                    game.controller.random_movement()
                    print("Performed a random movement")
                    game.click_game_pause()
                elif current_screen == GameScreen.PAUSE_MENU:
                    print("Detected screen: PAUSE_MENU")
                    input("\nPress Enter to test clicking the 'Continue' button...")
                    game.bring_to_front()
                    game.click_continue()
                    print("Clicked the 'Resume' button")
                else:
                    print("Unknown screen detected")
                moves += 1
                if moves < max_moves:
                    time.sleep(0.2)
                    continue
                time.sleep(1)
                restart = input("\nPress 'y' to restart the test or any other key to exit: ")
                if restart.lower() != 'y':
                    break
                moves = 0
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in main: {e}", exc_info=True)