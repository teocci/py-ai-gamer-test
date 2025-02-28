#!/usr/bin/env python3
"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-27
Description: Interface for interacting with Monanimal Mayhem game on dual monitor setup.
             Ensures all mouse interactions target the correct window on the left monitor.
             Uses window handle exclusively to avoid issues with multiple windows with the same title.
"""
import logging
import os
import random
import time
import tkinter as tk
from enum import Enum
from typing import Tuple, Optional

import cv2
import pyautogui
import pygetwindow as gw
from screeninfo import get_monitors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='game_interface.log'
)
logger = logging.getLogger(__name__)

# Constant for the game window handle - can be updated at runtime
GAME_HWND = 53281724

# Ensure PyAutoGUI doesn't move the mouse too fast
pyautogui.PAUSE = 0.1  # Add small pause between PyAutoGUI commands

# Create screens directory if it doesn't exist
SCREENS_DIR = "screens"
os.makedirs(SCREENS_DIR, exist_ok=True)


# Game screen types
class GameScreen(Enum):
    MAIN_MENU = "main.png"
    GAME_MODE = "mode.png"
    GAMEPLAY = "gameplay.png"
    LEVEL_UP = "level-up.png"
    GAME_OVER = "game-over.png"
    PAUSE_MENU = "pause.png"


class GameInterface:
    """Class for interfacing with the Monanimal Mayhem game."""

    def __init__(self, hwnd=None):
        """Initialize the game interface with optional window handle."""
        self.hwnd = hwnd or GAME_HWND
        self.window = None
        self.left_monitor = None
        self.current_screen = None
        self.game_state = {
            "level": 1,
            "health": 100,
            "coins": 0,
            "score": 0,
            "abilities": []
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

        # Save the reference screenshots
        self.save_reference_screenshots()

    def _initialize_monitor_info(self):
        """Initialize information about monitors."""
        monitors = get_monitors()
        if not monitors:
            raise Exception("No monitors found.")

        # Find the leftmost monitor (smallest x-coordinate)
        self.left_monitor = min(monitors, key=lambda m: m.x)
        logger.info(f"Detected {len(monitors)} monitors. Using leftmost monitor: {self.left_monitor}")

    def save_reference_screenshots(self):
        """Save the reference screenshots provided to the screens directory."""
        from shutil import copyfile

        # Move provided screenshots to screens directory
        for screen in GameScreen:
            source_path = screen.value
            target_path = os.path.join(SCREENS_DIR, screen.value)

            # Check if the source file exists before copying
            if os.path.exists(source_path):
                try:
                    copyfile(source_path, target_path)
                    logger.info(f"Saved reference screenshot: {target_path}")
                except Exception as e:
                    logger.error(f"Failed to save reference screenshot {source_path}: {e}")

    def detect_current_screen(self):
        """
        Detect which screen the game is currently showing based on reference screenshots.
        Returns the GameScreen enum of the current screen.
        """
        # Capture current screen
        current_screenshot = self.capture_screen()
        current_screenshot_path = os.path.join(SCREENS_DIR, "current.png")
        current_screenshot.save(current_screenshot_path)

        # Convert to OpenCV format for template matching
        current_img = cv2.imread(current_screenshot_path)
        if current_img is None:
            logger.error("Failed to load current screenshot for screen detection")
            return None

        # Convert to grayscale for better matching
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_match_val = -1

        # Compare with each reference screenshot
        for screen in GameScreen:
            ref_path = os.path.join(SCREENS_DIR, screen.value)
            if not os.path.exists(ref_path):
                continue

            try:
                template = cv2.imread(ref_path, 0)  # Read as grayscale
                if template is None:
                    logger.warning(f"Failed to load reference screenshot: {ref_path}")
                    continue

                # Resize template if needed to match current screenshot dimensions
                if template.shape[0] > current_gray.shape[0] or template.shape[1] > current_gray.shape[1]:
                    template = cv2.resize(template, (current_gray.shape[1], current_gray.shape[0]))

                # Template matching
                result = cv2.matchTemplate(current_gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                logger.info(f"Match score for {screen.value}: {max_val}")

                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match = screen
            except Exception as e:
                logger.error(f"Error comparing with {ref_path}: {e}")

        # Threshold for considering a match valid
        if best_match_val > 0.7:
            logger.info(f"Detected screen: {best_match.value} (score: {best_match_val})")
            self.current_screen = best_match
            return best_match
        else:
            logger.warning(
                f"Could not confidently identify current screen (best match: {best_match.value if best_match else 'None'}, score: {best_match_val})")
            return None

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

    def click_play_button(self):
        """
        Click the 'PLAY' button in the game window.
        The PLAY button is the large green button at the bottom.
        """
        # Based on the screenshot, PLAY button is at bottom center
        return self.click_at_relative_position(0.5, 0.82)

    def click_shop_button(self):
        """Click the Shop button in the game UI."""
        return self.click_at_relative_position(0.5, 0.55)

    def click_achievs_button(self):
        """Click the Achievements button in the game UI."""
        return self.click_at_relative_position(0.33, 0.43)

    def click_ranking_button(self):
        """Click the Ranking button in the game UI."""
        return self.click_at_relative_position(0.67, 0.43)

    def click_collectibles_button(self):
        """Click the Collectibles button in the game UI."""
        return self.click_at_relative_position(0.5, 0.49)

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
            self.window.activate()
            time.sleep(0.2)  # Wait for window to gain focus

            # Verify window is still valid (sometimes activate can fail silently)
            self.refresh_window()

            # Press the key
            logger.info(f"Pressing key: {key} for {duration}s in window: '{self.window.title}'")
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
        # Game-specific interaction methods

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

            if current_screen is None:
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

    def click_normal_mode(self):
        """Click the Normal Mode button on the game mode selection screen."""
        # Based on the screenshot, Normal Mode button is in the middle upper area
        return self.click_at_relative_position(0.5, 0.47)

    def click_restart(self):
        """Click the Restart button on the game over screen."""
        # Based on the screenshot, Restart button is in the middle bottom area
        return self.click_at_relative_position(0.5, 0.83)

    def select_upgrade(self):
        """
        Select an upgrade on the level up screen.
        Strategy: Prioritize upgrades in this order - speed, damage, health, special abilities
        """
        # First, capture the level up options
        # Since we can't easily read text, we'll pick the middle option
        # (In a more advanced version, we could use OCR to read the upgrade options)

        # Click to select the middle option
        self.click_at_relative_position(0.5, 0.55)
        time.sleep(0.2)

        # Click again to confirm selection
        self.click_at_relative_position(0.5, 0.55)

        # Update game state
        self.game_state["level"] += 1
        logger.info(f"Selected upgrade. New level: {self.game_state['level']}")

    def play_game_loop(self, max_time=30):
        """
        Play the actual gameplay, controlling the character to avoid/eliminate monsters.

        Args:
            max_time: Maximum time in seconds to play before checking screen state again
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
                move_strategy = self.get_move_strategy()

                # Execute the selected movement
                if move_strategy == "random":
                    self.random_movement()
                elif move_strategy == "avoid":
                    # Would use computer vision to identify threats and move away
                    # For now, just use random movement as placeholder
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

    def get_move_strategy(self):
        """
        Determine the best movement strategy based on the current game state.
        Returns the strategy name as a string.
        """
        # In a real implementation, this would analyze the game state
        # For now, just return "random" as placeholder
        return "random"

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
        print(f"Current screen detected as: {current_screen.value if current_screen else 'Unknown'}")

        # Ask user if they want to run the game
        choice = input("\nDo you want to let the AI play the game? (y/n): ")
        if choice.lower() == 'y':
            print("\nStarting gameplay. Press Ctrl+C to stop...")
            game.play_game(max_runs=3)
        else:
            # Manual testing section
            print("\nRunning manual tests...")

            # Test UI interactions with confirmation
            input("\nPress Enter to test clicking the 'PLAY' button...")
            game.click_play_button()
            print("Clicked the 'PLAY' button")

            # Detect current screen after click
            time.sleep(1)
            current_screen = game.detect_current_screen()
            print(f"Screen after clicking PLAY: {current_screen.value if current_screen else 'Unknown'}")
            time.sleep(0.5)

            game.click_normal_mode()
            print("Clicked the 'Normal Mode' button")

            # Detect current screen after click
            time.sleep(1)
            current_screen = game.detect_current_screen()
            print(f"Screen after clicking PLAY: {current_screen.value if current_screen else 'Unknown'}")

            # Test movement with confirmation
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
