"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-28
"""
import logging
import random
import re
import time

import pyautogui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='game_controller.log'
)
logger = logging.getLogger(__name__)


class GameController:
    # Allowed basic directions
    ALLOWED_DIRECTIONS = {'left', 'right', 'up', 'down'}
    # Mapping from WASD to arrow keys (or preferred direction strings)
    WASD_MAPPING = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}
    # All possible directions, including diagonal ones
    POSSIBLE_DIRECTIONS = [
        "up", "down", "left", "right",
        "up_left", "up_right", "down_left", "down_right"
    ]

    def __init__(self, game_interface):
        """
        Initialize the GameController with a reference to the GameInterface.
        :param game_interface: Instance of GameInterface.
        """
        self.game_interface = game_interface

    def _map_key(self, key: str) -> str:
        """
        Map a key if it's in the WASD mapping.
        """
        key = key.lower()
        return self.WASD_MAPPING.get(key, key)

    def press_key(self, key: str, duration: float = 0.05) -> bool:
        """
        Press a single key for the specified duration.
        """
        if not self.game_interface.window:
            if not self.game_interface.refresh_window():
                raise Exception(f"Game window with handle {self.game_interface.hwnd} not found")

        mapped_key = self._map_key(key)
        try:
            logger.info("Activating window...")
            self.game_interface.window.activate()
            time.sleep(0.2)
            self.game_interface.refresh_window()
            logger.info(f"Window activated: '{self.game_interface.window.title}'")
            logger.info(f"Pressing key: {mapped_key} for {duration:.2f}s")
            pyautogui.keyDown(mapped_key)
            time.sleep(duration)
        except Exception as e:
            logger.error(f"Error pressing key {mapped_key}: {e}")
            return False
        finally:
            try:
                pyautogui.keyUp(mapped_key)
            except Exception as e:
                logger.error(f"Error releasing key {mapped_key}: {e}")
        return True

    def simulate_movement(self, direction: str, duration: float = 0.05) -> bool:
        """
        Simulate keyboard input for game movement.
        Accepts either a single direction or a combination of directions (diagonal movement).
        Allowed directions are "up", "down", "left", "right" (or their WASD equivalents) and
        their combinations (e.g., "up_left", "up-right", "up left").
        """
        # Normalize the direction string: replace hyphens/spaces with underscore
        normalized = re.sub(r"[-\s]+", "_", direction.lower())
        keys = normalized.split("_")
        # Map each key individually
        mapped_keys = [self._map_key(k) for k in keys if
                       k in self.ALLOWED_DIRECTIONS or self._map_key(k) in self.ALLOWED_DIRECTIONS]

        if not mapped_keys:
            logger.warning(f"Invalid direction: {direction}. Defaulting to 'left'.")
            mapped_keys = ['left']

        # If only one key, use the single key press.
        if len(mapped_keys) == 1:
            return self.press_key(mapped_keys[0], duration)
        else:
            # Simultaneously press all keys for diagonal movement.
            try:
                logger.info(f"Simultaneously pressing keys: {mapped_keys} for {duration:.2f}s")
                if not self.game_interface.window:
                    if not self.game_interface.refresh_window():
                        raise Exception(f"Game window with handle {self.game_interface.hwnd} not found")
                self.game_interface.window.activate()
                time.sleep(0.2)
                self.game_interface.refresh_window()
                for key in mapped_keys:
                    pyautogui.keyDown(key)
                time.sleep(duration)
            except Exception as e:
                logger.error(f"Error pressing keys {mapped_keys}: {e}")
                return False
            finally:
                for key in mapped_keys:
                    try:
                        pyautogui.keyUp(key)
                    except Exception as e:
                        logger.error(f"Error releasing key {key}: {e}")
            return True

    def random_movement(self) -> bool:
        """
        Execute a random movement input, including diagonal directions.
        """
        direction = random.choice(self.POSSIBLE_DIRECTIONS)
        random_duration = random.uniform(0.05, 0.5)
        logger.info(f"Random movement: {direction} for {random_duration:.2f}s")
        return self.simulate_movement(direction, random_duration)
