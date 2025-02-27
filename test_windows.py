"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-27
"""
import time
import tkinter as tk

import pygetwindow as gw


def highlight_window(window):
    """
    Draw a red box around a window's position on the screen for 2 seconds.

    Args:
        window (gw.Win32Window): The window to highlight
    """
    try:
        # Skip if window is not visible
        if not window.visible or window.isMinimized:
            print("Window is not visible or minimized")
            return

        # Get window position and size
        left, top = window.left, window.top
        width, height = window.width, window.height

        # Skip windows with no size
        if width <= 0 or height <= 0:
            print("Window has invalid dimensions")
            return

        # Create a transparent window with a red border to highlight the target window
        highlight = tk.Tk()
        highlight.attributes("-topmost", True)  # Keep our highlight on top
        highlight.attributes("-alpha", 0.3)  # Make it semi-transparent
        highlight.overrideredirect(True)  # Remove window decorations

        # Position and size the highlight to match the target window
        highlight.geometry(f"{width}x{height}+{left}+{top}")

        # Create a red border frame inside our window
        frame = tk.Frame(highlight, borderwidth=3, bg="red")
        frame.pack(fill=tk.BOTH, expand=True)

        # Show the highlight
        highlight.update()

        print(f"Highlighting window: {window.title} (handle: {window._hWnd})")

        # Wait for 2 seconds as specified
        time.sleep(2)

        # Destroy the highlight window
        highlight.destroy()

    except Exception as e:
        print(f"Error highlighting window: {e}")


def test_windows():
    """
    Iterate through all open windows, draw a red box around each window's position on the screen,
    display it briefly, then remove it. Identify and return a window without a title.

    Returns:
        gw.Win32Window or None: The untitled window if found, None otherwise.
    """
    print("\nAvailable Windows (for reference):")

    windows = gw.getAllWindows()  # Get all windows (including those without titles)
    for window in windows:
        try:
            # Skip windows that are minimized or not visible
            if not window.visible or window.isMinimized:
                continue

            # Get window position and size
            left, top = window.left, window.top
            width, height = window.width, window.height

            # Skip windows with no size or off-screen windows
            if width <= 0 or height <= 0:
                continue

            highlight_window(window)

        except Exception as e:
            print(f"Error processing window {window}: {e}")
            continue


# Example usage
if __name__ == "__main__":
    try:
        # Try to find an untitled window first
        test_windows()

    except Exception as e:
        print(f"Error: {e}")
