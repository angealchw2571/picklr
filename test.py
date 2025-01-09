import pyautogui
import cv2
import numpy as np


def find_and_click(target_image_path, confidence=0.8):
    """
    Finds an image on the screen and clicks on it if found.

    :param target_image_path: Path to the image to search for.
    :param confidence: Matching confidence level (0 to 1).
    """
    # Take a screenshot of the entire screen
    screen = pyautogui.screenshot()

    # Convert the screenshot to a numpy array for OpenCV
    screen_np = np.array(screen)

    # Convert to grayscale for better matching performance
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    # Load and process the target image
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target_image is None:
        raise FileNotFoundError(f"Could not find the image at: {target_image_path}")

    # Use template matching to find the target image
    result = cv2.matchTemplate(screen_gray, target_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= confidence:
        # Get the center of the matched area
        target_height, target_width = target_image.shape[:2]
        center_x = max_loc[0] + target_width // 2
        center_y = max_loc[1] + target_height // 2

        # Move the mouse to the location and click
        pyautogui.moveTo(center_x, center_y, duration=0.2)
        pyautogui.click()
        print(f"Clicked on the target at: ({center_x}, {center_y})")
    else:
        print("Target image not found on screen.")


# Example usage
find_and_click("target_image.png")
