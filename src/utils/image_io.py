import cv2
import numpy as np
from typing import List
import os

def load_images(filenames: List) -> List:
    """
    Load multiple images from disk.
    """
    return [cv2.imread(filename) for filename in filenames]

def show_image(img: np.array, img_name: str = "Image"):
    """
    Display an image in a window and wait for a key press.
    """
    # Show the image
    cv2.imshow(img_name, img)
    # Wait until any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_image(output_folder:str, img_name:str, img:np.array):
    """
    Save an image to disk using the specified output folder and image name.
    """
    img_path = os.path.join(output_folder, f"{img_name}.jpg")
    cv2.imwrite(img_path, img)