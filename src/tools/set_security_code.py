import cv2
import os
import glob
from src.utils.color_calibration import get_hsv_color_range
from src.utils.frame_extractor import frame_extractor
from src.utils.image_io import *

# =====================================================
# This script should be executed from the project root as:
# python -m src.tools.set_security_code
# =====================================================

def set_security_code():
    """
    Interactive setup tool to define the security code colors.

    This function guides the user through capturing images of the objects 
    used in the security code and selecting the appropriate HSV color ranges 
    for each object. The selected ranges are saved in a CSV file for later use 
    by the main application.

    Workflow
    --------
    1. Ensure the directory "data/security_code_colors/" exists or create it.
    2. Capture images using the camera if the directory is empty.
    3. Load all images from the directory.
    4. For each image, allow the user to select the HSV range interactively 
       using trackbars.
    5. Save all selected HSV ranges to "data/security_code_color_ranges.csv".

    Notes
    -----
    - Run this function only when you want to change the security code.
    - Each run overwrites the CSV file with new values.
    - If the directory "data/security_code_colors/" already contains images,
      running this function will NOT capture new images. Instead, it will only
      allow the user to reselect the HSV color ranges for the existing images.
    - To start the security code setup from scratch (including image capture),
      delete all images inside "data/security_code_colors/" before running
      this function.
    - The function uses the `frame_extractor` and `get_hsv_color_range`
      utilities for capturing and calibrating images.
    - The displayed images may be scaled depending on your monitor resolution
      to fit with the trackbars.

    - IMPORTANT: To ensure imports work correctly, execute this script from
      the project root using:
      
        python -m src.tools.set_security_code
    """
    
    print("Get the images of the diferent objects you want to use for your security code")
    directory_path = "data/security_code_colors/"
    os.makedirs(directory_path, exist_ok=True)
    imgs_path = glob.glob(directory_path+"*.jpg")
    while not imgs_path:
        frame_extractor(camera_index=0, width=1280, height=720, output_dir=directory_path)
        imgs_path = glob.glob(directory_path+"*.jpg")
    imgs = load_images(imgs_path)
    color_ranges_csv_path =  "data/security_code_color_ranges.csv"
    # Create the csv to store values or overwrite it
    with open(color_ranges_csv_path, "w") as f:
        f.write("HMin,SMin,VMin,HMax,SMax,VMax\n")
    # Select the range that best captures the color for each object
    for img in imgs:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        get_hsv_color_range(img, color_ranges_csv_path)

if __name__ == "__main__":
    set_security_code()