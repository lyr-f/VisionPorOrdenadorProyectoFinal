import cv2
import os
import numpy as np

def nothing(x):
    """Dummy callback function for OpenCV trackbars."""
    pass

def get_hsv_color_range(image: np.array, csv_path: str, display_scale: float =0.5):
    """
    Interactive tool to select HSV color ranges from an image and save them to a CSV file.

    Opens a window displaying the input image with trackbars to adjust
    minimum and maximum HSV (Hue, Saturation, Value) values. The user can
    visualize the effect of HSV thresholds in real-time.

    Press:
    - 'g' to capture the current HSV range and append it to the CSV file.
    - 'q' to quit the tool and close the window.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.
    csv_path : str
        Path to the CSV file where HSV ranges will be saved.
        If the file does not exist, it will be created with a header.
    

    Notes
    -----
    - The window displays a real-time masked image showing which pixels fall
      within the selected HSV range.
    - The captured HSV ranges are saved as rows in the CSV in the format:
      HMin,SMin,VMin,HMax,SMax,VMax
    """

    # Create a window
    window_name = "Color range selector - press 'g' to capture de color range , 'q' to quit"
    cv2.namedWindow(window_name)

    # Create trackbars for color change
    cv2.createTrackbar('HMin', window_name, 0, 255, nothing)
    cv2.createTrackbar('SMin', window_name, 0, 255, nothing)
    cv2.createTrackbar('VMin', window_name, 0, 255, nothing)
    cv2.createTrackbar('HMax', window_name, 0, 255, nothing)
    cv2.createTrackbar('SMax', window_name, 0, 255, nothing)
    cv2.createTrackbar('VMax', window_name, 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', window_name, 255)
    cv2.setTrackbarPos('SMax', window_name, 255)
    cv2.setTrackbarPos('VMax', window_name, 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin',window_name)
        sMin = cv2.getTrackbarPos('SMin',window_name)
        vMin = cv2.getTrackbarPos('VMin',window_name)

        hMax = cv2.getTrackbarPos('HMax',window_name)
        sMax = cv2.getTrackbarPos('SMax',window_name)
        vMax = cv2.getTrackbarPos('VMax',window_name)

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image,image, mask= mask)

        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            # Print if there is a change in HSV value
            # print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        # Scale the image for display
        if display_scale != 1.0:
            display_output = cv2.resize(output, (0,0), fx=display_scale, fy=display_scale)
        else:
            display_output = output

        cv2.imshow(window_name, display_output)


        # Wait longer to prevent freeze for videos.
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('g'):
            with open(csv_path, "a") as f:
                line = f"{hMin},{sMin},{vMin},{hMax},{sMax},{vMax}\n"
                f.write(line)

            print("Saved in csv:")
            print(line.strip())
            break

        elif key == ord('q'):
            break


    cv2.destroyAllWindows()