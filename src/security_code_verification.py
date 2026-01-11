import cv2
import time
from utils.detect_color import detect_color

def security_code_verification(csv_path, camera_index=0, width=1280, height=720):
    """
    Verifies a security code based on a sequence of colors detected through a camera feed.

    The expected security code is defined in a CSV file containing HSV color ranges.
    This CSV file can be created or modified using the script:
    `tools/set_security_code.py`.

    Each color is associated with an index, and the correct code corresponds to showing
    the colors in ascending index order. The user must present the colors one by one
    in front of the camera within a limited time window.

    The function continuously captures frames from the specified camera, detects which
    color (if any) is present using `detect_color`, and with it builds the introduced 
    code. If the full sequence is detected in the correct order, the verification 
    succeeds.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing HSV color ranges. Each row (after the header)
        must define a color as: hMin, sMin, vMin, hMax, sMax, vMax.
    camera_index : int, optional (default=0)
        Index of the camera to use for video capture.
    width : int, optional (default=1280)
        Desired width of the camera capture (currently not enforced internally).
    height : int, optional (default=720)
        Desired height of the camera capture (currently not enforced internally).

    Returns:
    --------
    bool or None
        Returns True if the correct color sequence is successfully introduced.
        Returns False if the user quits manually.
        Returns None if the camera cannot be opened or an unexpected interruption occurs.

    Notes:
    ------
    - The user has 30 seconds to introduce the full code after the first color is detected.
      If the timeout is exceeded, the introduced code is reset.
    - Consecutive detections of the same color are ignored to avoid duplicates.
    - Pressing 'q' or ESC closes the window and ends the verification process.
    """
    # Read CSV file to obtain HSV ranges for each color in the security code
    # Each row represents one color using lower and upper HSV bounds
    colors = []
    with open(csv_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:   # Skip CSV header
                split_line = line.strip().split(",")
                hMin,sMin,vMin,hMax,sMax,vMax = [int(i) for i in split_line]
                # Store each color as a tuple: (lower_HSV, upper_HSV)
                colors.append(((hMin,sMin,vMin),(hMax,sMax,vMax)))

    code_lenght = len(colors)
    # The correct code is defined as the ordered sequence of color indices
    # Example: if there are 3 colors, the correct code is [0, 1, 2]    
    real_code = [i for i in range(code_lenght)]
    print(real_code)
    introduced_code = []
    time_limit = 30    # Reset code if the user exceeds the allowed time window


    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return
    
    window_name = "Security code verification - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # resizable window

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected).")
                break

            # Optional: flip horizontally (mirror), comment out if not desired
            frame = cv2.flip(frame, 1)

            # Display the frame
            cv2.imshow(window_name, frame)

            detected_color = detect_color(img=frame, colors=colors, threshold=20000)
            if detected_color is not None:
                # First detected color starts the code input and the timeout timer
                if not introduced_code:
                    starting_time = time.time()
                    introduced_code.append(detected_color)
                    print("A code color has been detected")
                    print(f"You have to show {code_lenght-len(introduced_code)} more code colors.")
                else:
                    # Only register a new color if it differs from the previous one
                    # This avoids multiple detections of the same color in consecutive frames
                    if detected_color != introduced_code[-1]:
                        introduced_code.append(detected_color)
                        print(f"Another code color has been detected")
                        print(f"You have to show {code_lenght-len(introduced_code)} more code colors.")
                    if len(introduced_code) == code_lenght:
                        if introduced_code==real_code:
                            return True
                        else:
                            introduced_code = []
                            print("The code introduced was incorrect. The code has been reset, you may try again.")
                timer = time.time()-starting_time
                # Reset code if the user exceeds the allowed time window
                if  timer > time_limit:
                    introduced_code = []
                    print("Code introduction timed out after 30 seconds. The code has been reset. Please begin again.")


            # Wait 1 ms for a key; exit with 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                return False
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # If the built-in webcam is not at index 0, change the first argument: main(1)
    security_code_verification(csv_path="data/security_code_color_ranges.csv" , camera_index=0, width=1280, height=720)
