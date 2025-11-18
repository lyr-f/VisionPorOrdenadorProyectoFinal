import cv2
import os
import time # Still needed for generating the unique filename

def main(camera_index=0, width=1280, height=720):
    # --- Output Configuration ---
    # Folder name where the images will be saved
    output_dir = "camera_captures"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Camera Initialization ---
    # Force to use DIRECTSHOW for faster initialization on Windows
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return
    
    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Optional check to confirm the actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened at: {actual_width}x{actual_height}")

    window_name = "Live Camera - press 'g' to CAPTURE, 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # resizable window

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("No frame received (the camera may have been disconnected).")
                break

            # Optional: flip horizontally (mirror)
            display_frame = cv2.flip(frame, 1) # Use a flipped copy for display

            # Display the frame
            cv2.imshow(window_name, display_frame)

            # Wait 1 ms for a key
            key = cv2.waitKey(1) & 0xFF
            
            # --- Key Check and Saving Logic ---
            
            # Exit with 'q' or ESC
            if key == ord('q') or key == 27:
                break
                
            # CAPTURE IMAGE when 'g' is pressed
            elif key == ord('g'):
                # Generate a unique filename using the current timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"frame_{timestamp}.jpg")
                
                # Save the unflipped frame (optional: save the flipped display_frame if preferred)
                success = cv2.imwrite(filename, display_frame) 
                
                if success:
                    print(f"*** CAPTURED: {filename} ***")
                else:
                    print(f"Error saving: {filename}")
                
    except KeyboardInterrupt:
        # Allows for safe exit using Ctrl+C
        pass
    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure camera_index is correct for your USB camera (0, 1, 2, etc.)
    main(camera_index=0, width=1280, height=720)