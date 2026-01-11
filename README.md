# VOI_ProyectoFinal: FitVision

Project developed by Ana Ling Gil González and Leyre Fontaneda Fernández

## Introduction

This project is conceived as an interactive exercise challenge platform, where users perform different physical activities in front of the camera and the system automatically detects and records their movements. The main objective of the project is to create a playful and competitive environment around physical activity, enabling applications such as automated personal training systems or exercise challenges that can be shared among users.

To ensure controlled access, a visual security system based on color recognition has been implemented. The user must present a predefined sequence of clothing colors to the camera, which acts as a visual password. Only after the correct sequence is detected does the system allow access to the exercise tracking functionalities.

## Camera Calibration

Camera calibration was performed to estimate the intrinsic parameters of the camera and correct lens distortions. The process used 20 images of a chessboard pattern captured from different viewpoints. OpenCV was employed to automatically detect the chessboard corners and compute the calibration parameters through 3D–2D correspondences.

## Security System

The security system controls access to the platform through the detection of a sequence of colors shown in front of the camera using clothing items. Each color corresponds to one element of the visual password, and repetitions are ignored to avoid duplicated detections. 

The system processes each video frame by converting it from BGR to HSV color space and applying color segmentation to generate binary masks for the predefined colors. Morphological operations are then applied to reduce noise and improve detection robustness. The number of detected pixels for each color is evaluated, and when a color exceeds a predefined threshold, it is considered valid.

Once the first color is detected, a 30-second timer is activated. If the full sequence is not correctly completed within this time, the process resets. When the detected sequence matches the expected pattern, access to the system is granted. To set the password, the user needs to capture images of the objects they want to use in the security code and select the appropriate HSV color ranges for each object.

## Proposed System: Tracker and Video Output

The system allows users to customize and perform a workout routine (squats, push-ups, and jumping jacks) detected in real time through a camera. Using OpenCV and MediaPipe Pose, body landmarks and joint angles are analyzed to recognize valid repetitions, update counters, and track progress toward predefined goals. The interface displays repetition status during execution, and once the routine is completed, a final completion message and visual feedback are shown.


## Results and Next Steps

The system successfully detects the color-based security pattern in real time and provides a solid foundation for integrating exercise tracking functionalities. Future improvements could include expanding the range of detectable exercises, increasing robustness under varying lighting conditions, and adding audio feedback to enhance the user experience.