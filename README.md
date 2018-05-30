# 3d-tracking

### Author: Patrik Reizinger

## General Description:
This project is set up to estimation 3D position and orientation based on 2 webcameras, based on the OpenCV wrapper for Python. 
The code contains parts for camera calibration (with the standard OpenCV chessboard pattern), rectification and currently disparity map
calculation. Other feature included is object tracking - implemented with morphology and HSL-based segmentation.

Todo list:
- [ ] Replace diparity based z-coordinate estimation with geometric calculations
