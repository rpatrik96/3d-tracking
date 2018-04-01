"""
Created on Sat Mar 31 11:13 2018

@author: Patrik Reizinger (W5PDBR)

@brief:
    This piece of code is intended to accomplish 3d position tracking
    (orienttation tracking is a future plan) based on the Python-wrapper
    for OpenCV, and, from a theoretical point of view, epipolar geometry
    (with 2 webcams).
"""

# Imports
import argparse
from os.path import join, dirname, abspath, isdir, splitext
from os import makedirs, listdir
import numpy as np
import cv2
from pdb import set_trace
"""-------------------------------------------------------------------------"""
"""----------------------------Argument parsing-----------------------------"""
"""-------------------------------------------------------------------------"""

parser = argparse.ArgumentParser(description='3D position estimation based on stereo vision')

parser.add_argument('--calibrate', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--camera', type=int, default=3, metavar='N',
                    help='camera number as last digit of S/N (default: 3)')
parser.add_argument('--calnum', type=int, default=2, metavar='N',
                    help='number of images used for calibration (default: 2)')

# Argument parsing
args = parser.parse_args()

if args.calibrate:

    """Filesystem setup"""
    calibration_dir = join(dirname(abspath(__file__)), "calibration")   # root directory for calibration
    camera_subdir = join(calibration_dir, "camera_" + str(args.camera))      # subdirectory for the actual camera

    # create dirs
    if not isdir(calibration_dir):
        makedirs(calibration_dir)

    if not isdir(camera_subdir):
        makedirs(camera_subdir)

    """Calibration setup"""

    # Create capture object for a camera to calibrate
    capture_object = cv2.VideoCapture(0)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # define pattern size specified as inner points on the checkerboard
    # (where black squares "touch" each other)
    pattern_size = (8,6)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) * 0.018
    # 3d point in real world space
    real_point = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    real_point[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * 0.018 # scale to mm

    # List to store image points from all the images.
    image_points = [] # 2d points in image plane.
    object_points = [real_point] * args.calnum # 3d points in real world space


    # loop while we do not have the specified number of images for calibration
    while len(image_points) is not args.calnum:


        # Capture frame-by-frame
        ret, frame = capture_object.read()

        # proceed only if capture was succesful
        if ret is True:

            # covert image to BW
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # wait for 'n' to proceed
            if cv2.waitKey(200) & 0xFF == ord('n'):


                #Todo: thresholding?

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(image=img, patternSize=pattern_size, corners=None)

                # proceed only if corners were found properly
                if ret is True:

                    # write image for future use:
                    cv2.imwrite(join(camera_subdir, 'image_' + str(len(image_points))) + '.png', img)

                    print('Calibration process: ' + str(len(image_points) + 1) + ' / ' + str(args.calnum))

                    corners_subpix = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                    image_points.append(corners_subpix)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, patternSize=pattern_size,
                                                    corners=corners_subpix, patternWasFound=ret)

            # Display the resulting frame
            cv2.imshow('Calibration process for camera ' + str(args.camera), img)

    # until now, we have sufficient number of points
    ret, camera_matrix, dist_coeffs, rotation_vec, translation_vec = \
                cv2.calibrateCamera(object_points, image_points, img.shape[::-1], None, None)

    for img_name in listdir(camera_subdir):
        img = cv2.imread(join(camera_subdir, img_name))
        # set_trace()
        height, width = img.shape[:2]

        # get new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,
                                          imageSize=(width, height), alpha=1, newImgSize=(width, height))

        # undistort
        img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # crop the image
        x, y, width, height = roi
        img_undistorted = img_undistorted[y : y + height, x : x + width]

        # save undistorted images
        cv2.imwrite(join(camera_subdir, splitext(img_name)[0] + 'undistorted' + splitext(img_name)[1]), img_undistorted)

        # save the parameters
        #Todo: why not the new camera matrix???
        np.savez(join(calibration_dir, 'calibration_parameters_camera_' + str(args.camera)),
                            camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                            rotation_vec=rotation_vec, translation_vec=translation_vec)

        print(camera_matrix, dist_coeffs, rotation_vec, translation_vec)

    # When everything done, release the capture
    capture_object.release()
    cv2.destroyAllWindows()
