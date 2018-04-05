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
from os.path import join, dirname, abspath, isdir, isfile
from os import makedirs, listdir
from shutil import rmtree
import numpy as np
import cv2
import h5py
from pdb import set_trace
"""-------------------------------------------------------------------------"""
"""----------------------------Argument parsing-----------------------------"""
"""-------------------------------------------------------------------------"""

parser = argparse.ArgumentParser(description='3D position estimation based on stereo vision')

parser.add_argument('--calibrate', action='store_true', default=False,
                    help='calibration process will be carried out based on a checkerboard pattern')
parser.add_argument('--run', action='store_true', default=True,
                    help='displays the undistorted stereo video stream from 2 webcams')
parser.add_argument('--force', action='store_true', default=False,
                    help='Overwrites existing files (e.g. calibration results)')
parser.add_argument('--calnum', type=int, default=10, metavar='N',
                    help='number of images used for calibration (default: 10)')

# Argument parsing
args = parser.parse_args()

"""-------------------------------------------------------------------------"""
"""Paths used for more functions"""
calibration_dir = join(dirname(abspath(__file__)), "calibration")   # root directory for calibration
if not isdir(calibration_dir):
    makedirs(calibration_dir)

parameter_path = join(calibration_dir, 'stereo_mapping_data.hdf5')

"""-------------------------------------------------------------------------"""
"""Capture objects for both webcameras"""
# Create capture object for a camera to calibrate
capture_object0 = cv2.VideoCapture(0)
capture_object1 = cv2.VideoCapture(1)


"""-------------------------------------------------------------------------"""
"""----------------------------Calibration process--------------------------"""
"""-------------------------------------------------------------------------"""
if args.calibrate:

    # zeroth check: whether calibration results exist?
    if isfile(parameter_path) and not args.force:
        print('File at ', parameter_path, ' already exists, stopping...')
        _ = input()
        exit(-1)

    calibration_needed = False # assume with this flag that we already have tha images

    """-------------------------------------------------------------------------"""
    """Filesystem setup for calibration"""
    camera_subdir0 = join(calibration_dir, "camera_0")      # subdirectory for the actual camera
    camera_subdir1 = join(calibration_dir, "camera_1")      # subdirectory for the actual camera

    # create dirs
    # if only one the folders exists, delete both
    if not isdir(camera_subdir0) or not isdir(camera_subdir1) or args.force:
        if isdir(camera_subdir0):
            rmtree(camera_subdir0)

        if isdir(camera_subdir1):
            rmtree(camera_subdir1)

        makedirs(camera_subdir0)
        makedirs(camera_subdir1)

        # set flag that we need to write to file new images
        # set flag that we need to write to file new images
        calibration_needed = True

    """-------------------------------------------------------------------------"""
    """Calibration setup"""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # define pattern size specified as inner points on the checkerboard
    # (where black squares "touch" each other)
    pattern_size = (8,6)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) * 0.018
    # 3d point in real world space
    real_point = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    real_point[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * 0.018 # scale to mm

    # List to store image points from all the images.
    image_points0 = [] # 2d points in image plane for camera0
    image_points1 = [] # 2d points in image plane for camera1
    object_points = [real_point] * args.calnum # 3d points in real world space


    # record new images if needed
    if calibration_needed:
        # loop while we do not have the specified number of images for calibration
        while len(image_points0) is not args.calnum:

            # Capture frame-by-frame
            ret0, frame0 = capture_object0.read()
            ret1, frame1 = capture_object1.read()

            # proceed only if capture was succesful
            if ret0 and ret1 is True:
                # covert image to BW
                img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret0, corners0 = cv2.findChessboardCorners(image=img0, patternSize=pattern_size, corners=None)
                ret1, corners1 = cv2.findChessboardCorners(image=img1, patternSize=pattern_size, corners=None)

                # proceed only if corners were found properly
                if ret0 and ret1 is True:
                    # write image for future use:
                    cv2.imwrite(join(camera_subdir0, 'image_' + str(len(image_points0))) + '.png', img0)
                    cv2.imwrite(join(camera_subdir1, 'image_' + str(len(image_points1))) + '.png', img1)

                    print('Calibration process: ' + str(len(image_points0) + 1) + ' / ' + str(args.calnum))

                    # find corner subpixels for better accuracy
                    corners_subpix0 = cv2.cornerSubPix(img0, corners0, (11, 11), (-1, -1), criteria)
                    corners_subpix1 = cv2.cornerSubPix(img1, corners1, (11, 11), (-1, -1), criteria)

                    # append the points for the proper lists
                    image_points0.append(corners_subpix0)
                    image_points1.append(corners_subpix1)

                    # Draw and display the corners
                    img0 = cv2.drawChessboardCorners(cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR), patternSize=pattern_size,
                                                    corners=corners_subpix0, patternWasFound=ret0)
                    img1 = cv2.drawChessboardCorners(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), patternSize=pattern_size,
                                                                    corners=corners_subpix1, patternWasFound=ret1)

                    # need for a key
                    cv2.waitKey(-1)

                # Display the resulting frame
                cv2.imshow('Calibration process for camera0 ', img0)
                cv2.imshow('Calibration process for camera1 ', img1)
                cv2.waitKey(25)

    else:
        # read images captured by the first camera
        for img_name in listdir(camera_subdir0):
            img0 = cv2.cvtColor(cv2.imread(join(camera_subdir0, img_name)), cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret0, corners0 = cv2.findChessboardCorners(image=img0, patternSize=pattern_size, corners=None)

            if ret0 is True:
                # find corner subpixels for better accuracy
                corners_subpix0 = cv2.cornerSubPix(img0, corners0, (11, 11), (-1, -1), criteria)

                # append corner points to list
                image_points0.append(corners_subpix0)

        # read images captured by the second camera
        for img_name in listdir(camera_subdir1):
            img1 = cv2.cvtColor(cv2.imread(join(camera_subdir1, img_name)), cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret1, corners1 = cv2.findChessboardCorners(image=img1, patternSize=pattern_size, corners=None)

            if ret1 is True:
                # find corner subpixels for better accuracy
                corners_subpix1 = cv2.cornerSubPix(img1, corners1, (11, 11), (-1, -1), criteria)

                # append corner points to list
                image_points1.append(corners_subpix1)




    """
    Camera calibration for individual cameras 
    (results used as an initial guess for cv2.stereoCalibrate for a more robust result)
    """
    ret0, camera_matrix0, dist_coeffs0, rotation_vec0, translation_vec0 = \
                cv2.calibrateCamera(object_points, image_points0, img0.shape[::-1], None, None)
    ret1, camera_matrix1, dist_coeffs1, rotation_vec1, translation_vec1 = \
                    cv2.calibrateCamera(object_points, image_points1, img1.shape[::-1], None, None)


    # set flag values
    flags = 0
    # flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST

    """Stereo calibration"""
    ret, M0, d0, M1, d1, R, T, E, F = cv2.stereoCalibrate(object_points, image_points0, image_points1,
                                      camera_matrix0, dist_coeffs0, camera_matrix1, dist_coeffs1,
                                      img0.shape[::-1], criteria=criteria, flags=flags)

    """Stereo rectification"""
    # Q holds the quintessence of the algorithm, the reprojection matrix
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(M0, d0, M1, d1, img0.shape[::-1], R, T)


    """Distortion map calculation"""
    mx0, my0 = cv2.initUndistortRectifyMap(M0, d0, R0, P0, img0.shape[::-1], cv2.CV_32FC1)
    mx1, my1 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img0.shape[::-1], cv2.CV_32FC1)

    """Save parameters to file"""
    # create file handle for the calibration results in hdf5 format
    with h5py.File(parameter_path, 'w') as f:
        # rectification maps for camera0
        f.create_dataset("mx0", data=mx0)
        f.create_dataset("my0", data=my0)

        # rectification maps for camera1
        f.create_dataset("mx1", data=mx1)
        f.create_dataset("my1", data=my1)

        # rerojection matrix
        f.create_dataset("Q", data=Q)


if args.run:

    # read in parameters
    # [()] is needed to read in the whole array if you don't do that,
    #  it doesn't read the whole data but instead gives you lazy access to sub-parts
    #  (very useful when the array is huge but you only need a small part of it).
    # https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
    with h5py.File(parameter_path, 'r') as f:
        # rectification maps for camera0
        mx0 = f['mx0'][()]
        mx1 = f['mx1'][()]

        # rectification maps for camera1
        my0 = f['my0'][()]
        my1 = f['my1'][()]

        # rerojection matrix
        Q = f['Q'][()]

    # create disparity matching object in advance
    stereoBM_object = cv2.StereoBM_create(numDisparities=128, blockSize=15)

    # start video acquisition loop
    while True:

        # read the camera streams
        img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        #Todo: filtering?/object localization?

        # remap
        img0_rm = cv2.remap(img0, mx0, my0, cv2.INTER_LINEAR)
        img1_rm = cv2.remap(img1, mx1, my1, cv2.INTER_LINEAR)

        """Calculate the disparity map"""
        disparity_map = stereoBM_object.compute(img0_rm, img1_rm)
        cv2.filterSpeckles(disparity_map, 0, 16, 32) #filter out noise

        # scale disparity map for displaying purposes only
        disparity_scaled = (disparity_map / 16.).astype(np.uint8) + abs(disparity_map.min())

        # show the remapped images and the scaled disparity map
        cv2.imshow('remapped0', img0_rm)
        cv2.imshow('remapped1', img1_rm)
        cv2.imshow('disp', disparity_scaled)

        """Image reprojection into 3D"""
        #Todo: it would be great if this transform could be only used for the object (ROI)
        img_in_3d = cv2.reprojectImageTo3D(disparity_map, Q)
        cv2.imshow('3d', img_in_3d)

        if cv2.waitKey(40) & 0xFF == ord('x'):
            break

    # When everything done, release the capture
    capture_object0.release()
    capture_object1.release()
    cv2.destroyAllWindows()
