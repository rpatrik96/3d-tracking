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
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pdb import set_trace

"""
Keycodes (presenter)

Rigtht arrow: 34
Left arrow: 33
Presentation play: 27 (ESC)
Display hide: 190 (0xBE)
"""
erode_se = 5
dilate_se = 7
close_se = 55
erode_iter = 2

"""Plot setup"""
matplotlib.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')


"""-------------------------------------------------------------------------"""
"""----------------------------Mask processing------------------------------"""
"""-------------------------------------------------------------------------"""
def mask_processing(mask, img):

    # erosion: let small particles vanish
    cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((erode_se, erode_se), np.uint8), dst=mask,
                     iterations=erode_iter)

    # closing: stabilize detection
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_se, close_se), np.uint8), dst=mask)

    # cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((dilate_se, dilate_se), np.uint8), dst=mask, iterations=2)

    ret, roi = cv2.threshold(cv2.GaussianBlur(cv2.bitwise_and(img, mask), (9, 9), 0), 48, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # get contours
    marker_x0, marker_y0 = 0, 0 #if not found
    ret_marker = False  #flag for found marker
    if ret:
        _, contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # check only if there is some contour to find
        if len(contours):
            cnt = contours[0]
            max_area = cv2.contourArea(cnt)
            for con in contours:
                tmp_area = cv2.contourArea(con)
                if tmp_area > max_area:
                    cnt = con
                    max_area = tmp_area

            if cnt.sum():
                ret_marker = True

                rect = cv2.minAreaRect(cnt)

                # # get points for rectangle plot
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(roi, [box], 0, 128, 2)
                # get center point
                (marker_y0, marker_x0), (_, _), _ = rect

    if marker_y0 is 0 and marker_x0 is 0 and ret_marker is True:
        print('Ooops')

    return mask, roi, marker_x0, marker_y0, ret_marker


"""-------------------------------------------------------------------------"""
"""----------------------------Argument parsing-----------------------------"""
"""-------------------------------------------------------------------------"""

parser = argparse.ArgumentParser(description='3D position estimation based on stereo vision')

parser.add_argument('--calibrate', action='store_true', default=False,
                    help='Calibration process will be carried out based on a checkerboard pattern')
parser.add_argument('--run', action='store_true', default=True,
                    help='Displays the undistorted stereo video stream from 2 webcams')
parser.add_argument('--force', action='store_true', default=False,
                    help='Overwrites existing files (e.g. calibration results)')
parser.add_argument('--display-markers', action='store_true', default=False,
                    help='Displays the marker centers')
parser.add_argument('--display-disparity', action='store_true', default=False,
                    help='Displays the disparity map')
parser.add_argument('--display-reprojection', action='store_true', default=False,
                    help='Displays the reprojected (3D) image')
parser.add_argument('--calnum', type=int, default=15, metavar='N',
                    help='Number of images used for calibration (default: 15)')



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
    """-------------------------------------------------------------------------"""
    camera_subdir0 = join(calibration_dir, "camera_0")      # subdirectory for the actual camera
    camera_subdir1 = join(calibration_dir, "camera_1")      # subdirectory for the actual camera

    # create dirs
    # if only one the folders exists, delete both
    if not isdir(camera_subdir0) or not isdir(camera_subdir1) or args.force \
                or len(listdir(camera_subdir0)) is not args.calnum \
                or len(listdir(camera_subdir1)) is not args.calnum:

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
    """-------------------------------------------------------------------------"""
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

        # image window names
        winname0 = 'Calibration process for camera 0'
        winname1 = 'Calibration process for camera 1'

        # indicates whether image is in a position to calibrate
        match_flag = False
        image_success = False

        # loop while we do not have the specified number of images for calibration
        while len(image_points0) is not args.calnum:

            # Capture frame-by-frame
            capture_object0.grab()
            capture_object1.grab()

            ret0, frame0 = capture_object0.retrieve()
            ret1, frame1 = capture_object1.retrieve()

            # proceed only if capture was succesful
            if ret0 and ret1 is True:
                # covert image to BW
                img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

                # proceed pnly if user moved image into position (and pressed one of the arrow buttons)
                if match_flag:
                    # Find the chess board corners
                    ret0, corners0 = cv2.findChessboardCorners(image=img0, patternSize=pattern_size, corners=None)
                    ret1, corners1 = cv2.findChessboardCorners(image=img1, patternSize=pattern_size, corners=None)

                    # proceed only if corners were found properly
                    if ret0 and ret1 is True:
                        # clear match_flag
                        match_flag = False

                        # set success flag for display delay
                        image_success = True

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



                # Display the resulting frame
                cv2.imshow(winname0, img0)
                cv2.imshow(winname1, img1)

                if image_success:
                    cv2.waitKey(500)
                    image_success = False

                # watch out for the key which switches on the search for chessboard corners
                if cv2.waitKey(40) & 0xFF == 46:
                    match_flag = (not match_flag)

        # close windows
        cv2.destroyWindow(winname0)
        cv2.destroyWindow(winname1)


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

    """-------------------------------------------------------------------------"""
    """
    Camera calibration for individual cameras 
    (results used as an initial guess for cv2.stereoCalibrate for a more robust result)
    """
    """-------------------------------------------------------------------------"""
    ret0, camera_matrix0, dist_coeffs0, rot_vec0, t_vec0 = cv2.calibrateCamera(object_points, image_points0, img0.shape[0:2][::-1], None, None)
    ret1, camera_matrix1, dist_coeffs1, rot_vec1, t_vec1 = cv2.calibrateCamera(object_points, image_points1, img1.shape[0:2][::-1], None, None)

    tot_errorL = 0
    for i in range(len(object_points)):
        imgpointsL2, _ = cv2.projectPoints(object_points[i], rot_vec0[i], t_vec0[i], camera_matrix0, dist_coeffs0)
        errorL = cv2.norm(image_points0[i], imgpointsL2, cv2.NORM_L2) / len(imgpointsL2)
        tot_errorL += errorL

    print("LEFT: Re-projection error: ", tot_errorL / len(object_points))

    tot_errorR = 0
    for i in range(len(object_points)):
        imgpointsR2, _ = cv2.projectPoints(object_points[i], rot_vec1[i], t_vec1[i], camera_matrix1, dist_coeffs1)
        errorR = cv2.norm(image_points1[i], imgpointsR2, cv2.NORM_L2) / len(imgpointsR2)
        tot_errorR += errorR

    print("RIGHT: Re-projection error: ", tot_errorR / len(object_points))
    
    # set flag values
    flags = 0
    # flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST

    flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +cv2.CALIB_USE_INTRINSIC_GUESS +cv2.CALIB_SAME_FOCAL_LENGTH + \
            cv2.CALIB_RATIONAL_MODEL +cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

    """-------------------------------------------------------------------------"""
    """Stereo calibration"""
    """-------------------------------------------------------------------------"""
    rms_stereo, M0, d0, M1, d1, R, T, E, F = cv2.stereoCalibrate(object_points, image_points0, image_points1,
                                      camera_matrix0, dist_coeffs0, camera_matrix1, dist_coeffs1,
                                      img0.shape[0:2][::-1], criteria=criteria, flags=flags)
    
    print("STEREO: RMS left to  right re-projection error: ", rms_stereo)
    """-------------------------------------------------------------------------"""
    """Stereo rectification"""
    """-------------------------------------------------------------------------"""
    # Q holds the quintessence of the algorithm, the reprojection matrix
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(M0, d0, M1, d1, img0.shape[0:2][::-1], R, T, alpha=-1, flags=cv2.CALIB_ZERO_DISPARITY)

    """-------------------------------------------------------------------------"""
    """Distortion map calculation"""
    """-------------------------------------------------------------------------"""
    mx0, my0 = cv2.initUndistortRectifyMap(M0, d0, R0, P0, img0.shape[0:2][::-1], cv2.CV_32FC1)
    mx1, my1 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img0.shape[0:2][::-1], cv2.CV_32FC1)

    """-------------------------------------------------------------------------"""
    """Save parameters to file"""
    """-------------------------------------------------------------------------"""
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

    # set_trace()
if args.run:
    if not isfile(parameter_path):
        print("Calibration is needed to proceed, please run the program with the --calibrate flag")
        input()
        print("Exiting program...")
        exit(-1)

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
    # stereo_disparity = cv2.StereoBM_create(numDisparities=64, blockSize=13)
    left_matcher = cv2.StereoBM_create(numDisparities=128, blockSize=5)

    # the bigger numDisparity is, the bigger the range of the z cootdinate
    # blockSize = 17
    # stereo_disparity = cv2.StereoSGBM_create(-16, 96, blockSize, 3*blockSize*blockSize, 16*blockSize*blockSize, 16, speckleWindowSize=0, speckleRange=1)


    window_size = 3
    # left_matcher = cv2.StereoSGBM_create(
        # minDisparity=0,
        # numDisparities=96,
        # blockSize=window_size,
        # P1=8*5*window_size**2,
        # P2=16 * 5 * window_size ** 2,
        # disp12MaxDiff=1,
        # uniquenessRatio=15,
        # speckleWindowSize=0,
        # speckleRange=2,
        # preFilterCap=63,
       # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 100000
    sigma = 1.0
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)



    # center point lists
    marker_b_array = np.ndarray((1,3))
    marker_b_array[0] = (0,0,0)
    marker_g_array= np.ndarray((1,3))
    marker_g_array[0] = (0,0,0)

    # start video acquisition loop
    while True:
        # Capture frame-by-frame
        capture_object0.grab()
        capture_object1.grab()

        ret0, frame0 = capture_object0.retrieve()
        ret1, frame1 = capture_object1.retrieve()

        # read the camera streams
        img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('im0', img0)
        # cv2.imshow('im1', img1)

        """-------------------------------------------------------------------------"""
        """RGB to HSV"""
        """-------------------------------------------------------------------------"""

        # color space to HSV
        img0_hsv = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
        img1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

        """-------------------------------------------------------------------------"""
        """Color masks"""
        """-------------------------------------------------------------------------"""
        # mask creation for green marker
        green_mask0 = cv2.inRange(img0_hsv, (40, 50, 0), (80, 225, 255))

        # blue mask
        blue_mask0 = cv2.inRange(img0_hsv, (90, 50, 0), (125, 225, 225))

        # # red mask (wraps around the values -> 2 parts)
        # red_mask0 = cv2.inRange(img0_hsv, (0, 40, 20), (10, 240, 200)) \
        #                     + cv2.inRange(img0_hsv, (170, 40, 20), (179, 240, 200))


        """-------------------------------------------------------------------------"""
        """Remap"""
        """-------------------------------------------------------------------------"""
        img0_rm = cv2.remap(img0, mx0, my0, cv2.INTER_LINEAR)
        img1_rm = cv2.remap(img1, mx1, my1, cv2.INTER_LINEAR)
        cv2.imshow('rm0', img0_rm)
        cv2.imshow('rm1', img1_rm)


        """-------------------------------------------------------------------------"""
        """Marker detection"""
        """-------------------------------------------------------------------------"""
        # green marker
        green_mask0, roi_g_0, gx0, gy0, ret_g = mask_processing(green_mask0, img0_rm)

        # blue marker
        blue_mask0, roi_b_0, bx0, by0, ret_b = mask_processing(blue_mask0, img0_rm)

        """-------------------------------------------------------------------------"""
        """Marker display"""
        """-------------------------------------------------------------------------"""
        # if args.display_markers:
        if True:
            marker_img = np.zeros(roi_g_0.shape)  # create black image for display

            # draw circles
            cv2.circle(marker_img, (int(np.round(by0)), int(np.round(bx0))), 10, 255, 20)
            cv2.circle(marker_img, (int(np.round(gy0)), int(np.round(gx0))), 30, 128, 40)

            cv2.imshow('Markers', marker_img)
            cv2.imshow('roi', roi_g_0)

        """-------------------------------------------------------------------------"""
        """Disparity map calculation"""
        """-------------------------------------------------------------------------"""
        # disparity_map = stereo_disparity.compute(img0_rm, img1_rm)
        # cv2.filterSpeckles(disparity_map, 0, 64, 32) #filter out noise
        # cv2.filterSpeckles(disparity_map, 0, 512, 32)



        # compute disparity image from undistorted and rectified versions
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)
        # credit goes to Toby Breckon, Durham University, UK for sharing this caveat
        # disparity_scaled = disparity_map
        # set_trace()
        # disparity_scaled = (disparity_map / 16.).astype(np.float32) + abs(disparity_map.min())
        # set_trace()



        displ = left_matcher.compute(img0, img1).astype(np.float32)/16.
        dispr = right_matcher.compute(img0, img1).astype(np.float32)/16.
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, img0_rm, None, dispr)  # important to put the left image here!!!

        if args.display_disparity:
            # show the remapped images and the scaled disparity map (modified for 8 bit display)
            # cv2.imshow('Disparity map', (disparity_map / 16.).astype(np.uint8))# + abs(disparity_map.min()))
            filteredImg2 = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            filteredImg2 = np.uint8(filteredImg2)
            cv2.imshow('Disparity Map', filteredImg2)

        """-------------------------------------------------------------------------"""
        """Image reprojection into 3D"""
        """-------------------------------------------------------------------------"""
        # use disparity scaled
        img_reproj = cv2.reprojectImageTo3D(filteredImg, Q)


        if args.display_reprojection:
            cv2.imshow('3d', img_reproj)

        """-------------------------------------------------------------------------"""
        """Marker coordinate logging"""
        """-------------------------------------------------------------------------"""
        center_gx = int(np.round(gx0))
        center_gy = int(np.round(gy0))
        # print(img0_rm.shape[1])
        filter_img = img_reproj[max(center_gx-5,0):min(center_gx+5, img0_rm.shape[0]),max(center_gy-5, 0):min(center_gy+5, img0_rm.shape[1])]

        center_3d = img_reproj[center_gx, center_gy].reshape((1,3))
        # set_trace()
        # print(center_3d)
        center_3d[0, 0] = filter_img[:,:,0].mean()
        center_3d[0, 1] = filter_img[:,:,1].mean()
        center_3d[0, 2] = filter_img[:,:,2].mean()
        # print(filter_z.shape)
        # print(filter_z)
        print(filteredImg.min(), filteredImg.max(), img_reproj[:,:,2].min(),img_reproj[:,:,2].max())
        # print(filter_z[:,:,2].mean())
        # exit(-1)
        # append points only if both markers were found
        print(img_reproj[center_gx,center_gy].reshape((1,3)))
        # print(disparity_map.min(), disparity_map.max(), disparity_scaled.min(), disparity_scaled.max(), img_reproj[:,:,2].min(),img_reproj[:,:,2].max())
        if ret_g: #and ret_b:
        # if True:
            # get marker coordinates
            # set_trace()
            #todo: centerpoint filtering?
            # marker_b_array = np.append(marker_b_array, img_reproj[int(np.round(bx0)),int(np.round(by0))].reshape((1,3)), axis=0)
            marker_g_array = np.append(marker_g_array, center_3d, axis=0)

        else:
            # append last element
            pass
            # note if one of the markers were found and appended, then
            # the geometry of the object could not have been guaranteed
            # marker_b_array = np.append(marker_b_array, marker_b_array[-1], axis=0)
            # marker_g_array = np.append(marker_g_array, marker_g_array[-1], axis=0)

        if len(marker_b_array) > 1 and marker_b_array[-1].sum() == 0:
            set_trace()


        # set_trace()
        """-------------------------------------------------------------------------"""
        """Keyboard handling"""
        """-------------------------------------------------------------------------"""
        if cv2.waitKey(40) & 0xFF == 27:
            # set_trace()
            # omit the first point (origin)
            # ax.scatter(marker_b_array[1:,0], marker_b_array[1:,1], marker_b_array[1:,2], label='blue marker')
            # ax.scatter(marker_g_array[1:,0], marker_g_array[1:,1], marker_g_array[1:,2], label='green marker', c='g')
            # ax.set_xlim(max(-1,marker_g_array[1:,0].min()), min(1,marker_g_array[1:,0].max()))
            # ax.set_ylim(max(-1,marker_g_array[1:,1].min()), min(1,marker_g_array[1:,1].max()))
            # ax.set_zlim(max(-1,marker_g_array[1:,2].min()), min(1,marker_g_array[1:,2].max()))
            # ax.legend()
            # plt.show()
            break


    """-------------------------------------------------------------------------"""
    """Cleanup"""
    """-------------------------------------------------------------------------"""
    # When everything done, release the capture
    capture_object0.release()
    capture_object1.release()
    cv2.destroyAllWindows()

