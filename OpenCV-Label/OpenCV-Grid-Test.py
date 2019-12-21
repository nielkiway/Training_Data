# Purpose - Video-Data-Analysis
# goal is to extract information out of the meltpool-monitoring-system
# 2019-12-20 modified for pore detection
# 2019-12-16 first commit
#
# 3.4.3.18 - OpenCV-Version

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math


def hough_circle_algorithmen(src_frame, threshold, distance_between_circles, param1, param2, min_radius, max_radius):
    """

    :param src_frame:
    :param threshold:
    :param distance_between_circles:
    :param param1:
    :param param2:
    :param min_radius:
    :param max_radius:
    :return:
    """
    # transform to gray image for edge detection
    frame_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)

    # obtain the threshold using the greyscale image / threshold-type binary: values above eg. 10 are raised to 255
    ret, frame_threshold = cv2.threshold(frame_gray, threshold, 255, 0)

    # blur the contur
    frame_threshold_blur = cv2.medianBlur(frame_threshold, 7)

    # find all the contours in the binary image
    # https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghcircles.html
    rows = frame_threshold_blur.shape[0]
    circles = cv2.HoughCircles(frame_threshold_blur, cv2.HOUGH_GRADIENT, 2, distance_between_circles, param1=param1, param2=param2, minRadius=min_radius,
                               maxRadius=max_radius)
    return circles

def show_window_with_user_setting(image, name, wait):
    """

    :param image:
    :param name:
    :param wait:
    :return:
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 600)
    cv2.imshow(name, image)
    cv2.waitKey(0)

def hough_circle_detection(file_name):
    # storage for circle-data
    circles_list = []

    # mask_circle_diameter = 75/337 # ratio from first-test image
    x_zero = 112
    y_zero = 85
    x_max = 248
    y_max = 220
    mask_circle_diameter = x_max-x_zero # diameter for big image
    offset_x = 0#-240
    offset_y = 0#-65

    # show image
    show_image_switch = True

    # display grid on top of image
    display_grid_switch = False

    # src_frame
    src_frame = None

    # for tif-Files
    if '.tif' in file_name:

        # Viel Spaß! Gruß M.

        # read in as uint16 - not working_MJ
        # src_frame = cv2.imread(file_name, cv2.COLOR_BGR2GRAY)
        # src_frame = np.uint8(src_frame)

        # read in as uint8
        src_frame_origin = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # show input image


        src_frame = src_frame_origin[x_zero:x_max, y_zero:y_max]

        if show_image_switch:
            #show_window_with_user_setting(src_frame[x_zero:x_max,y_zero:y_max], 'Input', 0)
            show_window_with_user_setting(src_frame_origin, 'Input', 0)
            show_window_with_user_setting(src_frame, 'Cut', 0)


        # create mask to remove the outer area of the image
        height = src_frame.shape[0]
        width = src_frame.shape[1]
        print(height)
        print(width)

        mask_one = np.zeros([height, width], dtype=np.uint8)
        mask_one.fill(0)

        # draw circle with ones
        cv2.circle(mask_one, (int(width/2 + offset_x), int(height/2 + offset_y)), int(mask_circle_diameter/2), (1, 1, 1), -1)

        # show the mask
        if show_image_switch:
            show_window_with_user_setting(mask_one, 'Mask one', 0)

        # with this matrix operation everything at the outside of the circle is removed
        src_frame *= mask_one

        # show the modified src_frame
        if show_image_switch:
            show_window_with_user_setting(src_frame, 'Input after first mask', 0)

        # create mask to make the outer black part white
        mask_two = np.zeros([height, width], dtype=np.uint8)
        mask_two.fill(255)

        # draw circle with zeros
        cv2.circle(mask_two, (int(width/2 + offset_x), int(height/2 + offset_y)), int(mask_circle_diameter/2), (0, 0, 0), -1)

        # show the mask
        if show_image_switch:
            show_window_with_user_setting(mask_two, 'Mask Two', 0)

        # with this matrix operation everything at the outside of the circle is removed
        img_2 = np.asarray(src_frame) + np.asarray(mask_two)

        # obtain the threshold using the greyscale image / threshold-type binary: values above eg. 10 are raised to 255
        ret, frame_threshold = cv2.threshold(img_2, 130, 255, 0)

        # show the mask
        show_window_with_user_setting(frame_threshold, 'Finish', 0)

        ##########################################################################################################


        #module for displaying the image with added grid on top of it
        # Grid lines at these intervals (in pixels)
        # dx and dy can be different

        if display_grid_switch:
            dx, dy = 40, 40

            # Custom grid color
            grid_color = 0

            image = frame_threshold
            arr = np.asarray(image)
            arr[:, ::dy] = grid_color
            arr[::dx, :] = grid_color
            plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.waitforbuttonpress()




        ###########################################################################################################
        #getting a subsection of the image and checking whether black point is inside

        size_voxel_pix = 20

        num_vox_x = math.ceil(width/size_voxel_pix)
        num_vox_y = math.ceil(height/size_voxel_pix)

        grid_display_picture = np.copy(frame_threshold)

        image_1_1 = frame_threshold[:size_voxel_pix, :size_voxel_pix]
        show_window_with_user_setting(image_1_1, 'Image_1_1', 0)


        #works in one dimension -> extend to other dimension

        for j in range (num_vox_y):
            for i in range(num_vox_x):
            #if np.isin(image_1_1, 0):
                cur_image = frame_threshold[i*size_voxel_pix:(i+1)*size_voxel_pix, j*size_voxel_pix:(j+1)*size_voxel_pix]
                cur_cutout = cv2.rectangle(grid_display_picture, (j*size_voxel_pix,i*size_voxel_pix), ((j+1)*size_voxel_pix, (i+1)* size_voxel_pix), 0, 1)

                cur_image_mask = mask_two[i*size_voxel_pix:(i+1)*size_voxel_pix, j*size_voxel_pix:(j+1)*size_voxel_pix]


                show_window_with_user_setting(cur_image_mask, 'mask_cutout', 0)
                show_window_with_user_setting(cur_cutout, 'cutout_image', 0)
                show_window_with_user_setting(cur_image, 'cur_image', 0)


                if np.any(cur_image[:,:] == 0):
                    text_1 = 'Black point detected'
                else:
                    text_1 = ('no Black point detected')


                if np.all(cur_image_mask [:,:] == 255):
                    text_2 = 'Outside Voxel'
                elif np.all(cur_image_mask [:,:] == 0):
                    text_2 = 'inside Voxel'
                else:
                    text_2 = 'border voxel'

                print('Voxel_{}_{}: '.format(i,j) + text_1 + ' and ' + text_2)
        ###############################################################################################################
        #checking whether voxel is a center, border or outside-voxel: check is performed with mask 2

        show_window_with_user_setting(cur_image, 'cur_image', 0)
        















        ##########################################################################################################


        #circles_list.append(hough_circle_algorithmen(src_frame, 2, 50, 50, 20, 10, 30))

        # # Crop
        # img_3 = img_2[8:200, 0:200]
        #
        # cv2.imshow('Circle detection', img_3)
        # cv2.waitKey(0)


    # for png-Files
    if '.png' in file_name:
        src_frame = cv2.imread(file_name, cv2.IMREAD_COLOR)
        circles_list.append(hough_circle_algorithmen(src_frame))

    # for avi-Files
    if '.avi' in file_name:
        # open the video
        cap = cv2.VideoCapture(file_name)

        # read out the metadata to receive the number of frames
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # define start frame
        #cap.set(1, frames_total / 1)

        # index
        j, k = 0, 0

        # loop to catch each frame
        while cap.isOpened():
            # get image from the video
            ret, src_frame = cap.read(cv2.IMREAD_COLOR)

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # remove text and general the informations on the button of the images
            frame_resized = src_frame[8:120, :]

            # detection of the circle on single frame and append list
            circle = hough_circle_algorithmen(frame_resized, 2, 50, 50, 20, 10, 30)
            circles_list.append(circle)

            # draw the circle
            frame = draw_circle_in_frame(circle, frame_resized)
            # first line: write text on image
            frame = text_on_image('Frame ' + str(j), (5, 15), 0.33, frame)
            # second line: write text on image
            if circle is not None:
                frame = text_on_image(
                    'X' + str(circle.item(0)) + ' Y' + str(circle.item(1)) + ' D' + str(round(circle.item(2), 2)),
                    (5, 30), 0.29, frame)
            # third line: write text on image
            frame = text_on_image(
                'Vectornumber ' + str(k) + ' - 25', (5, 45),
                0.29, frame)

            # Vector-number calculation
            frames_without_circle = 10
            if j > frames_without_circle:
                # scan vector number
                l = 0

                # for i in range(-2, frames_without_circle * -1, -1):
                #     if circles_list[i] is None:
                #         l += 1
                # if l >= frames_without_circle and circles_list[-1] is not None:
                #     k += 1
                if circles_list[-1] is not None and circles_list[-2] is not None and circles_list[-3] is not None and  circles_list[-4] is not None and circles_list[-5] is None:
                    k += 1

            cv2.imshow('Circle detection', frame)
            cv2.waitKey(1)

            j += 1

        # close video-object
        cap.release()






def get_full_file_name_path_of_folder(file_extension, path):
    '''
    receive a filtered list with all interesting files in a folder

    :param file_extension: eg. avi, png, tif
    :param path: input as raw string
    :return: a list
    '''

    local_list = []

    for elem in list(filter(lambda x: str(file_extension) in x, os.listdir(path))):
        local_list.append(path + '\\' + elem)

    return local_list

if __name__ == "__main__":

        # load all files out of a folder
        # file_list = get_full_file_name_path_of_folder('avi', r'D:\Jaretzki\QualiPro\10_Mini1_Datenauswertung_MJ\Videodaten')

        file_list = ['GridTest.tif']

        #file_list = ['Test50000483.tif']

        # execute order
        for file in file_list:
            hough_circle_detection(file)
