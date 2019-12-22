import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



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


######################################################################################################################

def find_best_fitting_circle(img_file, threshold_value, show_circle = False):
    img = cv2.imread(img_file)
    gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    # getting a binary image
    _, thresh = cv2.threshold(gray,threshold_value,255,0)
    if show_circle:
        cv2.imshow("img", thresh)
        cv2.waitKey()

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    morph_img = thresh.copy()
    cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
    contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    cnt=contours[areas.index(sorted_areas[-1])] # the biggest contour
    # min circle (green)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(img, center, radius, (0, 255, 0), 1)

    if show_circle:
        cv2.imshow("morph_img",morph_img)
        cv2.imshow("img", img)
        cv2.waitKey()

    return int(x), int(y), radius

########################################################################################################################
def picture_masked_cut(img_file, threshold_for_circle, border_distance, threshold_in_circle, show_circle=False, show_image_switch=False) :

    x_center, y_center, radius = find_best_fitting_circle(img_file, threshold_for_circle, show_circle)

    x_zero = x_center - radius
    y_zero = y_center - radius
    x_max = x_center + radius
    y_max = y_center + radius

    mask_circle_diameter = 2*radius-border_distance # diameter for big image

    # display grid on top of image
    display_grid_switch = False

    # read in as uint8
    src_frame_origin = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # cut image so the circle is in the center and the parts outside of the circle are deleted
    src_frame = src_frame_origin[ y_zero:y_max, x_zero:x_max]

    if show_image_switch:
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
    cv2.circle(mask_one, (int(width/2), int(height/2)), int(mask_circle_diameter/2), (1, 1, 1), -1)

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
    cv2.circle(mask_two, (int(width/2), int(height/2)), int(mask_circle_diameter/2), (0, 0, 0), -1)

    # show the mask
    if show_image_switch:
        show_window_with_user_setting(mask_two, 'Mask Two', 0)

    # with this matrix operation everything at the outside of the circle is removed
    img_2 = np.asarray(src_frame) + np.asarray(mask_two)

    # obtain the threshold using the greyscale image / threshold-type binary: values above eg. 10 are raised to 255
    ret, frame_threshold = cv2.threshold(img_2, threshold_in_circle, 255, 0)

    # show the mask
    if show_image_switch:
        show_window_with_user_setting(frame_threshold, 'Finish', 0)

    return frame_threshold, mask_two
##########################################################################################################


#module for displaying the image with added grid on top of it
# Grid lines at these intervals (in pixels)
# dx and dy can be different

def show_voxel_grid(grid_size, img_file):
    # Custom grid color
    grid_color = 0
    grid_img = img_file.copy()
    arr = np.asarray(grid_img)
    arr[:, ::grid_size] = grid_color
    arr[::grid_size, :] = grid_color
    show_window_with_user_setting(arr, 'arr', 0)

   # plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    #plt.show()
    #plt.waitforbuttonpress()
    print('done')


###########################################################################################################
#getting a subsection of the image and checking whether black point is inside

def check_image_for_black_and_border(img_file, contour_file, grid_size, show_image_switch=False):

    vox_loc_dict = {}
    vox_black_dict = {}

    height = img_file.shape[0]
    width = img_file.shape[1]

    num_grid_x = math.ceil(width/grid_size)
    num_grid_y = math.ceil(height/grid_size)

    grid_display_picture = np.copy(img_file)

    for j in range(num_grid_x):
        for i in range(num_grid_y):
            # if np.isin(image_1_1, 0):
            cur_image = img_file[i * grid_size:(i + 1) * grid_size,
                        j * grid_size:(j + 1) * grid_size]
            cur_cutout = cv2.rectangle(grid_display_picture, (j * grid_size, i * grid_size),
                                       ((j + 1) * grid_size, (i + 1) * grid_size), 0, 1)

            cur_image_mask = contour_file[i * grid_size:(i + 1) * grid_size,
                             j * grid_size:(j + 1) * grid_size]

            if show_image_switch:
                show_window_with_user_setting(cur_image_mask, 'mask_cutout', 0)
                show_window_with_user_setting(cur_cutout, 'cutout_image', 0)
                show_window_with_user_setting(cur_image, 'cur_image', 0)

            if np.any(cur_image[:, :] == 0):
                text_1 = '+blk'
            else:
                text_1 = '-blk'

            if np.all(cur_image_mask[:, :] == 255):
                text_2 = 'Outside'
            elif np.all(cur_image_mask[:, :] == 0):
                text_2 = 'inside'
            else:
                text_2 = 'border'

            #print('Voxel_{}_{}: '.format(i, j) + text_1 + ' and ' + text_2)

            vox_loc_dict['Vox_{}_{}'.format(j,i)] = text_2
            vox_black_dict['Vox_{}_{}'.format(j,i)] = text_1


    print(vox_black_dict)
    print(vox_loc_dict)


########################################################################################################################
if __name__ == "__main__":
    img_file = 'GridTest.tif'
    frame_threshold, mask_two = picture_masked_cut(img_file, 140, 5, 120, show_circle=True, show_image_switch=True)

    show_voxel_grid(30, frame_threshold)

    check_image_for_black_and_border(frame_threshold, mask_two, 30, show_image_switch=True)




