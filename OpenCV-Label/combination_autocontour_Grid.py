import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



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

def find_best_fitting_circle(img_file, threshold_value, correction_radius, show_circle = False):

    # reading in the image and transforming it ot grayscale
    img = cv2.imread(img_file)
    gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    height = gray.shape[0]
    width = gray.shape[1]

    # getting a binary image
    _, thresh = cv2.threshold(gray, threshold_value, 255, 0)

    # finding all the contours in the image
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(10, 10))   # adapt the kernel size to the picture size
    morph_img = thresh.copy()
    cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
    contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    cnt = contours[areas.index(sorted_areas[-1])]

    # finding the minEnclosingCircle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)  # manually added
    circle_img = gray.copy()
    cv2.circle(circle_img, center, radius, (255, 255, 255), 15)
    show_window_with_user_setting(circle_img, 'with added circle', 0)

    # making everything whithin the circle a 1 and everything outside a 0 and the other way round for a second mask

    # within the circle: 1, outside of circle: 0
    mask_circle_one = np.zeros([height, width], dtype=np.uint8)
    mask_circle_one.fill(0)

    # draw circle with ones
    cv2.circle(mask_circle_one, center, radius - correction_radius, (1, 1, 1), -1)
    show_window_with_user_setting(gray * mask_circle_one, 'mask_one', 0)

    inside_circle = gray.copy()
    inside_circle *= mask_circle_one

    # within the circle: 0, outside of circle: 1
    mask_circle_two = np.zeros([height, width], dtype=np.uint8)
    mask_circle_two.fill(1)

    # draw circle with ones
    cv2.circle(mask_circle_two, center, radius - correction_radius, (0, 0, 0), -1)
    show_window_with_user_setting(gray * mask_circle_two, 'mask_two', 0)

    # mask one
    stencil = np.zeros(shape = img.shape, dtype = np.uint8)
    stencil[:,:] = 0
    color = 1,1,1
    color_img = cv2.fillPoly(stencil, contours, color)
    mask_one = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # mask two
    stencil_2 = np.zeros(shape=img.shape, dtype=np.uint8)
    stencil_2[:, :] = 255
    color_2 = 0, 0, 0
    color_img_2 = cv2.fillPoly(stencil_2, contours, color_2)
    mask_two = cv2.cvtColor(color_img_2, cv2.COLOR_BGR2GRAY)
    mask_two = mask_two * mask_circle_two + mask_circle_one
    show_window_with_user_setting(mask_two, 'mask_two', 0)


    img_final_masked = mask_one * gray + mask_two

    # subtract the middle circle
    img_final_masked *= mask_circle_two

    # add the original inside circle
    img_final_masked += inside_circle

    if show_circle:
        show_window_with_user_setting(img_final_masked, 'with mask_one and mask_two ', 0)

    return img_final_masked, mask_two, int(x), int(y), radius

########################################################################################################################
def picture_masked_cut_binary(img_file, threshold_for_circle, border_distance, threshold_binary, correction_radius, show_circle=False, show_image_switch=False) :

    img, mask, x_center, y_center, radius = find_best_fitting_circle(img_file, threshold_for_circle, correction_radius, show_circle)

    x_zero = x_center - (radius-border_distance)
    y_zero = y_center - (radius-border_distance)
    x_max = x_center + (radius-border_distance)
    y_max = y_center + (radius-border_distance)

    #mask_circle_diameter = 2*radius-border_distance # diameter for big image

    # display grid on top of image
    display_grid_switch = False

    # read in as uint8
    #src_frame_origin = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # cut image so the circle is in the center and the parts outside of the circle are deleted
    cut_image = img[ y_zero:y_max, x_zero:x_max]
    cut_mask = mask[ y_zero:y_max, x_zero:x_max]

    _, thresh = cv2.threshold(cut_image, threshold_binary, 255, 0)

    if show_image_switch:
        show_window_with_user_setting(cut_image, 'Image masked and cut', 0)
        show_window_with_user_setting(thresh, 'Image masked and cut binary', 0)
        show_window_with_user_setting(cut_mask, 'Mask cut', 0)


    # show the mask
    #if show_image_switch:
    #    show_window_with_user_setting(frame_threshold, 'Finish', 0)

    return cut_image, cut_mask
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

    #vox_loc_dict = {}
    #vox_black_dict = {}
    #check_df = pd.DataFrame(columns = ['Voxel_name', 'num_x', 'num_y', 'location', 'detected'])
    check_array = np.empty([0, 4], dtype=int)

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
                text_1 = 1  # code for black is inside
            else:
                text_1 = 0  # code for no black is inside

            if np.all(cur_image_mask[:, :] == 255):
                text_2 = -1     # code for outside
            elif np.all(cur_image_mask[:, :] == 0):
                text_2 = 1   # code for inside
            else:
                text_2 = 0   # code for border

            # check_df = check_df.append({'Voxel_name': 'Voxel_{}_{}'.format(j,i), 'num_x': j, 'num_y': i , 'location': text_2, 'detected':text_1}, ignore_index = True)
            # combo_processed_array = np.empty([0, 4], dtype=int)
            cur_arr = np.stack((j, i, text_2, text_1), axis=-1)
            check_array = np.vstack((check_array, cur_arr))

    return check_array



########################################################################################################################
if __name__ == "__main__":
    img_file = '191223_ZP4_0426.tif'

    #find_best_fitting_circle(img_file, 190, 60, show_circle = True)


    img = cv2.imread(img_file)
    show_window_with_user_setting( cv2.imread(img_file), 'Raw', 0)
    frame_threshold, mask_two = picture_masked_cut_binary(img_file, 170, 10, 140, 60,  show_circle=True, show_image_switch=True)

   # show_voxel_grid(300, frame_threshold)

    #df = check_image_for_black_and_border(frame_threshold, mask_two, 300, show_image_switch=False)

    #print(df)

