'''
@ author: Jan Klein
@ date: 13-01-2020
@ co: Fraunhofer IWU

general idea:

by gridwise inspecting the CT-Images it is checked whether the grid contains black pixels (-> porosity). By comparing
the current grid with a mask representing the background it is additionally checked whether the grid is within the
sample, in the border region or outside of the sample. The information is stored in a nd numpy array, which is used as
a storage for the labelling information

'''

import cv2
from combination_autocontour_Grid import show_window_with_user_setting
from finding_minmax_values import xmin, xmax, ymin, ymax
import numpy as np
import math
import pandas as pd


# input data
min_Slice = 376
max_Slice = 1551

threshold_binary = 120
correction_radius = 20  # radius difference for moving in the contour circle

threshold_porosity_abs = 125
# minimum number of pixels that must be black in order for the algorithm to classify the grid as pore-grid
# needs to be set = 1 if no threshold should be applied

grid_size_in_bit = 874
max_diameter_bit = 1748

csv_file_path = '/home/jan/Documents/Trainingsdaten/ZPs/ZP9/grid_size={}_threshold_porosity={}.csv'.format(grid_size_in_bit, threshold_porosity_abs)
img_folder_path = '/home/jan/Documents/Trainingsdaten/ZPs/ZP9/201315_ZP9_Bildstapel_50µm_700x746_1880bis7755/201315_ZP9_Bildstapel_50µm_700x746_1880bis7755_'


show_images = False
show_label_generation = False

# basic calculations
max_diameter_pixel = xmax-xmin
grid_size_in_pixel = int(max_diameter_pixel / max_diameter_bit * grid_size_in_bit)
number_slices_total = max_Slice-min_Slice



# start image processing

label_storage_array = np.empty([0, 6], dtype=int)

for img_number in range(number_slices_total):

    cur_img_path = img_folder_path+'{:04d}.tif'.format(img_number)

    img = cv2.imread(cur_img_path)
    gray = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)

    cut_image_rgb = img[ymin:ymax, xmin:xmax]
    cut_image = gray[ymin:ymax, xmin:xmax]

    height = cut_image.shape[0]
    width = cut_image.shape[1]

    if show_images:
        show_window_with_user_setting(cut_image_rgb, 'image cut', 0)

    # getting a binary image
    _, thresh = cv2.threshold(cut_image, threshold_binary, 255, 0)

    if show_images:
        show_window_with_user_setting(thresh, 'thresh', 0)

    # finding all the contours in the image
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 1))
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
    circle_img = cut_image.copy()
    cv2.circle(circle_img, center, radius, (255, 255, 255), 2)
    if show_images:
        show_window_with_user_setting(circle_img, 'with added circle', 0)

    # so far: image changed to binary image and minEnclosingCircle found

    # making everything within the circle a 1 and everything outside a 0 and the other way round for a second mask

    # within the circle: 1, outside of circle: 0
    mask_circle_one = np.zeros([height, width], dtype=np.uint8)
    mask_circle_one.fill(0)

    # draw circle with ones
    cv2.circle(mask_circle_one, center, radius - correction_radius, (1, 1, 1), -1)
    if show_images:
        show_window_with_user_setting(cut_image * mask_circle_one, 'mask_one', 0)

    inside_circle = cut_image.copy()
    inside_circle *= mask_circle_one

    # so far mask created which makes everything outside of the circle a 0 and everything inside stays the same

    # within the circle: 0, outside of circle: 1
    mask_circle_two = np.zeros([height, width], dtype=np.uint8)
    mask_circle_two.fill(1)

    # draw circle with ones
    cv2.circle(mask_circle_two, center, radius - correction_radius, (0, 0, 0), -1)
    if show_images:
        show_window_with_user_setting(cut_image * mask_circle_two, 'mask_two', 0)

    # mask one
    stencil = np.zeros(shape = cut_image_rgb.shape, dtype = np.uint8)
    stencil[:,:] = 0
    color = 1,1,1
    color_img = cv2.fillPoly(stencil, contours, color)
    mask_one = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # mask two
    stencil_2 = np.zeros(shape=cut_image_rgb.shape, dtype=np.uint8)
    stencil_2[:, :] = 255
    color_2 = 0, 0, 0
    color_img_2 = cv2.fillPoly(stencil_2, contours, color_2)
    mask_two = cv2.cvtColor(color_img_2, cv2.COLOR_BGR2GRAY)
    mask_two = mask_two * mask_circle_two + mask_circle_one

    if show_images:
        show_window_with_user_setting(mask_two, 'mask_two', 0)

    img_final_masked = mask_one * cut_image + mask_two

    # subtract the middle circle
    img_final_masked *= mask_circle_two

    # add the original inside circle
    img_final_masked += inside_circle

    if show_images:
        show_window_with_user_setting(img_final_masked, 'with mask_one and mask_two ', 0)

    # so far background made white

    _, image_masked_and_cut_binary = cv2.threshold(img_final_masked, 120, 255, 0)

    if show_images:
        show_window_with_user_setting(image_masked_and_cut_binary, 'Image masked and cut binary', 0)
        show_window_with_user_setting(mask_two, 'mask_two', 0)

    grid_color = 0
    grid_img = image_masked_and_cut_binary.copy()
    arr = np.asarray(grid_img)
    arr[:, ::grid_size_in_pixel] = grid_color
    arr[::grid_size_in_pixel, :] = grid_color
    if show_images:
        show_window_with_user_setting(arr, 'arr', 0)

    # labeling
    check_array = np.empty([0, 6], dtype=int)

    height = image_masked_and_cut_binary.shape[0]
    width = image_masked_and_cut_binary.shape[1]

    num_grid_x = math.ceil(width/grid_size_in_pixel)
    num_grid_y = math.ceil(height/grid_size_in_pixel)


    #print('num_grid_x ' + str(num_grid_x))
    #print('num_grid_y ' + str(num_grid_y))

    grid_display_picture = np.copy(image_masked_and_cut_binary)

    for j in range(num_grid_x):
        for i in range(num_grid_y):
            cur_image = image_masked_and_cut_binary[i * grid_size_in_pixel:(i + 1) * grid_size_in_pixel,
                        j * grid_size_in_pixel:(j + 1) * grid_size_in_pixel]
            cur_cutout = cv2.rectangle(grid_display_picture, (j * grid_size_in_pixel, i * grid_size_in_pixel),
                                       ((j + 1) * grid_size_in_pixel, (i + 1) * grid_size_in_pixel), 0, 1)

            cur_image_mask = mask_two[i * grid_size_in_pixel:(i + 1) * grid_size_in_pixel,
                             j * grid_size_in_pixel:(j + 1) * grid_size_in_pixel]

            if show_label_generation:
                show_window_with_user_setting(cur_image_mask, 'mask_cutout', 0)
                show_window_with_user_setting(cur_cutout, 'cutout_image', 0)
                show_window_with_user_setting(cur_image, 'cur_image', 0)

            # edit here for setting a threshold or adding different categories
            # it is still just used for a binary classification here in this context

            total_num_pixels = cur_image.shape[0]*cur_image.shape[1]
            num_black_pixels = total_num_pixels - cv2.countNonZero(cur_image)
            print(num_black_pixels)

            if num_black_pixels >= threshold_porosity_abs:
                text_1 = 1  # code for black is inside
            else:
                text_1 = 0  # code for no black is inside


#            if np.any(cur_image[:, :] == 0):
#                text_1 = 1  # code for black is inside
#            else:
#                text_1 = 0  # code for no black is inside

            if np.all(cur_image_mask[:, :] == 255):
                text_2 = -1     # code for outside
            elif np.all(cur_image_mask[:, :] == 0):
                text_2 = 1   # code for inside
            else:
                text_2 = 0   # code for border

            cur_arr = np.stack((img_number+min_Slice, j, i, text_2, text_1, num_black_pixels), axis=-1)
            # img_number + min slice is the current slice which is also stored
            check_array = np.vstack((check_array, cur_arr))

    label_storage_array = np.vstack((label_storage_array, check_array))

label_df = pd.DataFrame(data = label_storage_array, columns=(['Slice', 'x-grid', 'y-grid', 'Position', 'Poren', 'num Black pixels']))
label_df.to_csv(csv_file_path)
print('done')