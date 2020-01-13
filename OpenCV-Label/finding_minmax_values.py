'''
general idea:

by inspecting the CT image with circle on it picture is manually cut according to CT-stl-data

'''

import cv2
from combination_autocontour_Grid import picture_masked_cut_binary, check_image_for_black_and_border, show_window_with_user_setting
import numpy as np
import math

show_circle = True

stl_circle_image_path = '/home/jan/Documents/Trainingsdaten/200113_ZP4_Bildstapel_50µm_700x746_1905bis7905/200113_ZP4_Bildstapel_50µm_700x746_1905bis7905_1167.tif'

img = cv2.imread(stl_circle_image_path)
gray = cv2.imread(stl_circle_image_path, cv2.IMREAD_GRAYSCALE)

# manually find the min and max values by looking at the ct picture with the perfect circle inside

grid_size_in_bit = 50
max_diameter_bit = 1748
max_diameter_pixel = 536
grid_size_in_pixel = int(max_diameter_pixel/max_diameter_bit * grid_size_in_bit)



xmin = 20
xmax = 556
ymin = 90
ymax = 626

cut_image_rgb = img[ymin:ymax, xmin:xmax]
cut_image = gray[ymin:ymax, xmin:xmax]

show_window_with_user_setting(cut_image, 'image cut', 0)


#def find_best_fitting_circle(img_file, threshold_value, correction_radius, show_circle = False):

# reading in the image and transforming it ot grayscale

height = cut_image.shape[0]
width = cut_image.shape[1]



# getting a binary image
_, thresh = cv2.threshold(cut_image, 120, 255, 0)

show_window_with_user_setting(thresh, 'thresh', 0)

# finding all the contours in the image
element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 1))   # adapt the kernel size to the picture size
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
if show_circle:
    show_window_with_user_setting(circle_img, 'with added circle', 0)

# so far: image changed to binary image and minEnclosingCircle found

# making everything within the circle a 1 and everything outside a 0 and the other way round for a second mask

# within the circle: 1, outside of circle: 0
mask_circle_one = np.zeros([height, width], dtype=np.uint8)
mask_circle_one.fill(0)

correction_radius = 20 #edit

# draw circle with ones
cv2.circle(mask_circle_one, center, radius - correction_radius, (1, 1, 1), -1)
if show_circle:
    show_window_with_user_setting(cut_image * mask_circle_one, 'mask_one', 0)

inside_circle = cut_image.copy()
inside_circle *= mask_circle_one

# so far mask created which makes everything outside of the circle a 0 and everything inside stays the same

# within the circle: 0, outside of circle: 1
mask_circle_two = np.zeros([height, width], dtype=np.uint8)
mask_circle_two.fill(1)

# draw circle with ones
cv2.circle(mask_circle_two, center, radius - correction_radius, (0, 0, 0), -1)
if show_circle:
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

if show_circle:
    show_window_with_user_setting(mask_two, 'mask_two', 0)

img_final_masked = mask_one * cut_image + mask_two

# subtract the middle circle
img_final_masked *= mask_circle_two

# add the original inside circle
img_final_masked += inside_circle

if show_circle:
    show_window_with_user_setting(img_final_masked, 'with mask_one and mask_two ', 0)

# so far background made white

_, image_masked_and_cut_binary = cv2.threshold(img_final_masked, 120, 255, 0)

show_window_with_user_setting(image_masked_and_cut_binary, 'Image masked and cut binary', 0)
show_window_with_user_setting(mask_two, 'mask_two', 0)



grid_color = 0
grid_img = image_masked_and_cut_binary.copy()
arr = np.asarray(grid_img)
arr[:, ::grid_size_in_pixel] = grid_color
arr[::grid_size_in_pixel, :] = grid_color
show_window_with_user_setting(arr, 'arr', 0)


# check image for black and border


check_array = np.empty([0, 4], dtype=int)

height = image_masked_and_cut_binary.shape[0]
width = image_masked_and_cut_binary.shape[1]

num_grid_x = math.ceil(width/grid_size_in_pixel)
num_grid_y = math.ceil(height/grid_size_in_pixel)

print('num_grid_x ' + str(num_grid_x))
print('num_grid_y ' + str(num_grid_y))

grid_display_picture = np.copy(image_masked_and_cut_binary)

for j in range(num_grid_x):
    for i in range(num_grid_y):
        # if np.isin(image_1_1, 0):
        cur_image = image_masked_and_cut_binary[i * grid_size_in_pixel:(i + 1) * grid_size_in_pixel,
                    j * grid_size_in_pixel:(j + 1) * grid_size_in_pixel]
        cur_cutout = cv2.rectangle(grid_display_picture, (j * grid_size_in_pixel, i * grid_size_in_pixel),
                                   ((j + 1) * grid_size_in_pixel, (i + 1) * grid_size_in_pixel), 0, 1)

        cur_image_mask = mask_two[i * grid_size_in_pixel:(i + 1) * grid_size_in_pixel,
                         j * grid_size_in_pixel:(j + 1) * grid_size_in_pixel]

        #if show_image_switch:
        #show_window_with_user_setting(cur_image_mask, 'mask_cutout', 0)
        #show_window_with_user_setting(cur_cutout, 'cutout_image', 0)
        #show_window_with_user_setting(cur_image, 'cur_image', 0)

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

print(check_array)













# plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
#plt.show()
#plt.waitforbuttonpress()
print('done')




