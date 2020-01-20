'''
@ author: Jan Klein
@ date: 13-01-2020
@ co: Fraunhofer IWU

general idea:

by inspecting the CT image with circle on it picture is manually cut according to CT-stl-data

'''

import cv2
from combination_autocontour_Grid import show_window_with_user_setting


# input
stl_circle_image_path = '/home/jan/Documents/Trainingsdaten/ZPs/ZP9/201315_ZP9_Bildstapel_50µm_700x746_1880bis7755/ZP9_with_circle_stl0.tif'


img = cv2.imread(stl_circle_image_path)
gray = cv2.imread(stl_circle_image_path, cv2.IMREAD_GRAYSCALE)

# manually find the min and max values by looking at the ct picture with the perfect circle inside

xmin = 31
xmax = 498
ymin = 99
ymax = 566

diameter_x = xmax-xmin
diameter_y = ymax-ymin

if diameter_x != diameter_y:
    print('diameter x is ' + str(diameter_x))
    print('diameter y is ' + str(diameter_y))
    print('Adjust min and max values so the diameter in both directions is the same!')

# cut image and show it
cut_image_rgb = img[ymin:ymax, xmin:xmax]
cut_image = gray[ymin:ymax, xmin:xmax]

show_window_with_user_setting(cut_image, 'image cut', 0)





