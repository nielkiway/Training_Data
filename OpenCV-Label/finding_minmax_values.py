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
stl_circle_image_path = '/home/jan/Documents/Trainingsdaten/ZPs/ZP4/200113_ZP4_Bildstapel_50µm_700x746_1905bis7905/mitGitter_0.tif'

img = cv2.imread(stl_circle_image_path)
gray = cv2.imread(stl_circle_image_path, cv2.IMREAD_GRAYSCALE)

# manually find the min and max values by looking at the ct picture with the perfect circle inside

xmin = 20
xmax = 556
ymin = 90
ymax = 626

diameter_x = xmax-xmin
diameter_y = ymax-ymin

if diameter_x != diameter_y:
    print('Adjust min and max values so the diameter in both directions is the same!')

# cut image and show it
cut_image_rgb = img[ymin:ymax, xmin:xmax]
cut_image = gray[ymin:ymax, xmin:xmax]

show_window_with_user_setting(cut_image, 'image cut', 0)






