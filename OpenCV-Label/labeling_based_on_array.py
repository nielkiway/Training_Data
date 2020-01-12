'''
@author: Jan Klein
@company: Fraunhofer IWU
@date: 25-12-2019

This code takes a hdf5 file with data structured as a grid (Ratio Bit/Pixel) as input and a number of pictures of
different slices as input and

'''

import h5py
import numpy as np
from combination_autocontour_Grid import picture_masked_cut_binary, check_image_for_black_and_border

h5_unlabelled_path = '/home/jan/Documents/Trainingsdaten/ZP_4/_ZP4_combined_200.h5'
h5_labelled_path = '/home/jan/Documents/Trainingsdaten/ZP_4/_ZP4_combined_200_labelled.h5'

h5_labelled = h5py.File(h5_labelled_path,'w')
# creating the label structure
h5_labelled.create_group('no pores')
h5_labelled.create_group('pores')
h5_labelled['no pores'].create_group('inside')
h5_labelled['no pores'].create_group('border')
h5_labelled['pores'].create_group('inside')
h5_labelled['pores'].create_group('border')
h5_labelled.close()


# setting the grid_size in bit
grid_size_in_bit = 200
max_diameter_bit = 1748
max_diameter_pixel = 2074
grid_size_in_pixel = int(max_diameter_pixel/max_diameter_bit * grid_size_in_bit)

#for num_slice in range():
num_slice = 40
image_path = '191223_ZP4_0426.tif'
#image_path = '19_017_QualiPro_006' + str("{:04d}".format(num_slice+1))

# create the labeling array for a certain slice
frame_threshold, mask_two = picture_masked_cut_binary(img_file=image_path,
                                                      threshold_for_circle=170,
                                                      border_distance=10,
                                                      threshold_binary=140,
                                                      correction_radius=60,
                                                      show_circle=True,
                                                      show_image_switch=True)

array, num_grid_x, num_grid_y = check_image_for_black_and_border(img_file = frame_threshold,
                                                                 contour_file=mask_two,
                                                                 grid_size_in_pixel = grid_size_in_pixel,
                                                                 show_image_switch=True)

#checking every single grid for its values in the grid array
for x in range(num_grid_x):
    for y in range (num_grid_y):
        # getting the data for inside outside and so on from the array

        relevant_index = np.where((array[:,0] == x)*(array[:,1] == y))
        location_value = array[relevant_index][:,2]
        pore_value = array[relevant_index][:,3]

        print('grid_{}_{}_{}'.format(x, y, num_slice + 1))
        if location_value == -1:
            break
        elif location_value == 1:
            loc_group = 'inside'
        elif location_value == 0:
            loc_group = 'border'

        if pore_value == 1:
            pore_group = 'pores'
        elif pore_value == 0:
            pore_group = 'no pores'

        print('grid_{}_{}_{}'.format(x, y, num_slice + 1) + ' ' + loc_group + ' ' + pore_group)

        with h5py.File(h5_unlabelled_path,'r') as h5_old:
            with h5py.File(h5_labelled_path,'a') as h5_new:
                h5_old['Slice' + str("{:05}".format(num_slice+1))].copy('grid_{}_{}_{}'.format(x, y, num_slice+1),
                                                                      h5_new[pore_group][loc_group])












