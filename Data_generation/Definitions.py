'''
@ coop: Fraunhofer IWU
@ author: Jan Klein
@ date: 25-12-2019
'''

import h5py
import os
import math
from helping_functions import get_min_max_values, get_max_number_grid

'''
--------------------------------------------------------------------------------
Input data
'''
path_buildjob_h5 = '/home/jan/Documents/Trainingsdaten/ZP_4/ZP_4_full_part.h5'
path_destination_h5_folder = '/home/jan/Documents/Trainingsdaten/ZP_4/'
part_name = 'ZP4_combined'
grid_size = 200
max_slice_number_part = 1593 # doesn't need to be manually added

'''
-------------------------------------------------------------------------------
Basic operations and calculations
'''
#path_voxel_h5 = path_voxel_h5_folder+name_voxel_h5_file
print('start getting the min max values')

minX, minY, maxX, maxY = get_min_max_values(path_buildjob_h5, part_name, max_slice_number_part)
print('maxX ' + str(maxX))
print('minX ' + str(minX))
print('maxy ' + str(maxY))
print('minY ' + str(minY))

print('done getting the min max values')

n_grid_x, n_grid_y = get_max_number_grid(minX, minY, maxX, maxY, grid_size)
print('n_grid_x ' + str(n_grid_x))
print('n_grid_y ' + str( n_grid_y))

num_z_list = [i for i in range(max_slice_number_part)]
