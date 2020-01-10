'''
@ coop: Fraunhofer IWU
@ author: Jan Klein
@ date: 25-12-2019

The function is used to reorder the hdf5 data in a grid structure
'''


import h5py
import time
import os

from helping_functions import get_2D_data_from_h5_filtered_np_v2, dock_array_to_zero, create_single_grid_array_storage_reduced
from Definitions import path_buildjob_h5, path_destination_h5_folder, part_name, grid_size, minX, minY, n_grid_x,\
    n_grid_y, max_slice_number_part



def create_grid_data (num_slice):

    start_time = time.time()
    # 1. getting the data
    array_not_docked = get_2D_data_from_h5_filtered_np_v2\
        (path_buildjob_h5, part_name, 'Slice' + str("{:05d}".format(num_slice+1)))
    #"{:05d}" -> 1 becomes 00001 for accessibility in h5 file

    # 2. docking the data to 0
    array = dock_array_to_zero(array_not_docked, minX, minY)

    # 3. iterating over grid range
    for n_grid_y_cur in range(n_grid_y):  # iterating over number of voxels in y-direction
        # print('n_vox_y_init: ' + str(n_vox_y_init))
        for n_grid_x_cur in range(n_grid_x):  # iterating over number of voxels in x-direction
            # print('n_vox_x_init: '+ str(n_vox_x_init))
            array_grid_final = create_single_grid_array_storage_reduced(n_grid_x_cur, n_grid_y_cur, grid_size, array)

    # 4. saving the data in a hdf5 file for each slice -> multiprocessing possible

            path = path_destination_h5_folder + part_name + '_' + str(grid_size) + '.h5' # '/Slice_{}.hdf5'.format(num_slice)

            if not os.path.isfile(path):
                grid_hdf = h5py.File(path, "w")
                grid_hdf.close()


            with h5py.File(path, "a") as grid_hdf:
                #creating a voxel with the numbers of voxels in both direction in its name and filling it with data
                #if group is already existing don't create a new group

                if 'Slice' + str("{:05d}".format(num_slice+1)) not in grid_hdf:
                    grid_hdf.create_group('Slice' + str("{:05d}".format(num_slice+1)))

                grid_hdf['Slice' + str("{:05d}".format(num_slice+1))].create_group('grid_{}_{}_{}'.format(n_grid_x_cur, n_grid_y_cur, num_slice+1))

                grid_hdf['Slice' + str("{:05d}".format(num_slice+1))]['grid_{}_{}_{}'.format(n_grid_x_cur, n_grid_y_cur, num_slice+1)].create_dataset('X-Axis', data=array_grid_final[:, 0])
                grid_hdf['Slice' + str("{:05d}".format(num_slice+1))]['grid_{}_{}_{}'.format(n_grid_x_cur, n_grid_y_cur, num_slice+1)].create_dataset('Y-Axis',data = array_grid_final[:,1])
                grid_hdf['Slice' + str("{:05d}".format(num_slice+1))]['grid_{}_{}_{}'.format(n_grid_x_cur, n_grid_y_cur, num_slice+1)].create_dataset('Area', data = array_grid_final[:,2])
                grid_hdf['Slice' + str("{:05d}".format(num_slice+1))]['grid_{}_{}_{}'.format(n_grid_x_cur, n_grid_y_cur, num_slice+1)].create_dataset('Intensity', data = array_grid_final[:,3])

    print("handling slice {} took {} seconds ".format(num_slice + 1, (time.time() - start_time)))


if __name__ == '__main__':
    # 1.Step creating an empty hdf5 file for the voxels
    # voxel_hdf = h5py.File(path_voxel_h5, "w")
    # voxel_hdf.close()
    # 2. Multiprocessed filling up of voxel layers
    # with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
    #    executor.map(create_single_vox_layer, num_z_list)

    for num_slice in range(50): # (max_slice_number_part):
        create_grid_data(num_slice)

    # both processes finished
    print("Done!")


