import argparse
import os
import zipfile
import numpy as np
import h5py
from glob import glob
from shutil import rmtree

import sys
sys.path.append('../')
from Helpers.HumanDataset import HumanDataset
from Helpers.cameras import world_to_camera_frame, normalize_screen_coordinates, project_point_radial, world_to_camera, image_coordinates, project_to_2d

SUBJECT_IDS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
output_filename = 'data_3d_h36m'
output_filename_2d = 'data_2d_h36m_gt'
data_dir = "../h36m/"


def Prepare3D(subjects):
    output = {}
    for subject in subjects:
        output[subject] = {}
        file_list = glob(data_dir + subject + '/MyPoses/3D_positions/*.h5')
        assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]
            if subject == 'S11' and action == 'Directions':
                continue  # Discard corrupted video
            with h5py.File(f) as hf:
                positions = hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1)
                positions /= 1000  # Meters instead of millimeters
                output[subject][action] = positions.astype('float32')
    print('Saving...')
    np.savez_compressed(output_filename, positions_3d=output)
    print('Cleaning up...')
    print('Done. preparing the ddd poses')



def Prepare2D():
    # Create 2D pose file
    print('Computing ground-truth 2D poses...')
    dataset = HumanDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], T=cam['translation'])
                pos_2d = project_to_2d(np.reshape(pos_3d, (1, pos_3d.shape[0], pos_3d.shape[1], pos_3d.shape[2])), np.reshape(cam['intrinsic'], (1, cam['intrinsic'].shape[0])))
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
    print('Done.')

Prepare3D(SUBJECT_IDS)
Prepare2D()