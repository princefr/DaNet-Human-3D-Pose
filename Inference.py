import tensorflow as tf
from tensorflow.python import keras
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from Helpers.Utils import convert, convert_18
import numpy as np
from Helpers.data_utils import H36M_NAMES, normalization_stats, normalize_data, unNormalizeData, define_actions, read_3d_data, create_2d_data

from Helpers.viz import show3Dpose, show2Dpose
import copy
from Helpers import cameras





class HumanPose():
    def __init__(self, predict_14=False, visualise=False):
        self.model_path = "./pose.h5"
        print("this")
        self.convert_open = [-1,8,9,10,11,12,13,-1,1,0,5,6,7,2,3,4]
        self.h36m_2d_mean = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
        # mean list
        self.h36m_3d_mean = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.model = self.loadModel()
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.X = []
        self.Y = []
        self.Z = []
        self.IS_3D = False
        self.cnt = 0
        self.visualise = visualise

        actions = define_actions("All")
        number_of_actions = len(actions)
        SAVE_PATH = "./pose.h5"
        # Load camera parameters
        SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
        rcams = cameras.load_cameras("./h36m/cameras.h5", SUBJECT_IDS)
        camera_frame = False  # weither to convert 3D pose to camera coordinate
        predict_14 = False  # "predict 14 joints"
        use_sh = False  # "Use 2d pose predictions from StackedHourglass"
        data_dir = "./h36m/"  # data directory

        # Load 3d data and load (or create) 2d projections
        train_set_3d, test_set_3d, self.data_mean_3d, self.data_std_3d, self.dim_to_ignore_3d, self.dim_to_use_3d, train_root_positions, test_root_positions = read_3d_data(
            actions, data_dir, camera_frame, rcams, predict_14)

        train_set_2d, test_set_2d, self.data_mean_2d, self.data_std_2d, self.dim_to_ignore_2d, self.dim_to_use_2d = create_2d_data(
            actions, data_dir, rcams)

    def loadModel(self):
        return keras.models.load_model(self.model_path)




    def normalize_screen_coordinates(self, x, w, h):
        assert x.shape[-1] == 2
        return x/w*2 - [1, h/w]

    def DoTheInference(self, openpose_keypoints):
        if not isinstance(openpose_keypoints, np.ndarray):
            openpose_keypoints = np.array(openpose_keypoints)
        #if len(openpose_keypoints) > 18:
        #keypoints = convert_18(openpose_keypoints)
        keypoints = self.openpose2H36M(openpose_keypoints)
        normalized = self.normalize_data(keypoints, self.data_mean_2d, self.data_std_2d, self.dim_to_use_2d)
        predictions = self.model.predict(np.expand_dims(normalized, axis=0))[0]
        predictions = unNormalizeData(predictions, self.data_mean_3d, self.data_std_3d, self.dim_to_ignore_3d)
        if self.visualise:
            self.show3D(predictions)
        return predictions

    def normalize_data(self, data, data_mean, data_std, dimensions_to_use):
        data = data[:, dimensions_to_use]
        mu = data_mean[dimensions_to_use]
        stddev = data_std[dimensions_to_use]
        return np.divide((data - mu), stddev)

    def openpose2H36M(self, poses):
        # hip = Rhip + Lhip/ 2
        Hip = poses[8] + poses[11] / 2
        # Spine = Rshoulders / Lshoulders / 2
        Spine = poses[2] + poses[5] / 2

        # concatenate the openpose result with the spine and hip newly calculated
        spine_hip = np.array((Hip, Spine))
        poses = np.concatenate((poses, spine_hip)).astype(int)

        # Permutation that goes from SH detections to H36M ordering.
        OP_TO_GT_PERM = np.array([18, 8, 9, 10, 11, 12, 13, 19, 1, 0, 5, 6, 7, 2, 3, 4])

        # Permute the loaded data to make it compatible with H36M
        poses = poses[OP_TO_GT_PERM]
        poses = np.expand_dims(poses, axis=0)

        # Reshape into n x (32*2) matrix
        poses = np.reshape(poses, [poses.shape[0], -1])
        poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 2])

        dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
        dim_to_use_y = dim_to_use_x + 1

        dim_to_use = np.zeros(32, dtype=np.int32)
        dim_to_use[0::2] = dim_to_use_x
        dim_to_use[1::2] = dim_to_use_y
        poses_final[:, dim_to_use] = poses
        return poses_final




    def show3D(self, Keypoints_3d):
        fig = plt.figure()
        ax = Axes3D(fig)
        show3Dpose(Keypoints_3d, ax)
        #ani = animation.FuncAnimation(fig, plot, interval=1000)
        plt.show()
        plt.savefig("test_image_3d.png")