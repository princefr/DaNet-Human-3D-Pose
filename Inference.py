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

from Helpers.viz import show3Dpose, show2Dpose, show3dpose_new
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
        self.X = []
        self.Y = []
        self.Z = []
        self.IS_3D = False
        self.cnt = 0
        self.visualise = visualise


        SAVE_PATH = "./pose.h5"
        # Load camera parameters


    def loadModel(self):
        return keras.models.load_model(self.model_path, custom_objects={'mpjpe': self.mpjpe})

    def mpjpe(self, y_true, y_pred):
        """
        mean per -joint position error(i.e mean eclidean distance)
        often referred to as "Protocol #1 in many paper"
        :param y_true:
        :param y_pred:
        :return:
        """
        return tf.reduce_mean(tf.norm(y_pred - y_true, axis=len(y_true.shape) - 1))

    def normalize_screen_coordinates(self, x, w=640, h=480):
        assert x.shape[-1] == 2
        return x/w*2 - [1, h/w]

    def image_coordinates(self, X, w=640, h=480):
        assert X.shape[-1] == 2
        # Reverse camera frame normalization
        return (X + [1, h / w]) * w / 2


    def direct_inference(self, keypoints):
        keypoints = self.normalize_screen_coordinates(keypoints)
        predictions = self.model.predict(np.expand_dims(keypoints, axis=0))[0]
        #rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
        predictions = np.reshape(predictions, (1, 17, 3))
        predictions[:, :, 2] -= np.min(predictions[:, :, 2])
        #predictions = cameras.camera_to_world(predictions, rot, 0)
        self.show3D(predictions[0])

    def direct_show_3d(self, pose):
        pose = np.reshape(pose, (1, 17, 3))

        self.show3D(pose[0])

    def DoTheInference(self, openpose_keypoints):
        if not isinstance(openpose_keypoints, np.ndarray):
            openpose_keypoints = np.array(openpose_keypoints)
        #if len(openpose_keypoints) > 18:
        #keypoints = convert_18(openpose_keypoints)
        keypoints = self.openpose2H36M(openpose_keypoints)
        keypoints = self.normalize_screen_coordinates(keypoints)


        predictions = self.model.predict(np.expand_dims(keypoints, axis=0))[0]
        rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
        predictions = np.reshape(predictions, (1, 17, 3))
        #predictions[:, :, 2] -= np.min(predictions[:, :, 2])
        #predictions = cameras.camera_to_world(predictions, rot, 0)
        #predictions = unNormalizeData(predictions, self.data_mean_3d, self.data_std_3d, self.dim_to_ignore_3d)
        if self.visualise:
            self.show3D(predictions[0])
        return predictions

    def normalize_data(self, data, data_mean, data_std, dimensions_to_use):
        data = data[:, dimensions_to_use]
        mu = data_mean[dimensions_to_use]
        stddev = data_std[dimensions_to_use]
        return np.divide((data - mu), stddev)

    def openpose2H36M(self, poses):
        # hip = Rhip + Lhip/ 2
        Hip = poses[8] + poses[11] / 2

        # Spine = Rshoulders + Lshoulders / 2
        Spine = poses[2] + poses[5] / 2

        #head left-eye + right eye / 2
        Head = poses[14] + poses[15] / 2

        # concatenate the openpose result with the spine and hip newly calculated
        spine_hip = np.array((Hip, Spine, Head)) # 18 , 19, 20 ==>
        poses = np.concatenate((poses, spine_hip)).astype(int)

        # Permutation that goes from SH detections to H36M ordering.
        OP_TO_GT_PERM = np.array([18, 8, 9, 10, 11, 12, 13, 19, 1, 0, 20, 5, 6, 7, 2, 3, 4])

        # Permute the loaded data to make it compatible with H36M
        poses = poses[OP_TO_GT_PERM]
        #poses = np.expand_dims(poses, axis=0)

        # Reshape into n x (32*2) matrix
        #poses = np.reshape(poses, [poses.shape[0], -1])
        #poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 2])

        #dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
        #dim_to_use_y = dim_to_use_x + 1

        #dim_to_use = np.zeros(32, dtype=np.int32)
        #dim_to_use[0::2] = dim_to_use_x
        #dim_to_use[1::2] = dim_to_use_y
        #poses_final[:, dim_to_use] = poses
        return poses




    def show3D(self, Keypoints_3d):
        fig = plt.figure()
        ax = Axes3D(fig)
        show3dpose_new(Keypoints_3d, ax)
        #ani = animation.FuncAnimation(fig, plot, interval=1000)
        plt.show()
        fig.savefig('figure_3d.png')