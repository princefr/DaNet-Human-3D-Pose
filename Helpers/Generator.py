import tensorflow as tf
from tensorflow.python import keras
import numpy as np






class Generator(keras.utils.Sequence):
    def __init__(self, batch_size, cameras, poses_3d, poses_2d):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
        self.batch_size = batch_size
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

    def __len__(self):
        print("yes")

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        poses_2d = [self.poses_2d[k] for k in indexes]
        poses_3d = [self.poses_3d[k] for k in indexes]

        return self.generate_data(poses_2d, poses_3d)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.poses_2d))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def generate_data(self, poses_2d, poses_3d):
        print("yikes")