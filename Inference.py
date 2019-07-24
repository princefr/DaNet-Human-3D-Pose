import tensorflow as tf
from tensorflow.python import keras
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from Utils import convert, convert_18
import numpy as np
from data_utils import H36M_NAMES

from viz import show3Dpose, show2Dpose






class HumanPose():
    def __init__(self):
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

    def loadModel(self):
        return keras.models.load_model(self.model_path)

    def search_name(self, name):
        j = 0
        for i in range(32):
            if (self.IS_3D):
                if (H36M_NAMES[i] == "Hip"):
                    continue
            else:
                if (H36M_NAMES[i] == "Neck/Nose"):
                    continue
            if (H36M_NAMES[i] == ""):
                continue
            if (H36M_NAMES[i] == name):
                return j
            j = j + 1
        return -1

    def draw_connect(self, from_id, to_id, color="#00aa00"):
        from_id = self.search_name(from_id)
        to_id = self.search_name(to_id)
        if (from_id == -1 or to_id == -1):
            return
        x = [self.X[from_id], self.X[to_id]]
        y = [self.Y[from_id], self.Y[to_id]]
        z = [self.Z[from_id], self.Z[to_id]]

        self.ax.plot(x, z, y, "o-", color=color, ms=4, mew=0.5)

    def plot(self, data, data_std_3d, data_mean_3d):
        plt.cla()

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Z axis')
        self.ax.set_zlabel('Y axis')
        self.ax.set_zlim([600, -600])

        global cnt, X, Y, Z, IS_3D
        k = self.cnt

        for mode in range(2):
            X = []
            Y = []
            Z = []

            if (mode == 0):
                self.IS_3D = True
            else:
                self.IS_3D = False

            for i in range(16):
                if self.IS_3D:
                    j = self.h36m_3d_mean[i]
                    X.append(data[i * 3 + 0] * data_std_3d[j * 3 + 0] + data_mean_3d[j * 3 + 0])
                    Y.append(data[i * 3 + 1] * data_std_3d[j * 3 + 1] + data_mean_3d[j * 3 + 1])
                    Z.append(data[i * 3 + 2] * data_std_3d[j * 3 + 2] + data_mean_3d[j * 3 + 2])
                else:
                    print("")


            if (IS_3D):
                self.draw_connect("Head", "Thorax", "#0000aa")
                self.draw_connect("Thorax", 'RShoulder')
                self.draw_connect('RShoulder', 'RElbow')
                self.draw_connect('RElbow', 'RWrist')
                self.draw_connect("Thorax", 'LShoulder')
                self.draw_connect('LShoulder', 'LElbow')
                self.draw_connect('LElbow', 'LWrist')
                self.draw_connect('Thorax', 'Spine')
                self.draw_connect('Spine', 'LHip')
                self.draw_connect('Spine', 'RHip')
                self.draw_connect('RHip', 'RKnee')
                self.draw_connect('RKnee', 'RFoot')
                self.draw_connect('LHip', 'LKnee')
                self.draw_connect('LKnee', 'LFoot')
            else:
                self.draw_connect("Head", "Thorax", "#0000ff")
                self.draw_connect("Thorax", 'RShoulder', "#00ff00")
                self.draw_connect('RShoulder', 'RElbow', "#00ff00")
                self.draw_connect('RElbow', 'RWrist', "#00ff00")
                self.draw_connect("Thorax", 'LShoulder', "#00ff00")
                self.draw_connect('LShoulder', 'LElbow', "#00ff00")
                self.draw_connect('LElbow', 'LWrist', "#00ff00")
                self.draw_connect('Thorax', 'Spine', "#00ff00")
                self.draw_connect('Spine', 'Hip', "#00ff00")
                self.draw_connect('Hip', 'LHip', "#ff0000")
                self.draw_connect('Hip', 'RHip', "#ff0000")
                self.draw_connect('RHip', 'RKnee', "#ff0000")
                self.draw_connect('RKnee', 'RFoot', "#ff0000")
                self.draw_connect('LHip', 'LKnee', "#ff0000")
                self.draw_connect('LKnee', 'LFoot', "#ff0000")

    def Openposetoh362D(self, points, data_mean_2d, data_std_2d):
        inputs = np.zeros(32)
        for i in range(16):
            if self.convert_open[i] == -1:
                continue
            inputs[i * 2 + 0] = points[self.convert_open[i] * 2 + 0]
            inputs[i * 2 + 1] = points[self.convert_open[i] * 2 + 1]

        inputs[0 * 2 + 0] = (points[11 * 2 + 0] + points[8 * 2 + 0]) / 2
        inputs[0 * 2 + 1] = (points[11 * 2 + 1] + points[8 * 2 + 1]) / 2
        inputs[7 * 2 + 0] = (points[5 * 2 + 0] + points[2 * 2 + 0]) / 2
        inputs[7 * 2 + 1] = (points[5 * 2 + 1] + points[2 * 2 + 1]) / 2

        for i in range(16):
            j = self.h36m_2d_mean[i]
            inputs[i * 2 + 0] = (inputs[i * 2 + 0] - data_mean_2d[j * 2 + 0]) / data_std_2d[j * 2 + 0]
            inputs[i * 2 + 1] = (inputs[i * 2 + 1] - data_mean_2d[j * 2 + 1]) / data_std_2d[j * 2 + 1]
        return inputs


    def DoTheInference(self, openpose_keypoints):

        if not isinstance(openpose_keypoints, np.array):
            openpose_keypoints = np.array(openpose_keypoints)
        keypoints = convert_18(openpose_keypoints)
        keypoints = keypoints.copy()
        transformed = self.Openposetoh362D(keypoints, np.mean(keypoints, axis=0), np.std(keypoints, axis=0))



    def inference_test(self, keypoints,data_std_3d, data_mean_3d, data_std_2d, data_mean_2d):
        re = np.expand_dims(keypoints, axis=0)
        result = self.model.predict(re)[0]
        print(np.mean(result, axis=0), "this is the mean")
        print(np.std(result, axis=0), "this is the std", np.std(result, axis=0).shape)
        self.plot(result, np.std(result, axis=0), np.mean(result, axis=0))
        plt.show()


    def show3D(self, Keypoints_3d):
        fig = plt.figure()
        ax = Axes3D(fig)
        show2Dpose(Keypoints_3d, ax)
        #ani = animation.FuncAnimation(fig, plot, interval=1000)
        plt.show()