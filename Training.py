import tensorflow as tf
from tensorflow.python import keras
from Helpers import data_utils, cameras, HumanDataset
import numpy as np
import time
import math


SAVE_PATH = "./pose.h5"
dataset_path_3d ='./data/data_3d_h36m.npz'
dataset_path_2d = './data/data_2d_h36m_gt.npz'
TRAIN_SUBJECTS = ['S1','S5','S6','S7','S8']
VALIDATION_SUBJECTS = ['S9','S11']

print("loading dataset")
humandataset = HumanDataset.HumanDataset(dataset_path_3d)
print("3d dataset loaded")

print("preparing data")
for subject in humandataset.subjects():
    for action in humandataset[subject].keys():
        anim = humandataset[subject][action]
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = anim['positions'] #cameras.world_to_camera(anim['positions'], R=cam['orientation'], T=cam['translation'])
                #pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d


print('Loading 2D detections...')
keypoints = np.load(dataset_path_2d, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(humandataset.skeleton().joints_left()), list(humandataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()
print("2d detection loaded")


print("check if 3d length equal to 2d length dataset")
for subject in humandataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in humandataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action, subject)
        if 'positions_3d' not in humandataset[subject][action]:
            continue
        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = humandataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
        assert len(keypoints[subject][action]) == len(humandataset[subject][action]['positions_3d'])
print("done checking the length")

print("normalize 2d keypoints")
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = humandataset.cameras()[subject][cam_idx]
            kps[..., :2] = cameras.normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps
print("done normalising")


# fetch the data
def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
            if subject in humandataset.cameras():
                cams = humandataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in humandataset[subject][action]:
                poses_3d = humandataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = 1
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = cameras.deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    return out_camera_params, out_poses_3d, out_poses_2d


cameras_train, poses_train, poses_train_2d = fetch(TRAIN_SUBJECTS, None)
train_encoder_inputs, train_decoder_outputs = data_utils.build_encoder(poses_train_2d, poses_train)
cameras_validation, poses_validation, poses_validation_2d = fetch(VALIDATION_SUBJECTS, None)
test_encoder_inputs, test_decoder_outputs = data_utils.build_encoder(poses_validation_2d, poses_validation)


# losses
def mpjpe(y_true, y_pred):
    """
    mean per -joint position error(i.e mean eclidean distance)
    often referred to as "Protocol #1 in many paper"
    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.reduce_mean(tf.norm(y_pred - y_true, axis=len(y_true.shape) -1, ord='euclidean'))



def weight_mpjpe(y_true, y_pred, weight):
    """
    mean per -joint position error(i.e mean eclidean distance)
    often referred to as "Protocol #1 in many paper"
    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.reduce_mean(weight * tf.norm(y_pred - y_true, axis=len(y_true.shape) -1, ord='euclidean'))




#model
def new_model(input, momentum=0.1, dropout=0.1, output_size=17):
    x = keras.layers.Conv1D(64, kernel_size=9,  use_bias=False)(input)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.Conv1D(128, kernel_size=3, use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.Conv1D(128, kernel_size=3,  use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.Conv1D(256, kernel_size=3,  use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.Conv1D(512, kernel_size=2,  use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.Conv1D(512, kernel_size=2,  use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.Conv1D(684, kernel_size=1,  use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=momentum)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    prediction = keras.layers.Conv1D(3 * output_size, kernel_size=1)(x)
    return keras.Model(input, prediction)





model_input = keras.layers.Input(shape=(17, 2))
model = new_model(model_input)
model.summary()

def learning_rate_schedelur(epoch):
    initial_lrate = 1e-3
    drop = 0.1
    epoch_drops = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epoch_drops))
    return lrate

#tensorflow callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-3, cooldown=1)
schudle_lr = keras.callbacks.LearningRateScheduler(learning_rate_schedelur)
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time.time()))


# optimizer
adam_optimmizer = keras.optimizers.Adam(lr=1e-3)



# model compile
#from Inference import HumanPose
#tree_d_model = HumanPose(predict_14=False, visualise=True)
#tree_d_model.direct_show_3d(train_decoder_outputs[0])
model.compile(loss=mpjpe, optimizer=adam_optimmizer, metrics=["accuracy"])
model.fit(train_encoder_inputs, train_decoder_outputs,  epochs=10,  batch_size=128, validation_data=(test_encoder_inputs, test_decoder_outputs), use_multiprocessing=True)
model.save(SAVE_PATH)