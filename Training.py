import tensorflow as tf
from tensorflow.python import keras
from Helpers import data_utils, cameras
import numpy as np


# link to download https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip

actions = data_utils.define_actions("All")
number_of_actions = len(actions)
SAVE_PATH = "./pose.h5"
# Load camera parameters
SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
rcams = cameras.load_cameras("./h36m/cameras.h5", SUBJECT_IDS)
camera_frame = False #weither to convert 3D pose to camera coordinate
predict_14 = False #"predict 14 joints"
use_sh = False # "Use 2d pose predictions from StackedHourglass"
data_dir = "./h36m/" # data directory

# Load 3d data and load (or create) 2d projections
train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, data_dir, camera_frame, rcams, predict_14)



# Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
if use_sh:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, data_dir)
else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data(
        actions, data_dir, rcams)




# dataset available

train_encoder_inputs, train_decoder_outputs = data_utils.get_all_batches(train_set_2d, train_set_3d, camera_frame, training=True)
test_encoder_inputs, test_decoder_outputs = data_utils.get_all_batches(test_set_2d, test_set_3d, camera_frame, training=True)





linear_size = 1024
dropout_rate = 0.5
OuputSize = 48



def build_model(input):
    """
    :param input:
    :return: a model 
    """
    x = keras.layers.Reshape((1, input.shape[-2], input.shape[-1]))(input)
    x = keras.layers.Conv2D(64, kernel_size=1, padding="same",  activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((1, 1))(x)

    x = keras.layers.Conv2D(64, kernel_size=1, padding="same", activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((1, 1))(x)


    x = keras.layers.Conv2D(64, kernel_size=1,  padding="same",  activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((1, 1))(x)

    x = keras.layers.Reshape((x.shape[-2], x.shape[-1]))(x)
    x = keras.layers.Dense(linear_size, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # output
    prediction = keras.layers.Dense(OuputSize)(x)
    return keras.Model(input, prediction)



model_input = keras.layers.Input(shape=(1, 32))
model = build_model(model_input)
model.summary()
adam_optimmizer = keras.optimizers.Adam(lr=3e-4)
model.compile(loss='mean_squared_error', optimizer=adam_optimmizer, metrics=['accuracy'])
model.fit(np.expand_dims(train_encoder_inputs, axis=1), np.expand_dims(train_decoder_outputs, axis=1), epochs=100, batch_size=128, validation_data=(np.expand_dims(test_encoder_inputs, axis=1), np.expand_dims(test_decoder_outputs, axis=1)))
model.save(SAVE_PATH)

