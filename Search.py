import tensorflow as tf
from tensorflow.python import keras
from Helpers import data_utils, cameras
import numpy as np
import time
from kerastuner.tuners import RandomSearch


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



def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Range('num_layers', 2, 15)):
        model.add(keras.layers.Conv1D(hp.Range('units_conv' + str(i), min_value=32, max_value=512, step=32),  padding="same", strides=1, kernel_size=1, input_shape=(1, 32)))
        model.add(keras.layers.Dense(48))
        model.add(keras.layers.Activation(tf.nn.relu))
        model.add(keras.layers.MaxPool1D(pool_size=1))




    model.add(keras.layers.Conv1D(OuputSize, kernel_size=1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[3e-3, 3e-4, 3e-5])), metrics=['accuracy'])

    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=4,
    directory='my_dir',
    project_name='Human3D')

tuner.search_space_summary()
tuner.search(np.expand_dims(train_encoder_inputs, axis=1), np.expand_dims(train_decoder_outputs, axis=1), epochs=10, batch_size=128, validation_data=(np.expand_dims(test_encoder_inputs, axis=1), np.expand_dims(test_decoder_outputs, axis=1)))
models = tuner.get_best_models(num_models=1)

