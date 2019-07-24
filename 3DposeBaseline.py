import tensorflow as tf
from tensorflow.python import keras
import data_utils
import cameras
import numpy as np
from Inference import HumanPose


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


print(len(train_set_2d))
print("done reading and normalizing data.")




# dataset available

train_encoder_inputs, train_decoder_outputs = data_utils.get_all_batches(train_set_2d, train_set_3d, camera_frame, training=True)
test_encoder_inputs, test_decoder_outputs = data_utils.get_all_batches(test_set_2d, test_set_3d, camera_frame, training=True)

print(train_encoder_inputs[0])
print(len(train_encoder_inputs[0]))
print(len(train_encoder_inputs))



LinearDense = 1024
Dropout = 0.5
OuputSize = 48


def build_model(input):
    """

    :param input:
    :return: a model 
    """

    x1 = keras.layers.Dense(LinearDense)(input)
    x = keras.layers.BatchNormalization()(x1)
    x = keras.layers.Activation(tf.nn.relu)(x)
    x = keras.layers.Dropout(Dropout)(x)

    x = keras.layers.Dense(LinearDense)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.nn.relu)(x)
    x = keras.layers.Dropout(Dropout)(x)


    x = keras.layers.Add()([x1, x])


    x2 = keras.layers.Dense(LinearDense)(x)
    x0 = keras.layers.BatchNormalization()(x2)
    x0 = keras.layers.Activation(tf.nn.relu)(x0)
    x0 = keras.layers.Dropout(Dropout)(x0)

    x0 = keras.layers.Dense(LinearDense)(x0)
    x0 = keras.layers.BatchNormalization()(x0)
    x0 = keras.layers.Activation(tf.nn.relu)(x0)
    x0 = keras.layers.Dropout(Dropout)(x0)

    x0 = keras.layers.Add()([x2, x0])


    prediction = keras.layers.Dense(OuputSize)(x0)


    return keras.Model(input, prediction)



model_input = keras.layers.Input(shape=(32, ))
model = build_model(model_input)
sgd = keras.optimizers.Adam(1e-4)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
#model.fit(np.array(train_encoder_inputs), np.array(train_decoder_outputs), epochs=4, batch_size=128, validation_data=(np.array(test_encoder_inputs), np.array(test_decoder_outputs)))
#model.save(SAVE_PATH)

humanpose = HumanPose()
print(data_mean_3d[0], " this is the main area")
print(data_mean_2d[0], " this is the main area")
humanpose.inference_test(train_encoder_inputs[0], data_std_3d[0], data_mean_3d[0], data_std_2d[0], data_mean_2d[0])