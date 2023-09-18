import argparse
import numpy as np
import os
import pandas as pd
import sys

# silence excessive TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization


in_joints = 3
out_joints = 3
jp_names = ["jp_" + str(i) for i in range(in_joints)]
jv_names = ["jv_" + str(i) for i in range(in_joints)]
jf_names = ["jf_" + str(i) for i in range(out_joints)]

def split_features_labels(data):
    features = np.array(data[jp_names + jv_names])
    labels = np.array(data[jf_names])

    return features, labels

def prepare_dataset(file_name):
    data_columns = jp_names + jv_names + jf_names
    dtypes = {name: np.float32 for name in data_columns}
    data = pd.read_csv(file_name, dtype=dtypes)
    data.head()

    features, labels = split_features_labels(data)

    return features, labels

def prepare_data(arm_name):
    train_features = []
    train_labels = []

    train_1 = prepare_dataset(f"data/new_{arm_name}_js_train3.csv")
    train_2 = prepare_dataset(f"data/new_{arm_name}_js_train4.csv")
    train_3 = prepare_dataset(f"data/new_{arm_name}_js_train5.csv")
    train_4 = prepare_dataset(f"data/new_{arm_name}_js_train4.csv")

    test_1 = prepare_dataset(f"data/new_{arm_name}_js_test3.csv")

    train_1 = split_sequences(train_1, 200)
    train_2 = split_sequences(train_2, 200)
    train_3 = split_sequences(train_3, 200)
    train_4 = split_sequences(train_4, 200)

    train = (np.concatenate((train_1[0], train_2[0], train_3[0], train_4[0])), np.concatenate((train_1[1], train_2[1], train_3[1], train_4[1])))
    #test = (np.concatenate((test_1[0], test_2[0])), np.concatenate((test_1[1], test_2[1])))

    test_1 = split_sequences(test_1, 200)
    return train, test_1

def split_sequences(dataset, sequence_length):
    features, labels = dataset

    batches = features.shape[0] // sequence_length
    length = sequence_length * batches

    features = features[0:length, :]
    labels = labels[0:length, :]

    features = features.reshape(batches, sequence_length, features.shape[1])
    labels = labels.reshape(batches, sequence_length, labels.shape[1])

    return features, labels

def train_model(train_data, test_data, epochs=5):
    regularizer = regularizers.L1L2(l1=2e-4, l2=2e-3)

    train_features, train_labels = train_data
    test_features, test_labels = test_data

    print("    Input/output normalization...")

    input_normalization = Normalization(axis=1, input_shape=(None, 2*in_joints))
    input_normalization.adapt(train_features)

    label_normalization = Normalization(axis=2, input_shape=(None, None, out_joints))
    label_normalization.adapt(train_labels)

    train_labels = label_normalization(train_labels).numpy()
    test_labels = label_normalization(test_labels).numpy()

    model = tf.keras.Sequential([
        input_normalization,
        layers.LSTM(256, kernel_regularizer=regularizer, dropout=0.2, return_sequences=True),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(out_joints, activation=None),
    ])
    
    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005))

    print("    Beginning training...")

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        model.fit(train_features, train_labels, shuffle=True, batch_size=1, epochs=25, validation_data=(test_features, test_labels))

    output_denormalization = Normalization(axis=2, mean=-label_normalization.mean/tf.maximum(tf.sqrt(label_normalization.variance), 1e-5), variance=np.ones((3,)) / label_normalization.variance,
                                                input_shape=(None, None, out_joints))
    model.add(output_denormalization)

    return model

def main(arm_name):
    print("TF version: ", tf.__version__)

    print("Loading data...")
    train_data, test_data = prepare_data(arm_name)
    print("Initializing model training...")
    model = train_model(train_data, test_data, epochs=5)
    model.save(os.path.join("models", arm_name + "6_force_estimation"))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True, help = 'arm name')
    args = parser.parse_args(sys.argv[1:])

    main(args.arm)
