import argparse
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

in_joints = 3
jp_names = ["jp_" + str(i) for i in range(in_joints)]
jv_names = ["jv_" + str(i) for i in range(in_joints)]
jf_names = ["jf_" + str(i) for i in range(in_joints)]

def prepare_dataset(file_name, has_contact):
    data_columns = jv_names + jf_names
    dtypes = {name: np.float32 for name in data_columns}
    data = pd.read_csv(file_name, dtype=dtypes)
    data.head()

    features = np.array(data[data_columns])
    labels = np.ones((features.shape[0], 1)) if has_contact else np.zeros((features.shape[0], 1)) 

    return features, labels

def prepare_data(arm_name):
    train_1 = prepare_dataset(f"data/contact_psm2_train.csv", True)
    train_2 = prepare_dataset(f"data/no_contact_psm2_train.csv", False)
    test_1 = prepare_dataset(f"data/contact_psm2_test.csv", True)
    test_2 = prepare_dataset(f"data/no_contact_psm2_test.csv", False)

    train_1 = split_sequences(train_1, 50)
    train_2 = split_sequences(train_2, 50)
    test_1 = split_sequences(test_1, 50)
    test_2 = split_sequences(test_2, 50)

    train = (np.concatenate((train_1[0], train_2[0])), np.concatenate((train_1[1], train_2[1])))
    test = (np.concatenate((test_1[0], test_2[0])), np.concatenate((test_1[1], test_2[1])))

    return train, test

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

    input_normalization = Normalization(axis=1, input_shape=(None, 2*in_joints))
    input_normalization.adapt(train_features)

    model = tf.keras.Sequential([
        input_normalization,
        layers.LSTM(16, kernel_regularizer=regularizer, dropout=0.5),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.keras.optimizers.Adam())

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        model.fit(train_features, train_labels, shuffle=True, batch_size=1, epochs=epochs, validation_data=(test_features, test_labels))

    return model

def main(arm_name):
    print("TF version: ", tf.__version__)

    train_data, test_data = prepare_data(arm_name)
    model = train_model(train_data, test_data, epochs=5)
    model.save(os.path.join("models", arm_name + "_contact_detection"))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True, help = 'arm name')
    args = parser.parse_args(sys.argv[1:])

    main(args.arm)
