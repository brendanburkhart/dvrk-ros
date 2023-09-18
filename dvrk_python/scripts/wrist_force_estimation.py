import argparse
import numpy as np
import os
import sys

import lsm_training as tr

import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization

in_joints = 3
out_joints = 3
jp_names = ["jp_" + str(3 + i) for i in range(in_joints)]
jv_names = ["jv_" + str(3 + i) for i in range(in_joints)]
jf_names = ["jf_" + str(3 + i) for i in range(out_joints)]

def split_features_labels(data):
    features = np.array(data[jp_names + jv_names])
    labels = np.array(data[jf_names])

    return features, labels

def prepare_dataset(file_name):
    data_columns = jp_names + jv_names + jf_names
    data = tr.read_csv(file_name, data_columns)

    features, labels = split_features_labels(data)
    features = features.reshape(features.shape[0], features.shape[1], 1)

    return features, labels

def prepare_data(arm_name):
    train_1 = prepare_dataset(f"data/wrist/{arm_name}_js_train.csv")
    test_1 = prepare_dataset(f"data/wrist/{arm_name}_js_test.csv")

    train = tr.split_sequences(train_1, 200)
    test = tr.split_sequences(test_1, 200)

    return train, test

def construct_model(train_data):
    train_features, train_labels = train_data

    print("    Input/output normalization...")

    input_normalization = Normalization(axis=1, input_shape=(None, 2*in_joints))
    input_normalization.adapt(train_features)

    label_normalization = Normalization(axis=2, input_shape=(None, None, out_joints))
    label_normalization.adapt(train_labels)

    train_labels = label_normalization(train_labels).numpy()

    regularizer = keras.regularizers.L1L2(l1=2e-4, l2=2e-3)
    model = keras.Sequential([
        input_normalization,
        keras.layers.LSTM(48, kernel_regularizer=regularizer, dropout=0.2),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        keras.layers.Dense(out_joints, activation=None),
    ])
    
    output_shape = (None, None, out_joints)
    return model, label_normalization, output_shape

def train_model(train_data, test_data, epochs=5):
    model, label_normalization, output_shape = construct_model(train_data)


    test_features, test_labels = test_data
    test_labels = label_normalization(test_labels).numpy()
    test_data = (test_features, test_labels)

    train_features, train_labels = train_data
    train_labels = label_normalization(train_labels).numpy()
    train_data = (train_features, train_labels)

    print("    Beginning training...")
    model = tr.train_model(model, train_data, test_data, epochs=epochs)
    model = tr.add_output_denormalization(label_normalization, output_shape, model)

    return model

def main(arm_name):
    train_data, test_data = prepare_data(arm_name)
    model = train_model(train_data, test_data, epochs=3)
    model.save(os.path.join("models", arm_name + "_wrist"))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True, help = 'arm name')
    args = parser.parse_args(sys.argv[1:])

    main(args.arm)
