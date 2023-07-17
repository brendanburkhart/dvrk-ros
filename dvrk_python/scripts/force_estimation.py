import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers
import tf2onnx
import onnxruntime as rt

print("TF version: ", tf.__version__)

def label(data):
    features = np.array(data[["jp_0", "jp_1", "jp_2", "jv_0", "jv_1", "jv_2"]])
    labels = np.array(data[["jf_0", "jf_1", "jf_2"]])

    return features, labels


train = pd.read_csv("js_train.csv", names=["jp_0", "jp_1", "jp_2", "jp_3", "jp_4", "jp_5",
                                        "jv_0", "jv_1", "jv_2", "jv_3", "jv_4", "jv_5",
                                        "jf_0", "jf_1", "jf_2", "jf_3", "jf_4", "jf_5"])


test = pd.read_csv("js_test.csv", names=["jp_0", "jp_1", "jp_2", "jp_3", "jp_4", "jp_5",
                                        "jv_0", "jv_1", "jv_2", "jv_3", "jv_4", "jv_5",
                                        "jf_0", "jf_1", "jf_2", "jf_3", "jf_4", "jf_5"])


train.head()
test.head()

train_features, train_labels = label(train)
test_features, test_labels = label(test)

train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], 1)
test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], 1)

force_estimation_model = tf.keras.Sequential([
    layers.Dense(12),
    layers.LSTM(12),
    layers.Dense(24),
    layers.Dense(3),
])

force_estimation_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

force_estimation_model.fit(train_features, train_labels, epochs=5, validation_data=(test_features, test_labels))
force_estimation_model.save(os.path.join("model", "psm_force_estimation"))

spec = (tf.TensorSpec((None, 6, 1), tf.float64, name="input"),)
output_path = "psm_force_estimation" + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(force_estimation_model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": test_features})

preds = force_estimation_model.predict(test_features)
np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-5)
