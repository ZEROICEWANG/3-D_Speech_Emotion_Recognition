from Data import load_feature
from model import build, save_model, load_model
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6



def train(feature, label):
    model = build(label.shape[1])
    model.fit(x=feature, y=label, batch_size=40, epochs=20)
    save_model(model)


def predict(feature, label):
    model = build(label.shape[1])
    model.load_weights('./model/model.h5')
    result = model.predict(feature)
    result = np.argmax(result, axis=1)
    label = np.argmax(label, axis=1)
    rate = sum([int(result[i] == label[i]) for i in range(len(label))]) / len(label)
    mat = np.zeros((6, 6))
    for i in range(len(label)):
        mat[label[i], result[i]] += 1
    sums = np.sum(mat, axis=1)
    mat = mat / sums
    print(mat)
    print(rate)


if __name__ == '__main__':
    train_feature, train_label, test_feature, test_label = load_feature(0.2)
    train(train_feature, train_label)
    predict(test_feature, test_label)
