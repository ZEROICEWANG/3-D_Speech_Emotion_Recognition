import librosa
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

num_step = 3


def wav2img():
    scaler = StandardScaler()
    path = './data'
    names = os.listdir(path)
    dic_ = {}
    for i in range(6):
        dic_[i] = []
    for name in names:
        types = os.listdir(os.path.join(path, name))
        for i, type_ in enumerate(types):
            files = os.listdir(os.path.join(path, name, type_))
            for file in files:
                X, sampling_rate = librosa.load(os.path.join(path, name, type_, file))
                win_length = int(0.025 * sampling_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
                hop_length = int(0.010 * sampling_rate)  # Window shift  10ms
                melspec = librosa.feature.melspectrogram(X, sampling_rate, n_fft=win_length, hop_length=hop_length,
                                                         n_mels=40)
                # convert to log scale
                logmelspec = librosa.power_to_db(melspec)
                delta = deltas(logmelspec, 2)
                delta2 = deltas(delta, 2)
                logmelspec = cv.resize(logmelspec, (40, 300))
                delta = cv.resize(delta, (40, 300))
                delta2 = cv.resize(delta2, (40, 300))
                logmelspec = scaler.fit_transform(logmelspec)
                '''mean = np.mean(logmelspec, axis=1).reshape((logmelspec.shape[0], 1))
                std2 = np.std(logmelspec, axis=1).reshape((logmelspec.shape[0], 1)) ** 2
                logmelspec = (logmelspec - mean) / std2'''
                delta = scaler.fit_transform(delta)
                delta2 = scaler.fit_transform(delta2)
                feature = np.zeros((delta.shape[0], delta.shape[1], 3), dtype=np.float)
                feature[:, :, 0] = logmelspec
                feature[:, :, 1] = delta
                feature[:, :, 2] = delta2
                dic_[i].append(feature)
    for i in range(6):
        arrays = np.array(dic_[i])
        np.savez('./feature/%d.npz' % i, arrays)


def wav2img1():
    scaler = StandardScaler()
    label_path = './data1/label.txt'
    wav_path = './data1/sentences'
    with open(label_path, 'r') as f:
        labels = f.readlines()
    label_dic = {}
    key_list = []
    for label in labels:
        label_list = label.strip('\n').split(' ')
        label_dic[label_list[0]] = label_list[1]
        key_list.append(label_list[0])
    files = os.listdir(wav_path)
    types = ['fru', 'neu', 'ang', 'sad', 'exc']
    dic_ = {}
    for type_ in types:
        dic_[type_] = []
    for file in files:
        key = file.split('.')[0]
        if label_dic[key] in types:
            X, sampling_rate = librosa.load(os.path.join(wav_path, file))
            win_length = int(0.025 * sampling_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
            hop_length = int(0.010 * sampling_rate)  # Window shift  10ms
            melspec = librosa.feature.melspectrogram(X, sampling_rate, n_fft=win_length, hop_length=hop_length,
                                                     n_mels=40)
            # convert to log scale
            logmelspec = librosa.power_to_db(melspec)
            logmelspec = cv.resize(logmelspec, (40, 300))

            delta = deltas(logmelspec, 2)
            delta2 = deltas(delta, 2)
            logmelspec = scaler.fit_transform(logmelspec)
            '''mean = np.mean(logmelspec, axis=1).reshape((logmelspec.shape[0], 1))
            std2 = np.std(logmelspec, axis=1).reshape((logmelspec.shape[0], 1)) ** 2
            logmelspec = (logmelspec - mean) / std2'''
            delta = scaler.fit_transform(delta)
            delta2 = scaler.fit_transform(delta2)
            feature = np.zeros((delta.shape[0], delta.shape[1], 3), dtype=np.float)
            feature[:, :, 0] = logmelspec
            feature[:, :, 1] = delta
            feature[:, :, 2] = delta2
            dic_[label_dic[key]].append(feature)
            print(file, label_dic[key])

    for i, label in enumerate(types):
        arrays = np.array(dic_[label])
        np.savez('./feature1/%d.npz' % i, arrays)


def deltas(img, n):
    delta = np.zeros(img.shape)
    img = np.vstack((np.zeros((n, img.shape[1])), img, np.zeros((n, img.shape[1]))))
    sums = sum([i ** 2 for i in range(1, n + 1)]) * 2
    for i in range(n, img.shape[0] - n):
        for j in range(n):
            delta[i - n, :] += j * (img[i + j, :] - img[i - j, :])
        delta[i - n, :] /= sums
    return delta


def load_feature(test_rate):
    path = './feature'
    files = os.listdir(path)
    data = []
    step = 224
    for tag, file in enumerate(files):
        Data = np.load(os.path.join(path, file))
        feature = Data['arr_0']
        for i in range(feature.shape[0]):
            data.append([feature[i], tag])
    random.shuffle(data)
    num = len(data)
    feature = []
    label = np.zeros((num, len(files)))
    for i in range(num):
        feature.append(data[i][0])
        label[i, data[i][1]] = 1
    feature = np.array(feature)
    feature = feature.reshape((feature.shape[0], feature.shape[1], feature.shape[2], 3))
    train_size = int(num * (1 - test_rate))
    train_feature = feature[:train_size]
    train_label = label[:train_size]
    test_feature = feature[train_size:]
    test_label = label[train_size:]
    return train_feature, train_label, test_feature, test_label

# wav2img()
# load_feature(test_rate=0.3)
