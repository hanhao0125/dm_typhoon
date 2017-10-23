import numpy as np
from keras.layers import LSTM
from math import asin, sqrt
from sklearn import datasets, linear_model, metrics
from sklearn.ensemble import ExtraTreesRegressor
import random
from sklearn import model_selection
from sklearn import preprocessing
import os


def files_abspath():
    return ['data/all_data/' + path for path in os.listdir('data/all_data')]


# 返回训练集，标签
def train_data():
    paths = files_abspath()
    data = []
    for p in paths:
        data.append(format_data24(p))
    data_np = data[0]
    for i in range(1, len(data)):
        data_np = np.vstack((data_np, data[i]))
    return data_np[:, :15], data_np[:, 15:17]
    # return data_np


def neural_network():
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, initializers, regularizers
    from keras.layers.normalization import BatchNormalization
    from keras import optimizers
    np.random.seed(1337)  # for reproducibility

    batch_size = 100
    nb_epoch = 100

    x, y = train_data()
    test_data = test_data1('data/CH2014BST.txt')
    ss = preprocessing.StandardScaler()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)

    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    test_data_x = ss.transform(test_data[:, :15])
    test_data_y = test_data[:,15:17]
    model = Sequential()
    model.add(Dense(32, input_shape=(15,), kernel_initializer=initializers.random_normal(stddev=0.01), use_bias=True,
                    activation='relu'
                    ))
    model.add(
        Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01), activation='relu', use_bias=True
              ))
    model.add(
        Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01), use_bias=True
              ))
    model.add(
        Dense(32, kernel_initializer=initializers.random_normal(stddev=0.01), use_bias=True
              ))
    model.add(Dense(2, kernel_initializer=initializers.random_normal(stddev=0.01)))
    import keras.callbacks as cb

    model.compile(loss=loss_function, optimizer='nadam', metrics=[error_1, error_2])
    fuck = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                     validation_data=(x_test, y_test),
                     callbacks=[cb.ReduceLROnPlateau(monitor='val_loss')]
                     )
    predict = model.predict(test_data_x)
    print(error_1(test_data[:, 15:17], predict))
    print(error_2(test_data[:, 15:17], predict))
    print(mean_error(test_data_y, predict))


def error_1(truth, predict):
    return abs(truth - predict)[:, :1]


def error_2(truth, predict):
    return abs(truth - predict)[:, 1:]


def format_data24_1(file_path):
    data = [line.replace('\n', '').split() for line in open(file_path) if not line.startswith('6666')]
    index = []
    j = 0

    for line in open(file_path):
        if line.startswith('6666'):
            index.append(j - 1)
            continue
        j += 1
    data2 = []
    i = 0
    import netCDF4 as nc
    import datetime

    file = '/mnt/graeme/1979-2016Geopotential_500hpa.nc'

    data3 = nc.Dataset(file)

    while i < len(data):
        if i + 3 in index:
            i += 4
            continue
        if i < len(data) - 4:
            times = data[i][0]
            d2 = datetime.datetime(1900, 1, 1, 0)
            d1 = datetime.datetime(int(''.join(times[0:4])), int(''.join(times[4:6])), int(''.join(times[6:8])),
                                   int(''.join(times[8:])))
            hour = (d1 - d2).days * 24 + int(''.join(times[8:]))
            remain = hour % 6
            time_index = (hour - 692496) // 6
            if remain == 0:
                a = data3.variables['z'][time_index][int(int(data[i][2]) / 10)][int(int(data[i][3]) / 10)]
            else:
                a = 0
                print(a)

            data[i].append(a)
            data[i].append(data[i + 4][2])
            data[i].append(data[i + 4][3])
            data2.append(data[i])
            i += 1
        else:
            break
    # doesn't work , numpy is suck
    # data2 = [d[1:] for d in data]

    return np.array(
        [[d[0][:4], d[0][4:6], d[0][:6:8], d[0][8:10], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]] for d in data2 if
         int(d[5]) != 0 and d[1] != 0], dtype=np.int)


# 返回有标签数据，24小时之后
def format_data24(file_path):
    data = [line.replace('\n', '').split() for line in open(file_path) if not line.startswith('6666')]
    index = []
    j = 0

    for line in open(file_path):
        if line.startswith('6666'):
            index.append(j - 1)
            continue
        j += 1
    data2 = []
    i = 0

    while i < len(data):
        if i + 3 in index:
            i += 4
            continue
        if i < len(data) - 4:
            if i in index:
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(data[i][2])
                data[i].append(data[i][3])
            else:
                data[i].append(abs(int(data[i][2]) - int(data[i - 1][2])))
                data[i].append(abs(int(data[i][3]) - int(data[i - 1][3])))
                data[i].append(abs(int(data[i][4]) - int(data[i - 1][4])))
                data[i].append(abs(int(data[i][5]) - int(data[i - 1][5])))
                data[i].append((data[i - 1][2]))
                data[i].append((data[i - 1][3]))

            data[i].append(data[i + 4][2])
            data[i].append(data[i + 4][3])
            data2.append(data[i])
            i += 1
        else:
            break
    # doesn't work , numpy is suck
    # data2 = [d[1:] for d in data]

    return np.array(
        [[d[0][:4], d[0][4:6], d[0][:6:8], d[0][8:10], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10],
          d[11], d[12]] for d in data2 if int(d[5]) != 0 and d[1] != 0],
        dtype=np.int)


def save_data():
    x = train_data()
    np.savetxt('data.csv', x, delimiter=',')


def train_data2():
    data = np.loadtxt('data.csv', delimiter=",", skiprows=0)
    return data[:, :6], data[:, 6:8]


def mean_error(truth,predict):
    return np.mean(abs(truth - predict)[:, :1]),np.mean(abs(truth - predict)[:, 1:])


def loss_function(truth, predict):
    return 5 * abs(truth - predict)[:, :1] + abs(truth - predict)[:, 1:]


def test_data1(file_path):
    data = [line.replace('\n', '').split() for line in open(file_path) if not line.startswith('6666')]
    index = []
    j = 0

    for line in open(file_path):
        if line.startswith('6666'):
            index.append(j - 1)
            continue
        j += 1
    data2 = []
    i = 0

    while i < len(data):
        if i + 3 in index:
            i += 4
            continue
        if i < len(data) - 4:
            if i in index:
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(abs(int(data[i][2]) - int(data[i + 1][2])))
                data[i].append(data[i][2])
                data[i].append(data[i][3])
            else:
                data[i].append(abs(int(data[i][2]) - int(data[i - 1][2])))
                data[i].append(abs(int(data[i][3]) - int(data[i - 1][3])))
                data[i].append(abs(int(data[i][4]) - int(data[i - 1][4])))
                data[i].append(abs(int(data[i][5]) - int(data[i - 1][5])))
                data[i].append((data[i - 1][2]))
                data[i].append((data[i - 1][3]))

            data[i].append(data[i + 4][2])
            data[i].append(data[i + 4][3])
            data2.append(data[i])
            i += 1
        else:
            break
    # doesn't work , numpy is suck
    # data2 = [d[1:] for d in data]

    return np.array(
        [[d[0][:4], d[0][4:6], d[0][:6:8], d[0][8:10], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10],
          d[11], d[12]] for d in data2 if int(d[5]) != 0 and d[1] != 0],
        dtype=np.int)


if __name__ == '__main__':
    neural_network()





def feature_selection():
    x, y = train_data()

    ss = preprocessing.StandardScaler()
    x = ss.fit_transform(x)
    cls = ExtraTreesRegressor()

    cls = cls.fit(x, y)
    print(cls.feature_importances_)


# 距离 loss ，不工作
def haversine2(truth, predict):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """
    # 将十进制度数转化为弧度
    predict = np.radians(predict)
    truth = np.radians(truth)

    dlon = predict[:, 1:] - truth[:, 1:]
    dlat = predict[:, :1] - truth[:, :1]
    a = np.sin(dlat / 2) ** 2 + np.cos(truth[:, :1]) * np.cos(predict[:, :1]) * np.sin(dlon / 2) ** 2
    return list(map(lambda x: 2 * 6371 * asin(sqrt(x)), a))
    # return abs(truth - predict)[:, :1] + abs(truth - predict)[:, 1:]
