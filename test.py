import numpy as np
from sklearn import datasets, linear_model, metrics

import random
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE, SelectKBest
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import matplotlib.pyplot as plt
import os


def files_abspath():
    return ['data/all_data/' + path for path in os.listdir('data/all_data')]


def format_data(file_path):
    data = [line.replace('\n', '').split() for line in open(file_path) if not line.startswith('666')]
    index = []
    i = 0
    for line in open(file_path):

        if line.startswith('6666'):
            index.append(i - 1)
            continue
        i += 1
    for i in range(len(data)):
        if i in index:
            data[i].append(data[i][2])
            data[i].append(data[i][3])
            continue
        if i < len(data) - 1:
            data[i].append(data[i + 1][2])
            data[i].append(data[i + 1][3])
    data[-1].append(data[-1][2])
    data[-1].append(data[-1][3])
    # doesn't work , numpy is suck
    # data2 = [d[1:] for d in data]
    return np.array([[d[1], d[2], d[3], d[4], d[5], d[6], d[7]] for d in data], dtype=np.int)


def train_data():
    paths = files_abspath()
    data = []
    for p in paths:
        data.append(format_data(p))
    data_np = data[0]
    for i in range(1, len(data)):
        data_np = np.vstack((data_np, data[i]))
    return data_np


def line_res():
    data = train_data()
    x = data[:, :5]
    y = data[:, 5:7]
    ss = preprocessing.MinMaxScaler()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.002)
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    # rfe = RFE(linreg)
    # rfe.fit(X_train, y_train)
    #
    # for i in range(len(feature)-1):
    #     print('%d:%d' % (feature[i], rfe.ranking_[i]))
    # predicted = cross_val_predict(linreg, X_test, y_test, cv=10)
    # print(linreg.coef_)
    # print(linreg.intercept_)
    # print(linreg.score(X_test, y_test))
    predicted = lin_reg.predict(x_test)

    test_y = y_test
    predicted = predicted

    x = np.arange(len(np.arange(test_y.shape[0])))
    width = 0.4
    # plt.scatter(x,test_y[:, :1])
    plt.bar(x, test_y[:, 1:2], width=width, label='truth')
    plt.bar(x + width, predicted[:, 1:2], width=width, label='predicted')
    # plt.bar(x + width * 2, test_y[:, 1:2], width=width, label='predicted')
    # plt.bar(x + width * 3, predicted[:, 1:2], width=width, label='predicted')
    plt.xlabel("nums")
    plt.ylabel('box office')
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.show()
    # plt_fig(test_y, predicted)


def plt_fig(truth, predict):
    plt.scatter(truth[:, :1], truth[:, 1:2])
    plt.scatter(predict[:, :1], predict[:, 1:2])
    plt.show()


def test_data():
    pass


def neural_network():
    np.random.seed(1337)  # for reproducibility

    batch_size = 5
    nb_epoch = 20

    data = train_data()

    x = data[:, :5]
    y = data[:, 5:7]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.02)

    model = Sequential()
    model.add(Dense(16, input_shape=(5,)))
    model.add(Dense(32))
    model.add(Dense(2))

    model.compile(loss='mape', optimizer='adam')

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
              validation_data=(x_test, y_test))
    predict = model.predict(x_test)
    width = 0.4
    x = np.arange(len(np.arange(y_test.shape[0])))
    plt.bar(x, y_test[:, 1:2], width=width, label='truth')
    plt.bar(x + width, predict[:, 1:2], width=width, label='predicted')
    plt.show()
    # from keras.utils.vis_utils import plot_model
    # plot_model(model,show_shapes=True)


if __name__ == '__main__':
    line_res()
