import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.ensemble import ExtraTreesRegressor
import random
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import os


def files_abspath():
    return ['data/all_data/' + path for path in os.listdir('data/all_data')]


def format_data(file_path):
    data = [line.replace('\n', '').split() for line in open(file_path) if not line.startswith('6666')]
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
        data.append(format_data24(p))
    data_np = data[0]
    for i in range(1, len(data)):
        data_np = np.vstack((data_np, data[i]))
    return data_np[:,:5],data_np[:,5:7]


def line_res():
    x, y = train_data()
    ss = preprocessing.MinMaxScaler()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)
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
    print(error(y_test, predicted))

    x = np.arange(len(np.arange(test_y.shape[0])))
    width = 0.4
    # plt.scatter(x,test_y[:, :1])
    # plt.bar(x, test_y[:, :1], width=width, label='truth')
    # plt.bar(x + width, predicted[:, :1], width=width, label='predicted')
    # plt.bar(x + width * 2, test_y[:, 1:2], width=width, label='predicted')
    # plt.bar(x + width * 3, predicted[:, 1:2], width=width, label='predicted')
    # plt.xlabel("nums")
    # plt.ylabel('box office')
    # plt.legend(loc="upper right")  # 显示图中的标签
    # plt.show()
    # plt_fig(test_y, predicted)


def plt_fig(truth, predict):
    plt.scatter(truth[:, :1], truth[:, 1:2])
    plt.scatter(predict[:, :1], predict[:, 1:2])
    plt.show()


def test_data():
    pass


def neural_network():
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, initializers, regularizers
    from keras.layers.normalization import BatchNormalization
    from keras import optimizers
    np.random.seed(1337)  # for reproducibility

    batch_size = 8
    nb_epoch = 20

    x,y = train_data()
    ss = preprocessing.StandardScaler()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)

    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    model = Sequential()
    model.add(Dense(32, input_shape=(5,), kernel_initializer=initializers.random_normal(stddev=0.01),
                    kernel_regularizer=regularizers.l1(0.01)
                    ))
    model.add(
        Dense(32, kernel_initializer=initializers.random_normal(stddev=0.01),
              kernel_regularizer=regularizers.l1(0.01)))
    model.add(
        Dense(16, kernel_initializer=initializers.random_normal(stddev=0.01),
              kernel_regularizer=regularizers.l1(0.01)))
    model.add(
        Dense(8, kernel_initializer=initializers.random_normal(stddev=0.01),
              kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(2))

    model.compile(loss='mape', optimizer='adam')
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
              validation_data=(x_test, y_test))
    predict = model.predict(x_test)
    print(error(y_test, predict))

    # width = 0.4
    # x = np.arange(len(np.arange(y_test.shape[0])))
    # plt.bar(x, y_test[:, 1:2], width=width, label='truth')
    # plt.bar(x + width, predict[:, 1:2], width=width, label='predicted')
    # plt.show()
    # from keras.utils.vis_utils import plot_model
    # plot_model(model,show_shapes=True)


def loss_func(y_true, y_pred):
    pass


def feature_selection():
    data = train_data()
    x = data[:, :4]
    y = data[:, 4:6]
    ss = preprocessing.StandardScaler()
    x = ss.fit_transform(x)
    print(x)
    cls = ExtraTreesRegressor()

    cls = cls.fit(x,y)
    print(cls.feature_importances_)


# cluster by Latitude and longitude , default n_clusters = 10
def kmeans(n_clusters=10):
    from sklearn.cluster import KMeans
    data = train_data()[:, 1:3]
    y_pred = KMeans(n_clusters=n_clusters).fit_predict(data)
    print(y_pred)
    plt.figure(figsize=(12, 12))
    plt.scatter(data[:, 0], data[:, 1], c=np.arange(len(y_pred)))
    plt.show()


def knn():
    from sklearn import neighbors
    x,y = train_data()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)
    knn = neighbors.KNeighborsRegressor()
    knn.fit(x_train, y_train)
    p = knn.predict(x_test)
    print(error(y_test, p))
    # for i in range(len(p)):
    #     print(y_test[i][0],p[i][0],y_test[i][1],p[i][1])

    # x = np.arange(len(np.arange(y_test.shape[0])))
    # width = 0.4
    # plt.scatter(x,test_y[:, :1])
    # plt.bar(x, y_test[:, :1], width=width, label='truth')
    # plt.bar(x + width, p[:, :1], width=width, label='predicted')
    # plt.bar(x + width * 2, test_y[:, 1:2], width=width, label='predicted')
    # plt.bar(x + width * 3, predicted[:, 1:2], width=width, label='predicted')
    # plt.xlabel("nums")
    # plt.ylabel('longitude')
    # plt.legend(loc="upper right")  # 显示图中的标签
    # plt.show()


def error(truth, predict):
    return np.mean(abs(truth - predict)[:, :1]), np.mean(abs(truth - predict)[:, 1:])


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
            data[i].append(data[i + 4][2])
            data[i].append(data[i + 4][3])
            data2.append(data[i])
            i += 1
        else:
            break
    # doesn't work , numpy is suck
    # data2 = [d[1:] for d in data]
    return np.array([[d[1], d[2], d[3], d[4], d[5], d[6], d[7]] for d in data2], dtype=np.int)


def kernal_ridge_res():
    x,y = train_data()
    ss = preprocessing.MinMaxScaler()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    krr = KernelRidge()
    krr.fit(x_train, y_train)
    pred = krr.predict(x_test)
    print(error(y_test,pred))


def ridge_res():
    import sklearn.linear_model
    x,y = train_data()
    ss = preprocessing.MinMaxScaler()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    krr = linear_model.Lasso()
    krr.fit(x_train, y_train)
    pred = krr.predict(x_test)
    print(error(y_test,pred))


if __name__ == '__main__':
    neural_network()
