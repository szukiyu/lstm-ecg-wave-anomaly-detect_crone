B1;3409;0c""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from savitzky_golay import savitzky_golay

np.random.seed(1234)

# Hyper-parameters
sequence_length = 50
#random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1
batch_size = 50
#path_to_dataset = 'mitdbx_mitdbx_108.txt'
path_to_dataset = 'test.csv'


def dropin(X, y):
    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:",y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0,20)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    #print(np.Aasarray(X_hat), np.asarray(X_hat).shape)
    #print(np.asarray(y_hat),np.asarray(y_hat).shape)
    return np.asarray(X_hat), np.asarray(y_hat)


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean

def get_split_prep_data(train_start, train_end,
                          test_start, test_end):
    data = np.loadtxt(path_to_dataset,delimiter=',')
    label = data[:, 3]
    #data = savitzky_golay(data[:, 1], 11, 3) # smoothed version
    data = (data[:, 1]) # not smoothed version

    print("Length of Data", len(data))
    print(data,data.shape)
    # train data
    print "Creating train data..."

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print "Mean of train data : ", result_mean
    print "Train data shape  : ", result.shape

    train = result[train_start:train_end, :]
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = dropin(X_train, y_train)

    # test data
    print "Creating test data..."

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print "Mean of test data : ", result_mean
    print "Test data shape  : ", result.shape

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    label_test = label[test_start:test_end]
    print("label_test=",label_test)
    return X_train, y_train, X_test, y_test, label_test


def build_model():
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
            input_length=sequence_length - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model


def run_network(model=None, data=None):
    test_start = 37000
    test_stop = 39000
    global_start_time = time.time()
    epochs = 1
    x_range = np.arange(test_start+sequence_length,test_stop)
    if data is None:
        print 'Loading data... '
        X_train, y_train, X_test, y_test, label = get_split_prep_data(
                0, 3000, test_start, test_stop)
    else:
        X_train, y_train, X_test, y_test, label = data

    print '\nData Loaded. Compiling...\n'

    if model is None:
        model = build_model()

    try:
        print("Training")
        model.fit(
                X_train, y_train,
                batch_size=512, nb_epoch=epochs, validation_split=0.05)
        print("Predicting")
        predicted = model.predict(X_test)
        print("shape of predicted", np.shape(predicted), "size", predicted.size)
        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print("prediction exception")
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0, x_range

    try:
        plt.figure(1)
        plt.subplot(411)
        plt.title("Actual Signal w/Anomalies")
        #plt.plot(x_range,y_test[:len(y_test)], 'b')
        plt.plot(x_range,y_test, 'b')
        plt.subplot(412)
        plt.title("Predicted Signal")
        #plt.plot(x_range,predicted[:len(y_test)], 'g')
        plt.plot(x_range,predicted, 'g')
        plt.subplot(413)
        plt.title("Squared Error")
        mse = ((y_test - predicted) ** 2)
        #plt.plot(x_range,mse, 'r')
        plt.plot(x_range,mse, 'r')
        plt.subplot(414)
        plt.title("label")
        #plt.plot(x_range,label[:len(label)], 'g')
        plt.plot(x_range,label[:len(x_range)], 'g')
        plt.ylim([-1,3])
        plt.show()
    except Exception as e:
        print("plotting exception")
        print str(e)
    print 'Training duration (s) : ', time.time() - global_start_time

    return model, y_test, predicted

run_network()
