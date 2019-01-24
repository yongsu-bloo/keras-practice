from __future__ import print_function
import time, os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
# from tensorflow import set_random_seed
# set_random_seed(1)
np.random.seed(999)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# dataset info
batch_size = 128
epochs = 20
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)
# x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
x_test /= 255
noise = np.random.normal(0., 0.01, (10000, 784))
x_test += noise
x_test = np.clip(x_test, 0., 1.)

lambdas = {'base':(0.,0.), 'l1':(2e-06, 0.), 'l2':(0., 2e-06)}
layer_sizes = [(300,100), (500,200)]

for layer1_size, layer2_size in layer_sizes:
#     for reg_type in lambdas:
    loss_dict = {'base':0., 'l1':0., 'l2':0.}
    acc_dict = {'base':0., 'l1':0., 'l2':0.}
    time_dict = {'base':0., 'l1':0., 'l2':0.}


    # PATH info
    PATH = './checkpoints/{}-{}-relu/'.format(layer1_size, layer2_size)
    dirs = os.listdir(PATH)
    for filename in dirs:
        if filename.endswith("ckpt"):
            if "l1=2e-06-l2=0.0" in filename:
                #L1 case
                reg_type = "l1"
            elif "l1=0.0-l2=2e-06" in filename:
                #L2 case
                reg_type = "l2"
            else:
                # base case
                reg_type = "base"

            lambda1, lambda2 = lambdas[reg_type]

            model = Sequential()
            model.add(Dense(layer1_size, input_shape=(784,),
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # model.add(Activation('relu'))
            model.add(Dense(layer2_size,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # model.add(Activation('relu'))
            model.add(Dense(num_classes, activation='softmax',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            model.load_weights(PATH + filename)
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(),
                          metrics=['accuracy'])
            # model.summary()
            before_time = time.time()
            score = model.evaluate(x_test, y_test, verbose=0)
            model.predict(x_test, batch_size=batch_size, verbose=0, steps=None)
            time_dict[reg_type] += time.time() - before_time if time_dict[reg_type] != 0. else 1
            loss, accuracy = score
            loss_dict[reg_type] += loss
            acc_dict[reg_type] += accuracy
            # print(time_dict[reg_type])

            keras.backend.clear_session()
    for reg_type in ["base","l1","l2"]:
        time_dict[reg_type] /= 10 #= (time_dict[reg_type] - 1) / 9
        loss_dict[reg_type] /= 10
        acc_dict[reg_type] /= 10

    print("-"*20)
    print(layer1_size, layer2_size)
    print("Test time", time_dict)
    print("Test Loss", loss_dict)
    print("Test Accuracy", acc_dict)
    print("-"*20)
