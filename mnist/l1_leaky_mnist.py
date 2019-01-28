from __future__ import print_function
import datetime, time, os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from numpy.random import seed
from tensorflow import set_random_seed

def add_hparams_to_path(path, hparams, manual_seed, time_stamp):
    for label  in hparams:
        path += '{}={}-'.format(label, hparams[label])
    path += "[{}]".format(manual_seed) + time_stamp
    return path
# dataset info
batch_size = 128
epochs = 20
num_classes = 10

# model info
layer_size = (300,100)
lambda1s = [ float("{}e-0{}".format(i,j)) for i in range(1,2) for j in range(4,7) ]
lambda1s.append(0.0)
# hyper-parameters
learning_rate = 0.0002
adamb1 = 0.5
adamb2 = 0.99


# Variables to record
loss_dict = {}
acc_dict = {}
zero_dict = {}
dead_dict = {}

time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
with open('./{}-{}-l1-leaky_result.txt'.format(layer_size[0], layer_size[1]), "w") as f:
    f.write("{}-{}-{}\n".format(layer_size[0], layer_size[1], num_classes))
    for lambda1 in lambda1s:
        losses = [] # the last test loss list
        accuracies = [] # the last test accuracy list
        zero_acts = [] # the number of zero activations (3 dimension each)
        dead_acts = [] # totally dead activations (3 dimension each)
        # hyper parameters for the path name
        hparams = {}
        hparams['l1'] = lambda1
        # hparams['size'] = "{}-{}".format(layer_size[0], layer_size[1])
        hparams['op'] = 'Adam_b1-{}_b2-{}'.format(adamb1, adamb2)
        hparams['lr'] = learning_rate
        hparams['ep'] = epochs
        # hparams['act'] = 'relu'
        for manual_seed in range(10):
            seed(manual_seed*111)
            set_random_seed(manual_seed*111)
            # PATH setting and create
            PATH = './logs/{}-{}-l1-leaky/'.format(layer_size[0], layer_size[1])
            if not os.path.exists(PATH):
                os.mkdir(PATH)
            PATH = add_hparams_to_path(PATH, hparams, manual_seed, time_stamp)
            if not os.path.exists(PATH):
                os.mkdir(PATH)            # print(x_train.shape[0], 'train samples')
            # print(x_test.shape[0], 'test samples')
                print("Directory " , PATH ,  " Created ")

            # the data, split between train and test sets
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            model = Sequential()
            model.add(Dense(layer_size[0], input_shape=(784,),
                            activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            model.add(Dense(layer_size[1],
                            activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            model.add(Dense(num_classes,
                            activation='softmax'))

            model.summary()

            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=learning_rate, beta_1=adamb1, beta_2=adamb2, epsilon=None, decay=0.0, amsgrad=False),
                          metrics=['accuracy'])

            checkpoint_path = "checkpoints/{}-{}-l1-leaky/".format(layer_size[0], layer_size[1])
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)
            checkpoint_path = add_hparams_to_path(checkpoint_path, hparams, manual_seed, time_stamp)
            checkpoint_path += ".ckpt"
            # Create checkpoint and TensorBoard saving callbacks
            cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          save_weights_only=False,
                                                          verbose=0)

            tfboard_callback  = keras.callbacks.TensorBoard(log_dir=PATH,
                                                            histogram_freq=1,
                                                            batch_size=batch_size,
                                                            write_graph=True,
                                                            write_grads=False,
                                                            write_images=False,
                                                            embeddings_freq=0,
                                                            embeddings_layer_names=None,
                                                            embeddings_metadata=None,
                                                            embeddings_data=None)
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test),
                                callbacks=[tfboard_callback, cp_callback])
            # score = [ loss, accuracy ]
            score = model.evaluate(x_test, y_test, verbose=1)
            print('\nTest loss:', score[0])
            print('Test accuracy:', score[1])

            # Save activation values from each layer
            activations_list = []

            imodel1 = Sequential()
            imodel1.add(Dense(layer_size[0], weights=model.layers[0].get_weights() , input_shape=(784,), activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            activations_list.append(imodel1.predict(x_test, batch_size=batch_size, verbose=0, steps=None))

            imodel2 = Sequential()
            imodel2.add(Dense(layer_size[0], weights=model.layers[0].get_weights() , input_shape=(784,), activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            imodel2.add(Dense(layer_size[1], weights=model.layers[2].get_weights() , activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            activations_list.append(imodel2.predict(x_test, batch_size=batch_size, verbose=0, steps=None))

            imodel3 = Sequential()
            imodel3.add(Dense(layer_size[0], weights=model.layers[0].get_weights() , input_shape=(784,), activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            imodel3.add(Dense(layer_size[1], weights=model.layers[2].get_weights() , activity_regularizer=keras.regularizers.l1(lambda1)))
            model.add(LeakyReLU())
            imodel3.add(Dense(num_classes, activation='softmax', weights=model.layers[4].get_weights()))
            activations_list.append(imodel3.predict(x_test, batch_size=batch_size, verbose=0, steps=None))

            # Print out how many 0s are in activation of each layer
            print('# Dead Activations:{:.2f}-{:.2f}-{:.2f}'.format(
                    sum(sum(activations_list[0]==0))/len(x_test),
                    sum(sum(activations_list[1]==0))/len(x_test),
                    sum(sum(activations_list[2]==0))/len(x_test) ))
            print('# Totally Dead Activations:{}-{}-{}\n'.format(
                    sum(sum(activations_list[0]==0)==len(x_test)),
                    sum(sum(activations_list[1]==0)==len(x_test)),
                    sum(sum(activations_list[2]==0)==len(x_test))))

            zero_act = [ sum(sum(activations_list[i]==0))/len(x_test) for i in range(3) ] # the average number of 0s in activations from each layer
            f.write("{} {:.4f} {:.2f}-{:.2f}-{:.2f}\n".format(lambda1, score[1], *zero_act))
            losses.append(score[0])
            accuracies.append(score[1])
            zero_acts.append(zero_act)
            dead_acts.append([ sum(sum(activations_list[i]==0)==len(x_test)) for i in range(3) ])

            print("\nModel saved in {}\n".format(PATH))
            keras.backend.clear_session()

        mean_loss = np.mean(losses)
        mean_acc = np.mean(accuracies)
        mean_zero = np.mean(zero_acts, axis=0)
        mean_dead = np.mean(dead_acts, axis=0)

        print("\nLoss: {}".format(mean_loss))
        print("Accuracy: {:.4f}".format(mean_acc))
        print("# Zero activation: {:.2f}-{:.2f}-{:.2f}".format(*mean_zero))
        print("# Dead activation: {:.2f}-{:.2f}-{:.2f}\n".format(*mean_dead))

        loss_dict[lambda1] = mean_loss
        acc_dict[lambda1] = mean_acc
        zero_dict[lambda1] = mean_zero
        dead_dict[lambda1] = mean_dead

    print(str(loss_dict))
    print(str(acc_dict))
    print(str(zero_dict))
    print(str(dead_dict))
