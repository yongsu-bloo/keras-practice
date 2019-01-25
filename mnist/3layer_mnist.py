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

def schedule(epoch):
    lr = 0.0002
    if epoch < 10:
        return lr
    elif (10 <= epoch) and (epoch < 15):
        return lr / 10.
    else:
        return lr / 100.


# dataset info
batch_size = 128
epochs = 20
num_classes = 10
# hyper-parameters
layer1_size = 1000
layer2_size = 1000
layer3_size = 1000

lambda1s = [ 5e-05, 2e-05, 5e-06, 2e-06, 0. ]
lambda2 = 0.

loss_dict = {}
acc_dict = {}
zero_dict = {}
dead_dict = {}
with open('./logs/mnist/multi_test/{}-{}-{}_result.txt'.format(layer1_size, layer2_size, layer3_size), "w") as f:
    f.write("lambda1 accuracy layer1_zero-layer2_zero-layer3_zero\n")
    for lambda1 in lambda1s:
        print("L1 regularization coefficient: {}".format(lambda1))
        losses = [] # the last test loss list
        accuracies = [] # the last test accuracy list
        zero_acts = [] # the number of zero activations (3 dimension each)
        dead_acts = [] # totally dead activations (3 dimension each)
        # lambda1, lambda2 = lambdas[reg_type]
        for manual_seed in range(10):
            seed(manual_seed)
            set_random_seed(manual_seed)
            hparams = {}
            hparams['l1'] = lambda1
            # hparams['l2'] = lambda2
            hparams['size'] = "{}-{}-{}".format(layer1_size, layer2_size, layer3_size)

            hparams['op'] = 'Adam'
            hparams['ep'] = epochs

            hparams['act'] = 'relu'
            # PATH info
            PATH = './logs/mnist/multi_test/'
            for label  in hparams:
                PATH += '{}={}-'.format(label, hparams[label])
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            PATH += "[{}]".format(manual_seed) + st
            if not os.path.exists(PATH):
                os.mkdir(PATH)
                print("Directory " , PATH ,  " Created ")

            # the data, split between train and test sets
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            model = Sequential()
            model.add(Dense(layer1_size, input_shape=(784,),
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # model.add(LeakyReLU(alpha=0.3))
            model.add(Dense(layer2_size,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))

            model.add(Dense(layer3_size,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # model.add(LeakyReLU(alpha=0.3))
            model.add(Dense(num_classes, activation='softmax'))

            model.summary()

            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                          metrics=['accuracy'])

            checkpoint_path = "checkpoints/"
            for label  in hparams:
                checkpoint_path += '{}={}-'.format(label, hparams[label])
            checkpoint_path += "[{}]".format(manual_seed) + st + ".ckpt"

            # Create checkpoint callback
            lr_callback = keras.callbacks.LearningRateScheduler(schedule, verbose=0)
            cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only=False,
                                                             verbose=0)

            tfboard_callback  = keras.callbacks.TensorBoard(log_dir=PATH,
                                                            histogram_freq=1,
                                                            batch_size=32,
                                                            write_graph=True,
                                                            write_grads=False,
                                                            write_images=False,
                                                            embeddings_freq=0,
                                                            embeddings_layer_names=None,
                                                            embeddings_metadata=None,
                                                            embeddings_data=None)
            t1 = time.time()
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test),
                                callbacks=[lr_callback, tfboard_callback, cp_callback])
            print("Training Time: {:.2f} sec".format(time.time()))
            score = model.evaluate(x_test, y_test, verbose=1)
            print('\nTest loss:', score[0])
            print('Test accuracy:', score[1])

            activations_list = []

            imodel1 = Sequential()
            imodel1.add(Dense(layer1_size, input_shape=(784,),
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            activations_list.append(imodel1.predict(x_test, batch_size=batch_size, verbose=0, steps=None))

            imodel2 = Sequential()
            imodel2.add(Dense(layer1_size, input_shape=(784,),
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            imodel2.add(Dense(layer2_size,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            activations_list.append(imodel2.predict(x_test, batch_size=batch_size, verbose=0, steps=None))

            imodel3 = Sequential()
            imodel3.add(Dense(layer1_size, input_shape=(784,),
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            imodel3.add(Dense(layer2_size,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            imodel3.add(Dense(layer3_size,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            activations_list.append(imodel3.predict(x_test, batch_size=batch_size, verbose=0, steps=None))

            # imodel4 = Sequential()
            # imodel4.add(Dense(layer1_size, input_shape=(784,),
            #                 activation='relu',
            #                 activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # imodel4.add(Dense(layer2_size,
            #                 activation='relu',
            #                 activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # imodel4.add(Dense(layer3_size,
            #                 activation='relu',
            #                 activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # imodel4.add(Dense(num_classes, activation='softmax',
            #                 activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
            # activations_list.append(imodel4.predict(x_test, batch_size=batch_size, verbose=0, steps=None))



            print('# Dead Activations:{:.2f}-{:.2f}-{:.2f}'.format(
                    sum(sum(activations_list[0]==0))/len(x_test),
                    sum(sum(activations_list[1]==0))/len(x_test),
                    sum(sum(activations_list[2]==0))/len(x_test) ))
            # print('# Totally Dead Activations:{}-{}-{}'.format(
            #         sum(sum(activations_list[0]==0)==len(x_test)),
            #         sum(sum(activations_list[1]==0)==len(x_test)),
            #         sum(sum(activations_list[2]==0)==len(x_test))))
            zero_act = [ sum(sum(activations_list[i]==0))/len(x_test) for i in range(3) ]
            f.write("{} {:.4f} {:.2f}-{:.2f}-{:.2f}".format(lambda1, score[1], *zero_act))
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
