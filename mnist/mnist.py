'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import datetime, time, os, argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Training settings
parser = argparse.ArgumentParser(description='Keras MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
# parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lambbda1', type=float, default=2e-06, metavar='N',
                    help='regularization factor of L1 (default: 2e-06)')
parser.add_argument('--lambda2', type=float, default=2e-06, metavar='N',
                    help='regularization factor of L2 (default: 2e-06)')
parser.add_argument('--layer1', type=int, default=300, metavar='N',
                    help='size of first hidden layer (default: 300)')
parser.add_argument('--layer2', type=int, default=100, metavar='N',
                    help='size of second hidden layer (default: 100)')

args = parser.parse_args()
# dataset info
batch_size = args.batch_size
epochs = args.epochs
num_classes = 10
# hyper-parameters
lambda1 = args.lambda1
lambda2 = args.lambda2

layer1_size = args.layer1
layer2_size = args.layer2

hparams = {}
hparams['l1'] = lambda1
hparams['l2'] = lambda2
hparams['size'] = "{}-{}".format(layer1_size, layer2_size)

hparams['op'] = 'Adam'
hparams['ep'] = epochs
# PATH info
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
PATH = './logs/mnist/'
for label  in hparams:
    PATH += '{}={}-'.format(label, hparams[label])
PATH += st
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
model.add(Dense(layer1_size, activation='relu', input_shape=(784,), activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
model.add(Dense(layer2_size, activation='relu', activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))
model.add(Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.l1_l2(l1=lambda1, l2=lambda2)))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

checkpoint_path = "checkpoints/"
for label  in hparams:
    checkpoint_path += '{}={}-'.format(label, hparams[label])
checkpoint_path += st + ".ckpt"

# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=0)

tfboard_callback  = keras.callbacks.TensorBoard(log_dir=PATH, histogram_freq=1, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[tfboard_callback, cp_callback])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


imodel1 = Sequential()
imodel1.add(Dense(layer1_size, activation='relu', weights=model.layers[0].get_weights() , input_shape=(784,), activity_regularizer=keras.regularizers.l1(lambda1)))
activations1 = imodel1.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

imodel2 = Sequential()
imodel2.add(Dense(layer1_size, activation='relu', weights=model.layers[0].get_weights() , input_shape=(784,), activity_regularizer=keras.regularizers.l1(lambda1)))
imodel2.add(Dense(layer2_size, activation='relu', weights=model.layers[1].get_weights() , activity_regularizer=keras.regularizers.l1(lambda1)))
activations2 = imodel2.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

imodel3 = Sequential()
imodel3.add(Dense(layer1_size, activation='relu', weights=model.layers[0].get_weights() , input_shape=(784,), activity_regularizer=keras.regularizers.l1(lambda1)))
imodel3.add(Dense(layer2_size, activation='relu', weights=model.layers[1].get_weights() , activity_regularizer=keras.regularizers.l1(lambda1)))
imodel3.add(Dense(num_classes, weights=model.layers[2].get_weights(), activity_regularizer=keras.regularizers.l1(lambda1), activation='softmax'))
activations3 = imodel3.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

print('# Dead Activations:{}-{}-{}'.format(
        sum(sum(activations1==0))/len(x_test),
        sum(sum(activations2==0))/len(x_test),
        sum(sum(activations3==0))/len(x_test) ))
print('# Totally Dead Activations:{}-{}-{}'.format(
        sum(sum(activations1==0)==len(x_test)),
        sum(sum(activations2==0)==len(x_test)),
        sum(sum(activations3==0)==len(x_test))))
