import tensorflow as tf
from tensorflow import keras
import os, time, datetime
import numpy as np
import matplotlib.pyplot as plt


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
# hyper-parameters
NUM_WORDS = 10000
lambda1 = 0.0001
lambda2 = 0.0001

hparams = {}
hparams['l1'] = lambda1
hparams['l2'] = lambda2
hparams['op'] = 'Adam'


(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
# PATH info
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
PATH = './logs/overfit/'
for label  in hparams:
    PATH += '{}={}-'.format(label, hparams[label])
PATH += st
if not os.path.exists(PATH):
    os.mkdir(PATH)
    print("Directory " , PATH ,  " Created ")
tfboard_callback  = keras.callbacks.TensorBoard(log_dir=PATH, histogram_freq=1, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])


l1_model = keras.models.Sequential([
    keras.layers.Dense(16, activity_regularizer=keras.regularizers.l1(lambda1),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activity_regularizer=keras.regularizers.l1(lambda1),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model = keras.models.Sequential([
    keras.layers.Dense(16, activity_regularizer=keras.regularizers.l2(lambda2),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activity_regularizer=keras.regularizers.l2(lambda2),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=40,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2,
                                      callbacks=[tfboard_callback])



l1_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l1_model.summary()
l1_model_history = l1_model.fit(train_data, train_labels,
                                epochs=40,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2,
                                callbacks=[tfboard_callback])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model.summary()
l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=40,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2,
                                callbacks=[tfboard_callback])


plot_history([('baseline', baseline_history),
              ('l1', l1_model_history),
              ('l2', l2_model_history)])

plt.show()
