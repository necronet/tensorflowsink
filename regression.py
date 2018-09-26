# Tensor flow example with regression
from __future__ import absolute_import, division

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

boston_housing = keras.datasets.boston_housing

(train_data, train_labels),(test_data,test_labels) = boston_housing.load_data()

# Shuffle the train data/labels
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))
print("Testing set: {}".format(test_data.shape))

# Display sample features notice different scales
print(train_data[0])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
# print(df.head())

# Standardize the values in the guide this is called
# wrongly normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# print(train_data[0])

def build_model():
    lr = 0.001
    model = keras.Sequential([
        keras.layers.Dense(64,
            activation = tf.nn.relu,
            input_shape = (train_data.shape[1],)),
        keras.layers.Dense(64,
            activation = tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(lr)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


model = build_model()
model.summary()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print('')
        print('.',end='')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

EPOCHS = 500

history = model.fit(train_data,train_labels,
            epochs=EPOCHS,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stop, PrintDot()])


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()

# plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))


test_predictions = model.predict(test_data).flatten()

# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [1000$]')
# plt.ylabel('Predictions [1000$]')
# #plt.axis('equal')
# plt.xlim(plt.xlim())
# plt.ylim(plt.ylim())
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()
