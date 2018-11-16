import collections
import os
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from batch_generator import KerasBatchGenerator
from loader import load_data

# splitting file based on validation_steps
# split -l $[ $(wc -l logfile500k.csv_structured-test.csv | xargs | cut -d" " -f1) * 70 / 100 ] logfile500k.csv_structured-test.csv
# wc -l logfile500k.csv_structured-test.csv | xargs | cut -d" " -f1

data_path='/Users/necronet/Documents/repos/tensorflow-sink/data/data'

logger = logging.getLogger('RNN-DeepLog')
# valid_data, test_data,
train_data, test_data, valid_data, vocabulary, reversed_dictionary = load_data()

logger.info('Data loaded Vocabulary [{}]'.format(vocabulary))
num_steps = 30
batch_size = 10
num_epochs = 5

train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

hidden_size = 100
model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

model.save(data_path + "final_model.hdf5")
