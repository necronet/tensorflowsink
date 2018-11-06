import collections
import os
import logging
import tensorflow as tf
import pandas as pd

# splitting file based on validation_steps
# split -l $[ $(wc -l logfile500k.csv_structured.csv | xargs | cut -d" " -f1) * 70 / 100 ] logfile500k.csv_structured.csv
# wc -l logfile500k.csv_structured.csv | xargs | cut -d" " -f1

data_path='/Users/necronet/Documents/repos/tensorflow-sink/data/data'
train_file="logfile500k.csv_structured-train.csv"
test_file="logfile500k.csv_structured-test.csv"

logger = logging.getLogger('RNN-DeepLog')

def read_words(filename):
    #This should be optimize for loading stream based
    return pd.read_csv(filename).EventId


def build_vocab(filename):
    # Check later if there is a better way to transform it to id
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    logger.info('Loading data {}'.format(train_file))
    train_path = os.path.join(data_path, train_file)
    test_path = os.path.join(data_path, test_file)
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, test_data, vocabulary, reversed_dictionary

# valid_data, test_data,
train_data, test_data, vocabulary, reversed_dictionary = load_data()

logger.info('Data loaded Vocabulary [{}]'.format(vocabulary))
