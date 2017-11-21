import numpy as np
from keras.datasets import imdb
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.datasets.imdb
import os
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


class RNN:

    '''
    RNN classifier
    '''
    def __init__(self, dataset, train_x, train_y, test_x, test_y, dict_size=5000, example_length=500, embedding_length=32, epoches=15, batch_size=128):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epoches:
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.epoches = epoches
        self.example_len = example_length
        self.dict_size = dict_size
        self.embedding_len = embedding_length
        self.dataset = dataset
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


        # prepare embedding matrix

        embedding_matrix = np.zeros((5000, dict_size))
        for word, i in self.dataset.items():
            print(word)
            if i >= example_length:
                continue
            embedding_vector = self.dataset.get(i)

            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i,:] = embedding_vector[word]

        embedding_layer = Embedding(dict_size,
                                    dict_size,
                                    weights=[embedding_matrix],
                                    trainable=False)




        # TODO:build model
        self.model = Sequential()
        self.model.add(embedding_layer)
        self.model.add(LSTM(self.batch_size, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])



    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        # TODO: fit in data to train your model
        self.model.fit(self.dataset, self.train_x, self.train_y, self.batch_size, self.epoches)

    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y)


if __name__ == '__main__':

     dataset = keras.datasets.imdb.get_word_index()
     (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=5000)
     rnn = RNN(dataset, train_x, train_y, test_x, test_y)
     rnn.train()
     rnn.evaluate()

