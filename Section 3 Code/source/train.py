#!/usr/bin/env python3
"""
Build, train and evaluate a CNN model for text classification.
"""
import conf

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

import math
import os
import sys
import pandas
import pickle

from prep import get_data

def get_text_cnn(inputs, max_length, dim=25):
    """
    input - vocabulary size, a number of unique words in
            our data set

    max_lenght - the maximum number of words in our data set

    dim - word embedding dimension, the lenght of word vector
          that will be produced by this layer
    """
    print('CNN: inputs: %d, word embeddings dimesions: %d, input_length: %d' % (inputs, dim, max_length))
    model = Sequential()
    model.add(Embedding(inputs, dim, input_length=max_length))
    # Extract feature maps/most common "phrases".
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
    # Pick up the "best ones", pooling=reducting.
    model.add(MaxPooling1D(pool_size=4))
    # Just put everything together into one vector.
    model.add(Flatten())
    # This is the standard output for classification.
    # It matches our two classes 0 and 1.
    model.add(Dense(1, activation='sigmoid'))
    return model

confs={'default': dict(model=get_text_cnn)}

def train_model(name, train_x, train_y, epochs, batches, inputs, max_lenght, test_x, test_y):
    """
    Compile and train model with choosen parameters.
    """
    mparams=confs[name]
    model=mparams['model']
    model=model(inputs, max_lenght)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit model on training data, validate during training on test data.
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batches, verbose=2)
    return model, name, mparams

def get_params(script='train.py'):
    """
    Get command line parameters.
    """
    try:
        name, epochs, batches=sys.argv[1:4]
    except ValueError:
        print('Usage: %s model_name epochs batch_size' % sys.argv[0])
        exit(1)
    return name, int(epochs), int(batches)

if __name__ == '__main__':
    # Getting our command line parameters
    name, epochs, batches=get_params()
    train_x, train_y, test_x, test_y, inputs, max_length, t=get_data(do_cleanup=True, filter_stopwords=True)
    print('Train/Test Data lenght', len(train_x), len(test_x))
    model, name, mp =train_model(name, train_x, train_y, epochs, batches, inputs, max_length, test_x, test_y)
    # Save model to use for classification later on
    mname='models/model-%s-%d-%d' % (name, epochs, batches)
    model.save(mname+'.h5')
    with open(mname+'-tokenizer.pickle', 'wb') as ts:
        pickle.dump(t, ts)
    title='%s (epochs=%d, batch_size=%d)' % (name, epochs, batches)
    # Test our model on both data that has been seen
    # (training data set) and unseen (test data set)
    print('Evaluation for %s' % title)
    loss, acc = model.evaluate(train_x, train_y, verbose=2)
    print('Train Accuracy: %.2f%%' % (acc*100))
    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    print('Test Accuracy: %.2f%%' % (acc*100))
