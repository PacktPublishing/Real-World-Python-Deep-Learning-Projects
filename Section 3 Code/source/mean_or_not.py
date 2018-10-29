#!/usr/bin/env python3
"""
Predict if a tweet is mean or not and
the probability of it been mean.

Note: Once you've decied that your model is
good enough the common practice is to train it
with the whole data set that you have and then
use it for predictions.To make things simple
in this script, we're using our training model.
"""
from train import get_params, confs
from prep import clean

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import pickle
import sys

if __name__ == '__main__':
    name, epochs, batches=get_params()
    model=confs[name]
    mname='models/model-%s-%d-%d' % (name, epochs, batches)
    model_file=mname+'.h5'
    tokenizer_file=mname+'-tokenizer.pickle'
    # Loading the model.
    if os.path.exists(model_file):
        model=load_model(model_file)
        print('Model loaded!')
    else:
        print("Can't find %s model, train it first using 'train.py %s %d %d'" % (mname, name, epochs, batches))
    # Loading tokenizer.
    # We need to use the same tokenizer that we've used
    # for training and testing to get the same encoding
    # for known words in our vocabulary and also in
    # the word embedding that we've created during the training.
    if os.path.exists(tokenizer_file):
        tokenizer=pickle.load(open( mname+'-tokenizer.pickle', "rb" ))
        print('Tokenizer loaded!')
    else:
        print("Can't find tokenizer for %s model, train it first using 'train.py %s %d %d'" % (mname, name, epochs, batches))
    # Get the tweet.
    print("Type in one tweet per line and hit CTRT-D when you're done:")
    for tweet in sys.stdin.readlines():
        # Cleanup the tweet before we use our model.
        t=clean([tweet], True)
        # Encode and pad our tweet with the same tokenizer
        # that we've used for training and testing.
        # We've set our own variable in
        # tokenizer._max_padding_len on training to store
        # informations about the maximum lenght of our encoded text.
        t=tokenizer.texts_to_sequences(t)
        t=pad_sequences(t, maxlen=tokenizer._max_padding_len, padding='post')
        # Get one of a predicted classes
        # In our case it's 0 for negative tweet and 1 for positive.
        pc=model.predict_classes(t)
        pc=pc[0][0]
        # We can also can get the probablity of prediction been in a given class.
        # By default we get the probablity of being in class no. 1 which in our
        # case is probability of a tweet to be postive.
        # We can get the probablity of tweet being mean just by calculating 1-prob.
        prob=model.predict_proba(t)
        prob=prob[0][0]
        print('%s -%smean (%.2f%%)' % (tweet.rstrip(), (' ' if pc==0 else ' not '),(1-prob)*100))
