#!/usr/bin/env python3
"""
Detect smiles in a picture using a CNN network.
"""
import conf
from prep import img2array, prep_array
from train import get_params, confs

from keras.models import load_model

import os
import sys

if __name__ == '__main__':
    name, epochs, batches, filename=get_params(predict=True)
    model=confs[name]
    mname='models/model-%s-%d-%d' % (name, epochs, batches)
    model_file=mname+'.h5'
    # Loading the model.
    if os.path.exists(model_file):
        model=load_model(model_file)
    else:
        print("Can't find %s model, train it first using 'train.py %s %d %d'" % (mname, name, epochs, batches))
        exit(1)
    # Convert our input images in the same way
    # as we've prepared our training and testing data.
    img_data=prep_array(img2array(filename, detection=True), detection=True)
    # Here we're predicting on a single class as a number.
    pc=model.predict_classes(img_data)
    pc=pc[0]
    # Here the probablity for each class.
    prob=model.predict_proba(img_data)
    prob=prob[0]
    # If our prediction doesn't match file's mark (0/-ns for no smile, 1/-s for smile)
    # mark it with star for examiation.
    fname=filename.split('/')[-1]
    if pc==0:
        fmark=str(pc) if '-ns' in fname else '*'+str(pc)
    if pc==1:
        fmark=str(pc) if '-s' in fname else '*'+str(pc)
    print(filename, '%s (%.2f%% of smile(1/-s), %.2f%% of no smile(0/-ns))' % (fmark, prob[1]*100, prob[0]*100))
