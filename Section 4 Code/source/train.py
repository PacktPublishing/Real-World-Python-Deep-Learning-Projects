#!/usr/bin/env python3
"""
Build, train and evaluate a CNN model for smile recognition.
"""
import conf
from prep import get_data

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

import os
import sys


def get_smile_cnn(inputs=(32,32,1), classes=2):
    """
    Define our CNN model for simple recognition.

    inputs - a tupe of (rows, cols, channel) of our input arrays,
             describing the "shape" of our data, we need to provide
             it for the first Convolutional layer.

    classes - a number of classes we want to predict, here we have
              only two classes - 0 - not smiling, 1 - smiling.
    """
    model = Sequential()
    # X filters will create an X feature maps, the more we have
    # the more "features" we can "catch".
    # Kernel size is the size of a matrix that will extract
    # our "features".
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=inputs))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    # Factors by which to downscale and choose the maximum/most visible
    # values.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # % of neurons to ignore/help with making your model
    # being more general, doing better on unseen data.
    # 0.25 means 25%, try to comment out Dropout layers
    # and see what happens...
    model.add(Dropout(0.25))
    model.add(Flatten())
    # Adding a fully connected layer just before last layer
    # help us learn the combinations of features from previous
    # feature maps (results of Conv2D and MaxPooling2D layers).
    model.add(Dense(128, activation='relu'))
    # Same as comment above.
    model.add(Dropout(0.5))
    # Last layer is for classification, softmax allows us
    # to recognize a few different classes.
    model.add(Dense(classes, activation='softmax'))
    return model

confs={'default': dict(model=get_smile_cnn)}

def train_model(name, train_x, train_y, epochs, batches, inputs, classes):
    """
    Compile and train model with choosen parameters.
    """
    mparams=confs[name]
    model=mparams['model']
    model=model(inputs, classes)
    # Compile model.
    # We're using categorical_* and softmax function in the last
    # layer to classify multiple classes.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # We're using 90% of data for training and 10% for validation/testing.
    trs, tt=int(len(train_x)*0.90), int(len(train_x)*0.10)
    train_x, train_y, test_x, test_y=train_x[0:trs], train_y[0:trs], train_x[-tt:], train_y[-tt:]
    # Fit model on training data, validate during training on test data.
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batches, verbose=2)
    return model, name, mparams, train_x, train_y, test_x, test_y

def get_params(predict=False):
    """
    Get command line parameters.
    """
    try:
        name, epochs, batches=sys.argv[1:4]
    except ValueError:
        print('Usage: %s model_name epochs batch_size filename' % sys.argv[0])
        exit(1)
    filename=None
    if predict:
        try:
            filename=sys.argv[4]
        except IndexError:
            pass
    return name, int(epochs), int(batches), filename

if __name__ == '__main__':
    # Getting our command line parameters
    name, epochs, batches, _=get_params()
    # Getting our images correctly converted
    # to the right format of arrays/matrices.
    train_x, train_y, inputs, classes=get_data()
    # Time for training!
    model, name, mp, train_x, train_y, test_x, test_y =train_model(name, train_x, train_y, epochs, batches, inputs, classes)
    # Save model to use for classification later on.
    mname='models/model-%s-%d-%d' % (name, epochs, batches)
    model.save(mname+'.h5')
    title='%s (epochs=%d, batch_size=%d)' % (name, epochs, batches)
    print('Evaluation for %s' % title)
    loss, acc = model.evaluate(train_x, train_y, verbose=2)
    print('Train accuracy: %.2f%%' % (acc*100))
    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    print('Test accuracy: %.2f%%' % (acc*100))
