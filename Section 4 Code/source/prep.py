#!/usr/bin/env python3
"""
Prepare images to work with CNN model.

Inspired by https://github.com/kylemcdonald/SmileCNN
We're using data from https://github.com/hromi/SMILEsmileD/tree/master/SMILEs

Download the repository as zip file and put SMILEs/negatives and SMILEs/positives
into the data directory in the source direcotry for this section.

Please install sckit-image package before
using this script with:
$ conda install scikit-image
"""
from os import listdir, path, remove

from skimage.io import imread
from skimage.measure import block_reduce
from PIL import Image

import numpy as np
from keras.utils import np_utils

def img2array(f, detection=False, ii_size=(64, 64)):
    """
    Convert images into matrixes/two-dimensional arrays.

    detection - if True we will resize an image to fit the
                shape of a data that our first convolutional
                layer is accepting which is 32x32 array,
                used only on detection.

    ii_size - this is the size that our input images have.
    """
    rf=None
    if detection:
        rf=f.rsplit('.')
        rf=rf[0]+'-resampled.'+rf[1]
        im = Image.open(f)
        # Create a smaller scalled down thumbnail
        # of our image.
        im.thumbnail(ii_size)
        # Our thumbnail might not be of a perfect
        # dimensions, so we need to create a new
        # image and paste the thumbnail in.
        newi = Image.new('L', ii_size)
        newi.paste(im, (0,0))
        newi.save(rf, "JPEG")
        f=rf
    # Turn images into an array.
    data=imread(f, as_gray=True)
    # Downsample it from 64x64 to 32x32
    # (that's what we need to feed into our first convolutional layer).
    data=block_reduce(data, block_size=(2, 2), func=np.mean)
    if rf:
        remove(rf)
    return data

def prep_array(data, detection=False):
    """
    Convert our input array into the right format.

    detection - if True we just wrapping up a single
                image's array into a list to make things
                consistent.
    """
    if detection:
        data=[data]
    # By default values converted from our images
    # are integers in range from 0 to 255 and our
    # network will be really slow working with them.
    # So, we need to convert them into values from
    # 0.0 to 1.0 range which works much better in our case.
    data=np.asarray(data) / 255.0
    # We need to wrap each pixel value insided it's own array.
    # This is the quick way of doing it.
    data=np.expand_dims(data, axis=-1)
    return data

def load_data(data_directory):
    """
    Go trough each image in a data directory,
    convert it into an array, add into
    our input array X and return it.
    """
    X=[]
    for filename in listdir(data_directory):
        if not filename.endswith('.jpg'):
            continue
        p=path.join(data_directory, filename)
        data=img2array(p)
        X.append(data)
    return prep_array(X)

def gen_labels(length, label):
    """
    Return a length list of label.
    """
    return [ label for _ in range(length) ]

def get_data():
    """
    Generate X and Y arrays, inputs and classes
    ready for use in our convolutional network.
    """
    # Load images, generate labels, starting with negatives
    x_neg = load_data('data/negatives/negatives7')
    y_neg = gen_labels(len(x_neg), 0)

    x_pos = load_data('data/positives/positives7')
    y_pos = gen_labels(len(x_pos), 1)

    # Merge negative and postive data into one.
    X=np.concatenate([x_neg, x_pos])
    Y=np.asarray(y_neg+y_pos)

    # By default we will have 64 bit values,
    # it will run quicker if we convert them into
    # 32 bit.
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)

    # Get the dimensions and number of color channels
    # that we have in our data.
    # Here we have (32,32,1) which means 32x32 array with
    # one color channel (because we have black and white images)
    inputs=X.shape[1:]
    # Number of classes we want to predict.
    # 0 - not smiling, 1 - smiling.
    classes=2
    # Convert classes to vector, this is needed when we use
    # softmax in the last layer.
    Y = np_utils.to_categorical(Y, classes).astype(np.float32)

    # Shuffle all the data because
    # we have more negative samples
    # than positive ones.
    # Then keras will take care of
    # spliting the data for us
    # later on training.
    ixes = np.arange(len(X))
    np.random.shuffle(ixes)
    X = X[ixes]
    Y = Y[ixes]
    return X, Y, inputs, classes

if __name__ == '__main__':
    from pprint import pprint
    X, Y, inputs, classes=get_data()
    print('Inputs: %s' % repr(inputs))
    print('X[0] (first encoded image):')
    pprint(X[0])
    print('Y[0] (first encoded class):')
    pprint(Y[0])
    print('Classes %s' % classes)
    pprint(np_utils.to_categorical([0,1], classes).astype(np.float32))
