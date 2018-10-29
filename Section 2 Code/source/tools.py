import numpy as np
from pprint import pprint
import sys

def get_vpo2(values, goback=1):
    """
    Supporting rolling windows with goback.

    Just make sure that goback parameter here
    will match the one in network configuration
    in input parameter.

    if goback=2 inputs=2 in train.py:
    'new_model_conf': dict(model=get_mlp, inputs=2)
    """
    yy=[np.nan]*len(values)
    for i, v in enumerate(values):
        if i+goback>len(values)-1:
            break
        fdata=[]
        ii=0
        while ii < goback:
            fdata.append(values[i+ii])
            ii+=1
        if(len(fdata) == 1):
            fdata=fdata[0]
        yy[i+goback]=fdata
    return yy

def get_params(script='train.py'):
    xa=''
    if script == 'train.py':
        xa='[plot|ploth]'
    try:
        name, epochs, batches=sys.argv[1:4]
    except ValueError:
        print('Usage: %s model_name epochs batch_size %s' % (script, xa))
        exit(1)
    try:
        plot=sys.argv[4]
    except IndexError:
        plot=False

    return name, int(epochs), int(batches), plot

def train_test_split(rawx, xpo):
    train_size=int(len(rawx)*0.80)
    test_size=int(len(rawx)*0.20)
    #print(train_size, test_size, len(rawx))
    train_x, train_y = np.array(rawx[:train_size]), np.array(xpo[:train_size])
    test_x, test_y = np.array(rawx[train_size:]), np.array(xpo[train_size:])
    return train_x, train_y, test_x, test_y

def logme(msg):
    pprint(msg)
