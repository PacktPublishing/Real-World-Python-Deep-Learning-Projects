#!/usr/bin/env python3
"""
Prepare an airline data for use in a neural network to solve time-series
prediction problem.

Download data from https://data.worldbank.org/indicator/IS.AIR.PSGR in CSV
format and put it under data/airline_data.csv in current folder.
"""
import sys
import pandas
import numpy as np
import conf
from tools import get_vpo2

def get_raw_xy(data, column='Country Name', value='World'):
    """
    Get a number of carried passangers for each year
    for a specific Country or a part of the World.

    data - pandas DataFrame object with our airline data

    column - the name of the column to use for selecting data for a specific country,
             usually 'Country Name' or 'Country Code'

    value - depends on a specific column, when column is 'Country Name',
            value is just a name of a country or a part of world like 'France',
            when column is 'Country Code', value would be "FN"

    Returns a tuple with:
        - list of number of passangers carried for each year
        - list of years

    By default we're using the data for the whole World.
    """
    v=data.loc[data[column] == value].iloc[:,17:62]
    x=[ v for v in v.values.astype('float32')[0] ]
    return x, [ int(vv) for vv in list(v) ]

def get_vpo(values):
    """
    Reformat data as a supervising learning problem
    that can be used used in a neural network.

    values - list with a number of passangers for each year

    For example:
    values=[42, 43, 45, 46]

    Meaning:
    42 # data for 1973
    43 # '74
    45 # '75
    46 # '76
    ...

    Out neural network needs two set of values for training to solve
    a prediction problem.

    Y - is a value that we want to predict
    X - is a value that we base our prediction on

    We can assume that the value of passangers for a given year (Y)
    corelates with a value of passangers in previous year (or years) (X).

    X         Y (those are values that we want to predict)
    nan       42  # X=no data for previous year Y=data for 1973
    42        43  # X=data for 1973             Y=data for 1974
    43        45  # X=1974                      Y=1975
    45        46  # X=1975                      Y=1976
    ...

    Here we want to generate X as a list of values that we use
    for prediction for each year.
    """
    yy=[np.nan]*len(values)
    for i, v in enumerate(values):
        if i+1>len(values)-1:
            break
        yy[i+1]=values[i]
    return yy

def get_data(f='data/airline_data.csv', logme=lambda x:x):
    """
    Get airline data and prepare it for use in a neural network.
    Skip the first year where we have no past data, so we have no base
    for a prediction.
    """
    # Reading our data, skipping 2 lines of comments.
    d=pandas.read_csv(f, header=2)
    values, years = get_raw_xy(d)
    logme('Raw values')
    rv=list(zip(years, values))
    logme(rv)
    # To base prediction on more that year's value in the past
    # replace get_vpo with get_vpo2
    # and set goback parameter as the same as inputs in an entry
    # entry in confs inside train.py
    past_values=get_vpo(values)
    logme("Previous year's values X/ this year's value Y")
    logme("For every value from X expect value from Y")
    dpo=list(zip([1972]+years, past_values, years, values))
    logme(dpo)
    logme("X/Y values (wihout empty ones)")
    dpo=list(zip(past_values[1:], values[1:]))
    logme(dpo)
    logme('X')
    logme(past_values[1:])
    logme('Y')
    logme(values[1:])
    return years[1:], past_values[1:], values[1:]

if __name__ == "__main__":
    from pprint import pprint
    get_data(logme=pprint)
