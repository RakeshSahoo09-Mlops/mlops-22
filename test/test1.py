import sys, os
import numpy as np
from joblib import load


sys.path.append(".")

from utils import get_all_h_param_comb, tune_and_save
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

def train_dev_test_same_split(data, label, train_frac, dev_frac,random_state):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, random_state=42
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac,random_state=42)


    return x_train, y_train, x_dev, y_dev, x_test, y_test