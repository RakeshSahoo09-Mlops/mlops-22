## train.py

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pdb

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb_svm,
    get_all_h_param_comb_tree,
    tune_and_save,
)
from joblib import dump, load
from sklearn import tree

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

model_to_test = ["svm",'dt']

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

for classifier in model_to_test:
    if classifier == 'svm':
        clf = svm.SVC()
        # 1. set the ranges of hyper parameters
        gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
        params = {}
        params["gamma"] = gamma_list
        params["C"] = c_list
        h_param_comb = get_all_h_param_comb_svm(params)

    if classifier == 'dt':
        clf = tree.DecisionTreeClassifier()
        max_depth = [2, 4, 6, 8, 10]
        min_samples_leaf = [1, 2, 3, 4, 5]
        max_features = ['auto', 'sqrt', 'log2']
        params = {}
        params["max_depth"] = max_depth
        params["min_samples_leaf"] = min_samples_leaf
        params["max_features"] = max_features
        h_param_comb = get_all_h_param_comb_tree(params)

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    metric = metrics.accuracy_score
    actual_model_path = tune_and_save(
        clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
    )