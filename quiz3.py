# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
import pdb
#from tests.test_utils import check_bias_labels,check_all_class_pred
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn import svm


def get_all_h_param_comb(params):
    h_param_comb = [{"gamma": g, "C": c} for g in params['gamma'] for c in params['C']]
    return h_param_comb
    
def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

# other types of preprocessing
# - image : 8x8 : resize 16x16, 32x32, 4x4 : flatteing
# - normalize data: mean normalization: [x - mean(X)]
#                 - min-max normalization
# - smoothing the image: blur on the image


def data_viz(dataset):
    # PART: sanity check visualization of the data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


# PART: Sanity check of predictions
def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

# PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        # PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("Found new best metric with :" + str(cur_h_params))
            print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params


def tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path):
    best_model, best_metric, best_h_params = h_param_tuning(
        h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
    )

    # save the best_model
    best_param_config = "_".join([h + "=" + str(best_h_params[h]) for h in best_h_params])
    
    if type(clf) == svm.SVC:
        model_type = 'svm' 

    best_model_name = model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = best_model_name
    dump(best_model, model_path)

    print("Best hyperparameters were:")
    print(best_h_params)

    print("Best Metric on Dev was:{}".format(best_metric))

    return model_path

# from utils import (
#     preprocess_digits,
#     train_dev_test_split,
#     h_param_tuning,
#     data_viz,
#     pred_image_viz,
#     get_all_h_param_comb,
#     tune_and_save,
# )
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
# define the evaluation metric
metric = metrics.accuracy_score


actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
)


# 2. load the best_model
best_model = load(actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

def check_bias_labels_2(pred_array,actual_array):
    total_unique_pred_labels = np.unique(pred_array).shape[0]
    total_unique_pred_actual = np.unique(actual_array).shape[0]
    assert total_unique_pred_labels == total_unique_pred_actual

def check_all_class_pred_2(pred_array,actual_array):
    
    overlap = np.unique(pred_array) - np.unique(actual_array)

    assert np.sum(overlap)==0

check_bias_labels_2(pred_array=predicted,
                  actual_array=y_test)
print("\n Step 3 done")
# print(np.unique(predicted))
# print(np.unique(y_test))
#print(len(set(predicted)-set(y_test)))
check_all_class_pred_2(pred_array=predicted,
                     actual_array=y_test)
print("Step4 done")


print("\n")
print("\n")
# print(len(np.unique(predicted)))
# print(len(np.unique(y_test)))
check_bias_labels_2(pred_array=predicted,
                  actual_array=y_test)
print("\n Step 3 done")
# print(np.unique(predicted))
# print(np.unique(y_test))
#print(len(set(predicted)-set(y_test)))
check_all_class_pred_2(pred_array=predicted,
                     actual_array=y_test)
print("Step4 done")
print("\n")
print("\n")
pred_image_viz(x_test, predicted)

# # 4. report the test set accurancy with that best model.
# # PART: Compute evaluation metrics
# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )