import sys, os
import numpy as np
from joblib import load


sys.path.append(".")

from utils import get_all_h_param_comb, tune_and_save, train_dev_test_same_split, train_dev_test_different_split
from sklearn import svm, metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
n_samples = len(digits.images)
dev_test_frac = 1-train_frac
data = digits.images.reshape((n_samples, -1))
label = digits.target

# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)

def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train

def test_tune_and_save():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)


def test_not_biased():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert len(set(predicted))!=1


def test_predicts_all():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert set(predicted) == set(y_test)


def test_random_state():
    x_train, y_train, x_dev, y_dev, x_test, y_test=train_dev_test_same_split(data, label, train_frac, dev_frac)
    x_train2, y_train2, x_dev2, y_dev2, x_test2, y_test2=train_dev_test_same_split(data, label, train_frac, dev_frac)
    x_train3, y_train3, x_dev3, y_dev3, x_test3, y_test3=train_dev_test_same_split(data, label, train_frac, dev_frac)
    assert np.all(x_test == x_test2)  # same array
    #assert x_test != x_test3  #different array
    




# what more test cases should be there
# irrespective of the changes to the refactored code.

# train/dev/test split functionality : input 200 samples, fraction is 70:15:15, then op should have 140:30:30 samples in each set


# preprocessing gives ouput that is consumable by model

# accuracy check. if acc(model) < threshold, then must not be pushed.

# hardware requirement test cases are difficult to write.
# what is possible: (model size in execution) < max_memory_you_support

# latency: tik; model(input); tok == time passed < threshold
# this is dependent on the execution environment (as close the actual prod/runtime environment)


# model variance? --
# bias vs variance in ML ?
# std([model(train_1), model(train_2), ..., model(train_k)]) < threshold


# Data set we can verify, if it as desired
# dimensionality of the data --

# Verify output size, say if you want output in certain way
# assert len(prediction_y) == len(test_y)

# model persistance?
# train the model -- check perf -- write the model to disk
# is the model loaded from the disk same as what we had written?
# assert acc(loaded_model) == expected_acc
# assert predictions (loaded_model) == expected_prediction