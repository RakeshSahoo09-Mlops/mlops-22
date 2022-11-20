## Part 1: Import the Libraries

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.tree import DecisionTreeClassifier


## Part 2: Set the ranges of hyper parameters for hyper-parameter tunning

# SVC parameter: We will use gamma and c parameter for tunning

gamma = [0.01, 0.003, 0.001, 0.0003, 0.0001]
c_l = [0.15, 0.25, 0.6, 0.75, 1, 3, 5, 8, 10]

# Decision Tree classifier parameter: We will use max_depth,  min_samples_split, and max_features parameter for tunning

max_depth = [2, 4, 6, 10]
min_samples_split = [5, 10, 20, 50]
max_features =  [3,4,5]

# 5 different splits of train/test/validation set

train_frac = [0.8, 0.78, 0.75, 0.70, 0.65]
val_frac =   [0.1, 0.11, 0.125, 0.15, 0.175]
test_frac =  [0.1, 0.11, 0.125, 0.15, 0.175]


## Part 3: Load the Digit dataset

digits = datasets.load_digits()
print(f"\nGiven Image size : {digits.images.shape}")
print("\n5 Confusion Matrix for respective splits:\n")


## Part 4: Data Pre-processing

acc_df = pd.DataFrame()
acc_svc = list()
acc_dt = list()

best_acc_svc = -1.0
best_model_svc = None
best_h_params_svc = dict()
best_acc_dt = -1.0
best_model_dt = None
best_h_params_dt = dict()

## Train, Test, Val set split

for i in range(5):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_val, y_train, y_val = train_test_split(data, digits.target, test_size = val_frac[i], shuffle = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = test_frac[i], shuffle = True)
    
    # For Support Vector Machines (SVM)
    
    for g in gamma:
        for d in c_l:            
                                   
            clf = svm.SVC(gamma = g, C = d, random_state = 78)
            
            clf.fit(X_train, y_train)

            ## Get the prediction of validation set
            
            predicted = clf.predict(X_val)
            cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = y_val)    
                
            ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
                
            if (cur_acc > best_acc_svc):
                best_acc_svc = cur_acc
                best_model_svc = clf
                best_h_params_svc[i] = {"gamma":g, "C": d}

    
    # For Decision Tree Clssifier    
    
    for d in max_depth:
        for l in min_samples_split:
            for m in max_features:
            
                ## Define the model
                ## Create a classifier: a Decision Tree classifier
                
                clf = DecisionTreeClassifier(max_depth = d, min_samples_split = l, max_features = m, random_state = 42)

                ## Train the model
                ## Learn the digits on the train subset
                
                clf.fit(X_train, y_train)

                ## Get the prediction of validation set
                
                predicted = clf.predict(X_val)
                cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = y_val)
                
                ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest
                
                if (cur_acc > best_acc_dt):
                    best_acc_dt = cur_acc
                    best_model_dt = clf
                    best_h_params_dt[i] = {"max_depth":d, "min_samples_split": l, "max_features": m}

    
    ## Get the test set prediction on best params i.e. using best model (SVM model) and calculate the accuracy of test set using best model

    pred_svc = best_model_svc.predict(X_test)
    acc = metrics.accuracy_score(y_pred = pred_svc, y_true = y_test)
    acc_svc.append(acc)
    
    ## Confusion Matrix

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_svc)
    disp.figure_.suptitle("Cf Matrix")
    print(f"\nObtained CF matrix fro SVC:\n\n{disp.confusion_matrix}\n")
    plt.show()

    
    ## Get the test set prediction on best params i.e. using best model (Decision Tree model) and calculate the accuracy of test set using best model
    
    pred_dt = best_model_dt.predict(X_test)
    acc = metrics.accuracy_score(y_pred = pred_dt, y_true = y_test)
    acc_dt.append(acc)

    ## Confusion Matrix

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_dt)
    disp.figure_.suptitle("Cf Matrix")
    print(f"\nObtained CF matrix fro SVC:Decision_Tree:\n\n{disp.confusion_matrix}\n")
    plt.show()
    

## calculate the mean and standard deviations of both the classifier's performances

acc_svc.append(np.mean(acc_svc))
acc_dt.append(np.mean(acc_dt))
acc_svc.append(np.std(acc_svc))
acc_dt.append(np.std(acc_dt))
#acc_df[] = ['run svm decision_tree']
acc_df[''] = ['1', '2', '3', '4', '5', "mean", "std"]
acc_df["SVC"] = acc_svc
acc_df["DT"] = acc_dt

##
print("\nBest hyper-params for different splits are obtained as :\n")
print(best_h_params_svc, best_h_params_dt, sep='\n')
print("\nObtained Accuracy after run svm and decision_tree\n")
print(acc_df)  
print("\n")

## Write the performance metrics in the readme.md

acc_df.to_markdown("readme.md")

## save best models of DT and SVC

import joblib

joblib.dump(best_model_svc, './models/best_model_svc.pkl')
joblib.dump(best_model_dt, './models/best_model_dt.pkl')


## Save the results

acc_df.to_markdown("./results/svm_dt_random_state_42.txt")