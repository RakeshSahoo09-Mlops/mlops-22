## Ignore warnings

import warnings
warnings.filterwarnings("ignore")

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

## Part 1: Import the Libraries

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean

## Part 2: Set the ranges of hyper parameters for hyper-parameter tunning

## We will use gamma and c parameter for tunning

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10] 

#train_frac = 0.8
#test_frac = 0.1
#dev_frac = 0.1

train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1


## Part 3: Load the Digit dataset

digits = datasets.load_digits()
print(f"\nImage size in the digits dataset is: {digits.images.shape}")

## Sanity check of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

## Part 4: Data Pre-processing

## Define Image sizes

img_sizes = [(8,8),(4,4),(5,5),(7,7)]

for size in img_sizes:
    if size == (8, 8):
        print(f"\nModeling on original image data (digits dataset) of size: {digits.images.shape}\n")
    else:
        print(f"\nOriginal size of image data (digits datasets): {digits.images.shape}\n")
        images = np.empty(shape=(len(digits.images), *size))

        for idx, image in enumerate(digits.images):
            images[idx] = resize(image, size, anti_aliasing=True)

        print(f"\nNew size of image data (digits datasets): {images.shape}\n")
    
    ## Flatten the images
    
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    ## Part 5: Split data into Train, dev, test set

    ## Train set: used to train the model
    ## Dev set: used for hyper-parameter tunning of model and decide the best model
    ## Test set: Used to evaluate the performance of the model

    dev_test_frac = 1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )

    ## Model Building: Classification
    ## Initialize variables like best_acc, best_model and best_h_params

    best_acc = -1.0
    best_model = None
    best_h_params = None

    ## Part 6: For every combination-of-hyper-parameter values, calculate the accurcay
    ## and find the best combination of hyper-parameters

    ## df to capture accuracy for each param combination

    acc_df = pd.DataFrame(columns=["Gamma","C","train","dev","test"])
    acc_dict = dict()

    ## We will pass each combination of gamma and C
    
    for g in gamma_list:
        for c in c_list:
            acc_dict["Gamma"] = g
            acc_dict["C"] = c
            
            ## Define the model
            ## Create a classifier: a support vector classifier
            
            clf = svm.SVC(gamma=g, C=c)

            ## Train the model
            ## Learn the digits on the train subset
            
            clf.fit(X_train, y_train)

            ## Get dev set predictions
            
            for _, features, target in zip(["train","dev","test"],[X_train, X_dev, X_test], [y_train, y_dev, y_test]):
                predicted = clf.predict(features)
                cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = target)
                acc_dict[_] = cur_acc
                
                ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
                
                if (_ == "dev") & (cur_acc > best_acc):
                    best_acc = cur_acc
                    best_model = clf
                    best_h_params = {"gamma":g, "C": c}

            acc_df = acc_df.append(acc_dict, ignore_index=True)
            
    print(f"Model Accuracy for all combinations of hyper-params for train, dev & test: \n\n{acc_df}\n")

    print(f"\nBest hyper-params are: {best_h_params}")
    print(f"Dev set accuracy with Best hyper-params is: {best_acc}\n")

    ## Min, Max, mean, Median accurcay

    print("Train set:\n")
    print("Min accuracy on Train set", acc_df['train'].min())
    print("Max accuracy on Train set", acc_df['train'].max())
    print("Mean accuracy on Train set", acc_df['train'].mean())
    print("Median accuracy on Train set", acc_df['train'].median())

    print("\nTest set:\n")
    print("Min accuracy on Test set", acc_df['test'].min())
    print("Max accuracy on Test set", acc_df['test'].max())
    print("Mean accuracy on Test set", acc_df['test'].mean())
    print("Median accuracy on Test set", acc_df['test'].median())

    print("\ndev set:\n")
    print("Min accuracy on dev set", acc_df['dev'].min())
    print("Max accuracy on dev set", acc_df['dev'].max())
    print("Mean accuracy on dev set", acc_df['dev'].mean())
    print("Median accuracy on dev set", acc_df['dev'].median())

    
    ## Calculation of train, dev, test set accuracy with best hyper-params

    for _, features, target in zip(["train","dev","test"],[X_train, X_dev, X_test],[y_train, y_dev, y_test]):
        predicted = best_model.predict(features)
        acc = metrics.accuracy_score(y_pred = predicted, y_true = target)
        print(f"\n\nAccuracy for {_} set using best hyper-params is: {acc}")

    print("\n")

    ## Sanity check of predictions on test set
    
    predicted_testset = best_model.predict(X_test)
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_testset):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    ## Part 7: Report the test set accurancy with that best model.
    
    ## Compute evaluation metrics
    
    print(
        f"Classification report for classifier {best_model}:\n"
        f"{metrics.classification_report(y_test, predicted_testset)}\n"
    )

## Confusion Matrix

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_testset)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
