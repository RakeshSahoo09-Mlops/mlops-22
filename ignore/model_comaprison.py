##model_comparison.py

from train import model_to_test
from utils import train_dev_test_split
from joblib import load
from train import dev_frac,train_frac,data,label
import glob
from sklearn.metrics import accuracy_score
import numpy as np

svm_accuracy_list=[]
dt_accuracy_list=[]

for classifier in model_to_test:
    for i in range(5):
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac
        )
        if classifier =="svm":
            best_model = load(glob.glob(".\models\svm_*.joblib")[0])
            predicted = best_model.predict(x_test)
            svm_accuracy_list.append(accuracy_score(y_test, predicted))
        if classifier =="dt":
            best_model = load(glob.glob(".\models\dt_*.joblib")[0])
            predicted = best_model.predict(x_test)
            dt_accuracy_list.append(accuracy_score(y_test, predicted))

svm_accuracy_array = np.array(svm_accuracy_list)
dt_accuracy_array =np.array(dt_accuracy_list)
print("SVM Classifier:")
for i in range(svm_accuracy_array.shape[0]):
    print(f"{i} {svm_accuracy_array[i]}")
print(f"** Mean of SVM classifier is: {np.mean(svm_accuracy_array)}")
print(f"** Std Deviation of SVM classifier is: {np.std(svm_accuracy_array)}")

print("Decision Tree Classifier:")
for i in range(dt_accuracy_array.shape[0]):
    print(f"{i} {dt_accuracy_array[i]}")
print(f"** Mean of Decision Tree classifier is: {np.mean(dt_accuracy_array)}")
print(f"** Std Deviation of Decision Tree classifier is: {np.std(dt_accuracy_array)}")