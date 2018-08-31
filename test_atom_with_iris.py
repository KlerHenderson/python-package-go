#check versions
import sys
print('Python: {}'.format(sys.version))

import scipy
print('scipy: {}'.format(scipy.__version__))

import numpy
print('numpy: {}'.format(numpy.__version__))

import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

import pandas
print('pandas: {}'.format(pandas.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))
#load libraries
import pandas
from pandas import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#load dataset
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pandas.read_csv(url,names=names)
#summarize dataset
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('class').size())

scatter_matrix(dataset)
plt.show()
#split out validation dataset for analysis
array=dataset.values
x=array[:,0:4]
y=array[:,4]
validation_size=0.20
seed=7
x_train, x_validation, y_train, y_validation=model_selection.train_test_split(x,y,test_size=validation_size, random_state=seed)
#test options and evaluate metrics
seed=7
scoring='accuracy'
#spot check algoritms
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
#evaluate each model in turn
results=[]
names=[]
for name, model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#make predictions on validated (max value) dataset
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions=knn.predict(x_validation)
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
