# Jx-MLT : A Machine Learning Toolbox for Classification

[![License](https://img.shields.io/badge/license-BSD_3-blue.svg)](https://github.com/JingweiToo/Machine-Learning-Toolbox-Python/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/badge/release-pre-yellow.svg)](https://github.com/JingweiToo/Machine-Learning-Toolbox-Python)

---
> "Toward Talent Scientist: Sharing and Learning Together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/f9d2bb8c-ebfe-4590-b88c-d4ff92fa6f8f/c4229dd2-aaa5-4146-bafa-4fcccb2b1d30/images/screenshot.PNG) 


## Introduction
* This toolbox contains 6 widely used machine learning algorithms   
* The `Demo_KNN` and `Demo_LDA` provide the examples of how to use these methods on benchmark dataset 


## Usage
You may switch the algorithm by changing the `knn` in `from ML.knn import jkfold` to [other abbreviations](/README.md#list-of-available-machine-learning-methods)   
* If you wish to use linear discriminate analysis ( LDA ) classifier then you may write
```code 
from ML.lda import jkfold 
```

* If you want to use naive bayes ( NB ) classifier then you may write
```code 
from ML.nb import jkfold  
```


## Input
* *`feat`*    : feature vector matrix ( Instance *x* Features )
* *`label`*   : label matrix ( Instance *x* 1 )
* *`opts`*    : parameter settings
  + *`ho`*    : ratio of testing data in hold-out validation
  + *`kfold`* : number of folds in *k*-fold cross-validation

## Output
* *`mdl`* : Machine learning model ( It contains several results )  
  + *`acc`* : classification accuracy 
  + *`con`* : confusion matrix
  + *`r`*   : precision and recall


## How to choose the validation scheme?
There are three types of performance validations. These validation strategies are listed as following ( *KNN* is adopted as an example ). 
  + Hold-out cross-validation
```code 
from ML.knn import jho
```
  + *K*-fold cross-validation
```code 
from ML.knn import jkfold
```
  + Leave-one-out cross-validation
```code 
from ML.knn import jloo
```


### Example 1 : *K*-nearest neighbor ( KNN ) with *k*-fold cross-validation
```code 
import numpy as np
import pandas as pd
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from ML.knn import jkfold 
import matplotlib.pyplot as plt
import seaborn as sns


# load data
data  = pd.read_csv('ionosphere.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# parameters
k     = 5
kfold = 10
opts  = {'k':k, 'kfold':kfold}
# KNN with k-fold
mdl   = jkfold(feat, label, opts) 

# overall accuracy
accuracy = mdl['acc']

# confusion matrix
confmat  = mdl['con']
print(confmat)

# precision & recall
result   = mdl['r']
print(result)


# plot confusion matrix
uni     = np.unique(label)
# Normalise
con     = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
sns.heatmap(con, annot=True, fmt='.2f', xticklabels=uni, yticklabels=uni, cmap="YlGnBu")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('KNN')
plt.show()
```


### Example 2 : Support vector machine  ( SVM ) with hold-out validation
```code 
import numpy as np
from sklearn import datasets
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from ML.svm import jho  
import matplotlib.pyplot as plt
import seaborn as sns


iris  = datasets.load_iris()
feat  = iris.data 
label = iris.target

# parameters
ho     = 0.3    # 30% testing set
kernel = 'rbf'
opts   = {'ho':ho, 'kernel':kernel}
# machine learning
mdl    = jho(feat, label, opts) 

# overall accuracy
accuracy = mdl['acc']

# confusion matrix
confmat  = mdl['con']
print(confmat)

# precision & recall
result   = mdl['r']
print(result)

# plot confusion matrix
uni     = np.unique(label)
# Normalise
con     = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
sns.heatmap(con, annot=True, fmt='.2f', xticklabels=uni, yticklabels=uni, cmap="YlGnBu")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('SVM')
plt.show()
```


### Example 3 : Linear Discriminate Analysis ( LDA ) with leave-one-out validation
```code 
import numpy as np
from sklearn import datasets
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from ML.lda import jloo  
import matplotlib.pyplot as plt
import seaborn as sns


iris  = datasets.load_iris()
feat  = iris.data 
label = iris.target

# parameters
opts  = {}
# machine learning
mdl   = jloo(feat, label, opts) 

# overall accuracy
accuracy = mdl['acc']

# confusion matrix
confmat  = mdl['con']
print(confmat)

# precision & recall
result   = mdl['r']
print(result)

# plot confusion matrix
uni     = np.unique(label)
# Normalise
con     = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
sns.heatmap(con, annot=True, fmt='.2f', xticklabels=uni, yticklabels=uni, cmap="YlGnBu")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('LDA')
plt.show()
```


## Requirement
* Python 3 
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn


## List of available machine learning methods
* Click on the name of algorithm to check the parameters 
* Use the *`opts`* to set the specific parameters  
* If you do not set extra parameters then the algorithm will use default setting in [here](/Description.md)


| No. | Abbreviation | Name                                                                              | Support      |
|-----|--------------|-----------------------------------------------------------------------------------|--------------|
| 06  | `knn`        | [*K*-nearest Neighbor](/Description.md#k-nearest-neighbor-knn)                    | Multi-class  |
| 05  | `svm`        | [Support Vector Machine](/Description.md#support-vector-machine-svm)              | Multi-class  |
| 04  | `dt`         | Decision Tree                                                                     | Multi-class  |
| 03  | `lda`        | Linear Discriminate Analysis                                                      | Multi-class  |
| 02  | `nb`         | Naive Bayes                                                                       | Multi-class  |
| 01  | `rf`         | [Random Forest](Description.md#random-forest-rf)                                  | Multi-class  |               



