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
