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