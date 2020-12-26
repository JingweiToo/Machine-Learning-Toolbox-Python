import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def jho(feat, label, opts):
    k  = 5
    ho = 0.3   # ratio of testing set
    
    if 'k' in opts:
        k  = opts['k']
    if 'ho' in opts:
        ho = opts['ho']
    
    # number of instances
    num_data = np.size(feat, 0)
    label    = label.reshape(num_data)  # Solve bug
    
    # prepare data
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=ho, stratify=label) 
    
    # train model
    mdl     = KNeighborsClassifier(n_neighbors = k)
    mdl.fit(xtrain, ytrain)
    # prediction
    ypred   = mdl.predict(xtest)
    
    # confusion matric
    uni     = np.unique(ytest)
    confmat = confusion_matrix(ytest, ypred, labels=uni)
    # report
    report  = classification_report(ytest, ypred)
    # accuracy
    acc     = np.sum(ytest == ypred) / np.size(xtest,0)
    
    print("Accuracy (KNN_HO):", 100 * acc)
    
    knn = {'acc': acc, 'con': confmat, 'r': report}
    
    return knn
    

def jkfold(feat, label, opts):
    k     = 5
    kfold = 10   # number of k in kfold
    
    if 'k' in opts:
        k     = opts['k']
    if 'kfold' in opts:
        kfold = opts['kfold']
    
    # number of instances
    num_data = np.size(feat, 0)
    # define selected features
    x_data   = feat
    y_data   = label.reshape(num_data)  # Solve bug
    
    fold     = StratifiedKFold(n_splits = kfold)
    fold.get_n_splits(x_data, y_data)
    
    ytest2 = []
    ypred2 = []
    Afold2 = []
    for train_idx, test_idx in fold.split(x_data, y_data):
        xtrain  = x_data[train_idx,:] 
        ytrain  = y_data[train_idx]
        xtest   = x_data[test_idx,:]
        ytest   = y_data[test_idx]
        # train model
        mdl     = KNeighborsClassifier(n_neighbors = k)
        mdl.fit(xtrain, ytrain)
        # prediction
        ypred   = mdl.predict(xtest)
        # accuracy
        Afold   = np.sum(ytest == ypred) / np.size(xtest,0)
        
        ytest2  = np.concatenate((ytest2, ytest), axis=0)
        ypred2  = np.concatenate((ypred2, ypred), axis=0)
        Afold2.append(Afold) 
    
    # average accuracy
    Afold2  = np.array(Afold2)
    acc     = np.mean(Afold2)
    # confusion matric
    uni     = np.unique(ytest2)
    confmat = confusion_matrix(ytest2, ypred2, labels=uni)
    # report
    report  = classification_report(ytest2, ypred2)
        
    print("Accuracy (KNN_K-fold):", 100 * acc)
    
    knn = {'acc': acc, 'con': confmat, 'r': report}
    
    return knn


def jloo(feat, label, opts):
    k = 5
    
    if 'k' in opts:
        k     = opts['k']
    
    # number of instances
    num_data = np.size(feat, 0)
    # define selected features
    x_data   = feat
    y_data   = label.reshape(num_data)  # Solve bug
 
    loo      = LeaveOneOut()
    loo.get_n_splits(x_data)
    
    ytest2 = []
    ypred2 = []
    Afold2 = []
    for train_idx, test_idx in loo.split(x_data):
        xtrain = x_data[train_idx,:] 
        ytrain = y_data[train_idx]
        xtest  = x_data[test_idx,:]
        ytest  = y_data[test_idx]
        # train model
        mdl     = KNeighborsClassifier(n_neighbors = k)
        mdl.fit(xtrain, ytrain)
        # prediction
        ypred   = mdl.predict(xtest)
        # accuracy
        Afold   = np.sum(ytest == ypred) / np.size(xtest,0)
        
        ytest2  = np.concatenate((ytest2, ytest), axis=0)
        ypred2  = np.concatenate((ypred2, ypred), axis=0)
        Afold2.append(Afold) 
    
    # average accuracy
    Afold2  = np.array(Afold2)
    acc     = np.mean(Afold2)
    # confusion matric
    uni     = np.unique(ytest2)
    confmat = confusion_matrix(ytest2, ypred2, labels=uni)
    # report
    report  = classification_report(ytest2, ypred2)
        
    print("Accuracy (KNN_LOO):", 100 * acc)
    
    knn = {'acc': acc, 'con': confmat, 'r': report}
    
    return knn

    
    
    
    
    