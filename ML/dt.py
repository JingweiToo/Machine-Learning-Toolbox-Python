import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def jho(feat, label, opts):
    ho = 0.3   # ratio of testing set
    
    if 'ho' in opts:
        ho = opts['ho']
    
    # number of instances
    num_data = np.size(feat, 0)
    label    = label.reshape(num_data)  # Solve bug
    
    # prepare data
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=ho, stratify=label) 
    # train model
    mdl     = DecisionTreeClassifier(criterion="gini")
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
    
    print("Accuracy (DT_HO):", 100 * acc)
    
    dt = {'acc': acc, 'con': confmat, 'r': report}
    
    return dt
    

def jkfold(feat, label, opts):
    kfold = 10   # number of k in kfold
    
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
        mdl     = DecisionTreeClassifier(criterion="gini")
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
        
    print("Accuracy (DT_K-fold):", 100 * acc)
    
    dt = {'acc': acc, 'con': confmat, 'r': report}
    
    return dt


def jloo(feat, label, opts):
    
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
        mdl     = DecisionTreeClassifier(criterion="gini")
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
        
    print("Accuracy (DT_LOO):", 100 * acc)
    
    dt = {'acc': acc, 'con': confmat, 'r': report}
    
    return dt

    
    
    
    
    