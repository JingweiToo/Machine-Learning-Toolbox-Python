# Detail Parameter Settings / Default Setting

## *K*-nearest Neighbor (KNN) 
* Number of k-value
```code 
k    = 5
opts = {'k':k, 'kfold':kfold}
```


## Support Vector Machine (SVM)
* Selection of kernel function. You may choose one
```code 
kernel = 'rbf'      # radial basis function
kernel = 'poly'     # polynomial
kernel = 'linear'   # linear
opts   = {'kernel':kernel, 'kfold':kfold}
```


## Random Forest (RF)
* Number of trees
```code 
nTree = 100  
opts  = {'nTree':nTree, 'kfold':kfold}
```





