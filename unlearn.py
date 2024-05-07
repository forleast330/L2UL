"""
unlearn.
"""

# Authors: L2UL

import torch
import torch.nn as nn
import numpy as np

# unlearn D_f using trained U
def unlearn(net, mean_map_, mean_map_un, clf):
    """ unlearn.
        ----------
        net : the trained U
        mean_map_: ndarray of shape (1, psi*t)
                   The mean feature map of data.
        mean_map_: ndarray of shape (1, psi*t)
                   The mean feature map of unlearn data.
        clf : the model need to be unlearned

        Returns
        -------
        clf : The unlearned model.
    """
    X_un_test = []
    X_un_test = X_un_test + mean_map_.tolist()[0]
    X_un_test = X_un_test + mean_map_un.tolist()[0]
    X_un_test = X_un_test + clf.coef_.ravel().tolist()
    X_un_test = X_un_test + clf.intercept_.ravel().tolist()
    X_un_test = torch.tensor(np.array(X_un_test))
    
    outputs = net(X_un_test.double())
    predicted = outputs.data.numpy()
    clf.coef_ = predicted[:-1].reshape(1,-1) 
    clf.intercept_ = predicted[-1].reshape(1,-1)
    return clf