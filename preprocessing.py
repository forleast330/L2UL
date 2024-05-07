"""
preprocessing for L2UL.
Take LR as an example.
"""

# Authors: L2UL



from isolation_distribution_kernel import iNN_IK
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


def preprocessing(psi, t, fea, gt): 
    """ Get the data for training U.
        ----------
        psi : parameter for IDK
        t   : parameter for IDK
        fea : array-like of shape (n_samples, n_features)
            Input data.
        gt : array-like of shape (n_samples, 1)
            Input label.

        Returns
        -------
        fea_UN : ndarray of shape (n_sample_un, psi*t+theta)
                 The data for traing U.
        data_map: ndarray of shape (n_samples, psi*t)
                  The feature map of data.
        mean_map_: ndarray of shape (1, psi*t)
                   The mean feature map of data.
    """   
    iNN_ik = iNN_IK(psi, t)
    data_map = iNN_ik.fit_transform(fea)
    mean_map_ = np.mean(data_map[indices_train,:], axis=0)
    n_sample_un = 1000         
    train_size = 0.8       
    print(train_size)
    Data_UN = []
    for i_sample_un in tqdm(range(n_sample_un)):
        data_un = []
        number_train = 1000
        indices = np.arange(number_train)
        np.random.seed(2023+i_sample_un)
        index_sample = np.random.choice(fea.shape[0], number_train, replace=False)
        fea_sample = fea[index_sample,:]
        gt_sample = gt[index_sample]
        indices_train, indices_test, y_train, y_test = train_test_split(indices, gt_sample, test_size=1-train_size, random_state=i_sample_un, stratify=gt_sample)
        X_train, X_test = fea_sample[indices_train,:], fea_sample[indices_test,:]
        data_un.append(indices_train)
        clf = LogisticRegressionCV(cv=5, max_iter=500, random_state=2023).fit(X_train, y_train)    
        data_un.append([clf.coef_, clf.intercept_])
        clf_un = LogisticRegressionCV(cv=5, max_iter=500, random_state=2023).fit(X_train[1:,:], y_train[1:])
        data_un.append([clf_un.coef_, clf_un.intercept_])
        Data_UN.append(data_un)
    n_sample_un = 1000 
    fea_UN = []
    for i in tqdm(range(n_sample_un)):
        fea_i = []
        item_i = Data_UN[i]
        indices_i = item_i[0]
        mean_map_i = np.mean(data_map[indices_i,:], axis=0)
        fea_i = fea_i + mean_map_i.tolist()[0]
        mean_map_i_un = np.mean(data_map[indices_i[0],:], axis=0)
        fea_i = fea_i + mean_map_i_un.tolist()[0]
        para_i = item_i[1]
        para_i_1 = para_i[0].ravel().tolist()
        fea_i = fea_i + para_i_1
        para_i_2 = para_i[1].ravel().tolist()
        fea_i = fea_i + para_i_2
        para_i_un = item_i[2]
        para_i_un_1 = para_i_un[0].ravel().tolist()
        fea_i = fea_i + para_i_un_1
        para_i_un_2 = para_i_un[1].ravel().tolist()
        fea_i = fea_i + para_i_un_2   
        fea_UN.append(fea_i)
    fea_UN = np.array(fea_UN) 
    return fea_UN, data_map, mean_map_