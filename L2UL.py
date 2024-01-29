import numpy as np 
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

'''
The code for L2UL.
'''
# Isolation kernel
#------------------------------------------------------------------------------------------------
class iNN_IK:
    data = None
    centroid = []
    def __init__(self, psi, t):
        self.psi = psi
        self.t = t

    def fit_transform(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        n, d = self.data.shape
        IDX = np.array([])  #column index
        V = []
        for i in tqdm(range(self.t)):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
            nt_dis = cdist(tdata, self.data)
            centerIdx = np.argmin(nt_dis, axis=0)
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t) #row index
        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

    def transform(self, newdata):
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in tqdm(range(self.t)):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata
#----------------------------------end----------------------------------
    

# data
fea, gt = load_data("data")

'''
Take A=LR as an example
'''



# Preprocessing
#------------------------------------------------------------------------------------------------
psi, t = 4, 100
iNN_ik = iNN_IK(psi, t)
data_map = iNN_ik.fit_transform(fea)

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
#----------------------------------end----------------------------------

# Train A(D)
#------------------------------------------------------------------------------------------------
seed4un_split = 2023
indices = np.arange(fea.shape[0]) 
indices_train, indices_test, y_train, y_test = train_test_split(indices, gt, test_size=0.2, random_state=seed4un_split, stratify=gt)
X_train, X_test = fea[indices_train,:], fea[indices_test,:]
mean_map_ = np.mean(data_map[indices_train,:], axis=0)
clf = LogisticRegressionCV(cv=5, max_iter=500, random_state=2023, scoring="accuracy").fit(X_train, y_train)   
print(clf.score(X_test, y_test))
#----------------------------------end----------------------------------

# D_f
#------------------------------------------------------------------------------------------------
index4un = [i for i in range(1)]
indices = np.arange(X_train.shape[0])
retrain_set = indices[list(range(0, min(index4un))) +list(range(max(index4un)+1, X_train.shape[0]))]
#----------------------------------end----------------------------------

# Baseline Retrain A(D\D_f)
#------------------------------------------------------------------------------------------------
clf_re = LogisticRegressionCV(cv=5, max_iter=500, random_state=2023, scoring="accuracy").fit(X_train[retrain_set,:], y_train[retrain_set]) 
print(clf_re.score(X_test, y_test))
#----------------------------------end----------------------------------


# Train U
#------------------------------------------------------------------------------------------------
_, d_funfea = np.shape(fea_UN)
d_net_in = int((d_funfea-psi*t*2)/2+psi*t*2)
d_net_out = int((d_funfea-psi*t*2)/2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_net_in, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, d_net_out)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
X_un = torch.tensor(fea_UN[:-1,:d_net_in])
y_un = torch.tensor(fea_UN[:-1,d_net_in:])
net = Net()
net.double()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
Loss = []
for epoch in tqdm(range(100)):
    optimizer.zero_grad()
    outputs = net(X_un.double())
    loss = criterion(outputs, y_un.double())
    Loss.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
#----------------------------------end----------------------------------
    

# unlearn D_f using trained U
#------------------------------------------------------------------------------------------------  
X_un_test = []
X_un_test = X_un_test + mean_map_.tolist()[0]
mean_map_un = np.mean(data_map[indices_train[index4un],:], axis=0)
X_un_test = X_un_test + mean_map_un.tolist()[0]
X_un_test = X_un_test + clf.coef_.ravel().tolist()
X_un_test = X_un_test + clf.intercept_.ravel().tolist()
X_un_test = torch.tensor(np.array(X_un_test))
outputs = net(X_un_test.double())
predicted = outputs.data.numpy()
clf_re.coef_ = predicted[:-1].reshape(1,-1) 
clf_re.intercept_ = predicted[-1].reshape(1,-1)
print(clf_re.score(X_test, y_test))
#----------------------------------end----------------------------------