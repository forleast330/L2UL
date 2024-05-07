"""
train U.
"""

# Authors: L2UL

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train_u(fea_UN, psi, t):
    """ Train U.
        ----------
        fea : array-like of shape (n_samples, n_features)
            Input data.
        psi : parameter for IDK
        t   : parameter for IDK


        Returns
        -------
        net : The trained U.
    """   
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
    return net