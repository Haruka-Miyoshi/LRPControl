import torch.nn as nn

class Model(nn.Module):
    def __init__(self, i_dim, o_dim):
        super(Model, self).__init__()
        self.__linear=nn.Sequential(
        nn.ReLU(True),
        nn.Linear(i_dim, o_dim),
        nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.__linear(x)