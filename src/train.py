import os
import torch
import numpy as np
from lrpcontorl import LRPControl
import matplotlib.pyplot as plt

if not os.path.exists('./figs'):
    os.mkdir('./figs')

x=np.loadtxt('./data/e.txt')
y=np.loadtxt('./data/v.txt')

model=LRPControl(2,2)
model.fit(torch.from_numpy(x), torch.from_numpy(y), mode=True)
w1, b = model.get_params()

x_h=np.array([-100, 100])
y_h=w1*x_h+b

plt.plot(x_h, y_h, "r")
plt.title('fitting')
plt.xlabel('error')
plt.ylabel('v')
plt.scatter(x, y)
plt.savefig('./figs/fit.png')
plt.show()