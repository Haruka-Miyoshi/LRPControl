import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('./figs'):
    os.mkdir('./figs')

e=np.loadtxt('./data/e.txt')
v=np.loadtxt('./data/v.txt')

plt.title('fitting')
plt.xlabel('error')
plt.ylabel('v')
plt.scatter(e, v)
plt.savefig('./figs/data.png')
plt.show()