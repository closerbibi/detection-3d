import scipy.io as sio
import numpy as np
import os, pdb

path = '/home/closerbibi/workspace/caffe-repo/Amodal3Det/dataset/NYUV2/annotations/'

lst = sorted(os.listdir(path))

for i in lst:
    f = sio.loadmat(path+i)
    pdb.set_trace()
