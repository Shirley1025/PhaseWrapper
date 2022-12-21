import numpy as np
import os
for index in range(2):
    temp = np.load(f'data2/test_warp_phase_{index}.npy',mmap_mode='r+',allow_pickle=True)
    if index==0:
        res = temp
    else:
        res = np.concatenate((res,temp),axis=0)
    print(res.shape)
np.save('test_warp_phase.npy',res,allow_pickle=True)