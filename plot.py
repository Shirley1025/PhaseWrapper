import numpy as np
import  matplotlib.pyplot as plt
true_phase = np.load('./train_true_phase.npy',allow_pickle=True)
warp_phase = np.load('./train_noisy_warp_phase.npy',allow_pickle=True)
index=1250
plt.subplot(1,2,1)
plt.imshow(true_phase[index])
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(warp_phase[index])
plt.colorbar()
plt.show()