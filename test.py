import torch
from models.model import *
from dataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
phase_dataset = PhaseUnwarpDataset_withtrans(true_phase_dir= '../PhaseUnwarp_datagen/np_data/train30000/train_true_crop_phase_30000.npy',
                                             trans_phase_dir='../PhaseUnwarp_datagen/np_data/train30000/train_trans_phase_add_ground_rorate.npy')
checkpoint = torch.load('')
print(type(checkpoint))
print(checkpoint.keys())
model = ResUNET(in_channel=4)
model.load_state_dict(checkpoint['state_dict'])

data,label = phase_dataset.__getitem__(300)
data = np.transpose(data,axes=[1,2,0])
trans = torchvision.transforms.ToTensor()
data =trans(data)
data = torch.unsqueeze(data,dim=0)
model.eval()
with torch.no_grad():
    pre_label = model(data)

plt.subplot(1,3,1)
plt.imshow(pre_label.numpy()[0][0])
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(label[0])
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(pre_label.numpy()[0][0]-label[0])
plt.colorbar()
plt.show()

img2save = pre_label.numpy()[0][0]
np.save('test_sample300.npy',img2save,allow_pickle=True)