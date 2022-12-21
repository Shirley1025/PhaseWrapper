import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision


class PhaseUnwarpDataset(Dataset):
    def __init__(self, true_phase_dir, warp_phase_dir, transform=None, target_transform=None):
        super(PhaseUnwarpDataset, self).__init__()
        self.true_phases = np.load(true_phase_dir, allow_pickle=True, mmap_mode='r+')
        self.warp_phases = np.load(warp_phase_dir, allow_pickle=True, mmap_mode='r+')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.true_phases)

    def __getitem__(self, index):
        true_phase = self.true_phases[index]
        warp_phase = self.warp_phases[index]
        data = np.expand_dims(warp_phase, axis=0).astype(np.float32)
        label = np.expand_dims(true_phase, axis=0).astype(np.float32)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


class PhaseUnwarpDataset_withtrans(Dataset):
    def __init__(self, true_phase_dir, trans_phase_dir, transform=None, target_transform=None):
        super(PhaseUnwarpDataset_withtrans, self).__init__()
        self.true_phases = np.load(true_phase_dir, allow_pickle=True, mmap_mode='r+')
        self.trans_phases = np.load(trans_phase_dir, allow_pickle=True, mmap_mode='r+')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.true_phases)

    def __getitem__(self, index):
        true_phase = self.true_phases[index]
        trans_phase = self.trans_phases[index]
        data = trans_phase.astype(np.float32)
        label = np.expand_dims(true_phase, axis=0).astype(np.float32)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


if  __name__=='__main__':
    dataset = PhaseUnwarpDataset_withtrans(
        true_phase_dir='../PhaseUnwarp/np_data/test_true_crop_phase.npy',
        trans_phase_dir='../PhaseUnwarp/np_data/test_trans_phase_2.npy'
    )
    data,label = dataset.__getitem__(100)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    print(data.shape)
    print(label.shape)

