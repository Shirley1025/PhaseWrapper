import os.path
from pydoc import cli
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import skimage.util as utils
from multiprocessing import Pool

def data_gen(data_num,true_file,noisy_warp_file):
    #### hyper parameter
    num_2_generate=data_num
    value_min = 2
    value_max = 30
    mat_size_min=2
    mat_size_max=25
    generate_mat_size = 256
    mean = (value_max+value_min)/2
    sigma = (mean-value_min)/3
    gaussian_noise_std_min=0.01
    gaussian_noise_std_max=0.2
    mult_noise_std_min = 0.01
    mult_noise_std_max = 0.2
    salt_noise_density_min=0.01
    salt_noise_density_max=0.2
    true_phase_file=true_file
    warp_phase_file=noisy_warp_file
    random_mat_size_array = np.random.randint(mat_size_min, mat_size_max + 1,size=(num_2_generate))
    for index,mat_size in enumerate(random_mat_size_array):
        random_mat_size = np.random.randint(mat_size_min,mat_size_max+1)
        distributed_type = np.random.randint(1,3)
        if distributed_type ==1:
        ### unifrom distribution
            random_mat = np.random.uniform(low=value_min,high=value_max,size=(mat_size,mat_size))
        else:
            random_mat = np.random.normal(loc=mean,scale=sigma,size=(mat_size,mat_size))
        interpolate_type = np.random.randint(2,4)
        if interpolate_type == 1:
            ## using nearest interpolate
            true_phase = A.resize(random_mat,generate_mat_size,generate_mat_size,interpolation=cv2.INTER_NEAREST)
        elif interpolate_type == 2:
            true_phase = A.resize(random_mat,generate_mat_size,generate_mat_size,interpolation=cv2.INTER_LINEAR)
        elif interpolate_type == 3:
            true_phase = A.resize(random_mat,generate_mat_size,generate_mat_size,interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError('interpolate_type error')
        warp_phase = np.angle(np.exp(1j*true_phase),deg=False)
        norm_warp_phase = warp_phase/np.pi
        # noise_type = np.random.randint(low=1,high=4)
        noise_type = 1
        # if noise_type==1:
        #     ### gaussian noise
        #     random_std_value = np.random.uniform(low=gaussian_noise_std_min,high=gaussian_noise_std_max)
        #     noisy_norm_warp_phase = utils.random_noise(norm_warp_phase,mode='gaussian',clip=False,var=random_std_value**2)
        # elif noise_type==2:
        #     ### multive noise
        #     random_std_value = np.random.uniform(low=mult_noise_std_min,high=mult_noise_std_max)
        #     noisy_norm_warp_phase = utils.random_noise(norm_warp_phase,mode='speckle',clip=False,var=random_std_value**2)
        # elif noise_type==3:
        #     ### salt pepper noise
        #     noise_amount = np.random.uniform(low=salt_noise_density_min,high=salt_noise_density_max)
        #     noisy_norm_warp_phase = utils.random_noise(norm_warp_phase,mode='s&p',clip=True,amount = noise_amount)
        # else:
        #     raise ValueError()
        # noisy_warp_phase = noisy_norm_warp_phase*np.pi
        true_phase = np.expand_dims(true_phase,axis=0)
        warp_phase = np.expand_dims(warp_phase, axis=0)
        # noisy_warp_phase = np.expand_dims(noisy_warp_phase,axis=0)
        # print(np.max(warp_phase),np.min(warp_phase))
        # print(np.max(noisy_warp_phase),np.min(noisy_warp_phase))
        if index==0:
            true_phase_col = true_phase
            warp_phase_col = warp_phase
        else:
            true_phase_col = np.concatenate((true_phase_col,true_phase),axis=0)
            warp_phase_col = np.concatenate((warp_phase_col,warp_phase),axis=0)
    np.save(true_phase_file,true_phase_col,allow_pickle=True)
    np.save(warp_phase_file,warp_phase_col,allow_pickle=True)


if __name__=='__main__':
    pool =Pool(processes=12)
    data_num=1000
    file_num=10
    for i in range(file_num):
        true_phase_file = os.path.join('data2',f'train_ture_phase_{i}.npy')
        noisy_warp_phase_file = os.path.join('data2',f'train_warp_phase_{i}.npy')
        pool.apply_async(data_gen,args=(data_num,true_phase_file,noisy_warp_phase_file,))
    pool.close()
    pool.join()


