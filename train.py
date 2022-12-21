import numpy as np
import torch
import yaml
import os
import dataset
from torch.utils.data import DataLoader, Dataset
import models.model
import logging
import shutil
from torch.utils.tensorboard import SummaryWriter
from losses import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import math
import tqdm
import torchvision
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.config_path = config_path
        self._init_tensorboard_writter()
        self._init_logger()
        self._init_dataset()
        self._init_model()
        self._init_optim()
        self._init_loss()

    def _init_dataset(self):
        dataset_class = getattr(dataset, self.config['dataset']['type'])  # getattr()函数用于返回一个对象属性值
        self.train_dataset = dataset_class(**config['dataset']['train_args'])
        self.test_dataset = dataset_class(**config['dataset']['test_args'])
        self.train_dataloader = DataLoader(self.train_dataset, **config['dataloader']['train_args'])

        self.test_dataloader = DataLoader(self.test_dataset, **config['dataloader']['test_args'])

    def _init_optim(self):
        optim_class = getattr(torch.optim, self.config['optimizer']['type'])
        self.optim = optim_class(self.model.parameters(), **self.config['optimizer']['args'])

    def _init_model(self):
        model_class = getattr(models.model, self.config['model']['type'])
        self.model = model_class(**self.config['model']['args']).cuda()
        if self.config['model']['parallel']:
            self.model = torch.nn.DataParallel(self.model)

    def _init_logger(self):
        log_path = os.path.join(self.config['tensorboard_log'], self.config['config_save_file'])
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
        with open(self.config_path) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                logging.info(line)

    def _init_loss(self):
        loss_fn_list = []
        for i in self.config['loss']:
            loss_fn_list.append(get_loss(name=i['name'], factor=i['factor']))
        self.loss_fn = CombinedLoss(loss_fn_list)

    def _init_tensorboard_writter(self):
        log_path = self.config['tensorboard_log']
        if os.path.exists(log_path):
            print(f'tensorboard log dir{log_path} is existed')
            is_del = input('delete existed dir(Y/N)?:')
            if is_del == 'Y':
                shutil.rmtree(log_path)
        else:
            print('path not existed')
        self.writter = SummaryWriter(log_path)

    def train_one_epoch(self, cur_epoch: int):
        self.model.train()
        loop = tqdm.tqdm(self.train_dataloader)
        loop.set_description(f'{cur_epoch}/{self.config["train"]["epoch"]}')
        for index, (data, label) in enumerate(loop):
            data, label = data.cuda, label.cuda
            pre_label = self.model(data)
            loss_item = self.loss_fn(pre_label, label)
            self.optim.zero_grad()
            loss_item.backward()
            self.optim.step()
            loop.set_postfix_str(f'loss is {loss_item}')

    def check_metric(self, dataloader: DataLoader):
        self.model.eval()
        sample_num = 0
        total_ssim = 0
        total_psnr = 0
        total_nmse = 0
        device = 'cpu'
        with torch.no_grad():
            for index, (data, label) in enumerate(dataloader):
                data, label = data.to(device), label.to(device)
                pre_label = self.model(data)
                if index == 0:
                    vis_data_img = data
                    vis_label_img = label
                    vis_pre_label_img = pre_label
                pre_label, label = pre_label.cpu().numpy(), label.cpu().numpy()
                # self.save_fig(data[0][0],label[0][0],filename='dataandlabel.png')
                sample_num += pre_label.shape[0]
                for i in range(pre_label.shape[0]):
                    temp_label = np.squeeze(label[i])
                    temp_pre_label = np.squeeze(pre_label[i])
                    total_ssim += ssim(temp_label, temp_pre_label, multichannel=False)
                    total_psnr += psnr(temp_label, temp_pre_label, data_range=100)
                    total_nmse += math.sqrt(mse(temp_label, temp_pre_label))
        return total_ssim / sample_num, total_psnr / sample_num, total_nmse / sample_num, vis_data_img, vis_label_img, vis_pre_label_img

    def train(self):
        best_ssim = 0
        NUM_EPOCH = self.config['train']['epoch']
        for epoch in range(NUM_EPOCH):
            self.train_one_epoch(epoch)
            if (epoch + 1) % 5 == 0:
                train_ssim, train_psnr, train_nmse, train_data, train_label, train_pre_label = self.check_metric(
                    self.train_dataloader)
                test_ssim, test_psnr, test_nmse, test_data, test_label, test_pre_label = self.check_metric(
                    self.test_dataloader)
                print(
                    f'{epoch}/{NUM_EPOCH} train ssim is {train_ssim},train psnr is {train_psnr},train nmse is {train_nmse}')
                print(f'{epoch}/{NUM_EPOCH} test ssim is {test_ssim},test psnr is {test_psnr},test nmse is {test_nmse}')
                if test_ssim > best_ssim:
                    checkpoint = {'state_dict': self.model.state_dict()}
                    torch.save(checkpoint, str(self.config['checkpoint_file']))
                self.writter.add_scalar('train_ssim', train_ssim, global_step=epoch)
                self.writter.add_scalar('train_psnr', train_psnr, global_step=epoch)
                self.writter.add_scalar('train_nmse', train_nmse, global_step=epoch)
                self.writter.add_scalar('test_ssim', test_ssim, global_step=epoch)
                self.writter.add_scalar('test_psnr', test_psnr, global_step=epoch)
                self.writter.add_scalar('test_nmse', test_nmse, global_step=epoch)
                train_data_img = torchvision.utils.make_grid(train_data[:16])
                train_label_img = torchvision.utils.make_grid(train_label[:16])
                train_pre_label_img = torchvision.utils.make_grid(train_pre_label[:16])
                test_data_img = torchvision.utils.make_grid(test_data[:16])
                test_label_img = torchvision.utils.make_grid(test_label[:16])
                test_pre_label_img = torchvision.utils.make_grid(test_pre_label[:16])
                self.writter.add_image('train data', train_data_img, global_step=epoch)
                self.writter.add_image('train label', train_label_img, global_step=epoch)
                self.writter.add_image('train pre label', train_pre_label_img, global_step=epoch)
                self.writter.add_image('test data', test_data_img, global_step=epoch)
                self.writter.add_image('test label', test_label_img, global_step=epoch)
                self.writter.add_image('test pre label', test_pre_label_img, global_step=epoch)

    def save_fig(self, img1: np.ndarray, img2: np.ndarray, filename: str):
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap='binary')
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='binary')
        plt.show()
        plt.savefig(filename)


if __name__ == '__main__':
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(config['metric'])
    trainer = Trainer(config=config, config_path=config_path)
    trainer.train()
