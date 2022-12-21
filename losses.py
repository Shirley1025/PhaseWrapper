import torch
from pytorch_ssim import *
import torchvision


class L1_loss(torch.nn.Module):
    def __init__(self, factor=1, **kwargs):
        super(L1_loss, self).__init__()
        self.factor = factor
        self.loss = torch.nn.L1Loss(**kwargs)

    def forward(self, data, label):
        return self.factor * self.loss(data, label)


class MSE_loss(torch.nn.Module):
    def __init__(self, factor, **kwargs):
        super(MSE_loss, self).__init__()
        self.factor = factor
        self.loss = torch.nn.MSELoss(**kwargs)

    def forward(self, data, label):
        return self.factor * self.loss(data, label)


class SSIM_loss(torch.nn.Module):
    def __init__(self, factor):
        super(SSIM_loss, self).__init__()
        self.factor = factor
        self.ssim = SSIM()

    def forward(self, data, label):
        return self.factor * (1 - self.ssim(data, label))


class VGGloss(torch.nn.Module):
    def __init__(self, factor=1):
        super(VGGloss, self).__init__()
        self.net = torchvision.models.vgg19(pretrained=True).features[:35].eval().cuda()
        for param in self.net.parameters():
            param.requires_grad = False
        self.loss = torch.nn.MSELoss()
        self.factor = factor

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1).cuda()
            y = y.repeat(1, 3, 1, 1).cuda()
        x_features = self.net(x)
        y_features = self.net(y)
        loss = self.loss(x, y)
        return self.factor * loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_list: list):
        super(CombinedLoss, self).__init__()
        self.loss_list = loss_list

    def forward(self, data, label):
        if len(self.loss_list) == 1:
            total_loss = self.loss_list[0](data, label)
        elif len(self.loss_list) == 2:
            total_loss = self.loss_list[0](data, label) + self.loss_list[1](data, label)
        elif len(self.loss_list) == 3:
            total_loss = self.loss_list[0](data, label) + self.loss_list[1](data, label) + self.loss_list[2](data,
                                                                                                             label)
        return total_loss


def get_loss(name: str, factor):
    if name == 'MSE_loss':
        return MSE_loss(factor)
    elif name == 'L1_loss':
        return L1_loss(factor)
    elif name == 'SSIM_loss':
        return SSIM_loss(factor)
    elif name == 'VGG_loss':
        return VGGloss(factor)
    else:
        return ValueError(f'loss {name} not implemented')
