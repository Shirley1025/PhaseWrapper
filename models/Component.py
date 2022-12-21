import torch
import torchvision.transforms
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,in_channel=1,out_channel=2):
        super(DoubleConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return  self.op(x)



class DoubleConv2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DoubleConv2, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,2,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.op(x)

class SingleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(SingleConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.op(x)

class ContinusConv(nn.Module):
    def __init__(self,in_channel,out_channel,repeat_num=0):
        super(ContinusConv, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.stage_conv = nn.ModuleList()
        for i in range(repeat_num):
            self.stage_conv.append(nn.Sequential(
                nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )
        )
    def forward(self,x):
        x = self.basic_conv(x)
        for i in range(len(self.stage_conv)):
            x = self.stage_conv[i](x)
        return x


class Residual(nn.Module):
    def __init__(self,input_channels,out_channels,use_1x1conv=False,strides=1,momentum=0.2):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=strides,bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels,momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels,momentum=momentum)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1,stride=strides,bias=False)
        else:
            self.conv3=None
    def forward(self,x):
        y =  F.relu(self.bn1(self.conv1(x)),inplace=True)
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = y+x
        return F.relu(y,inplace=True)

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,down_sample=False,repeat_num=1):
        super(ResidualBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(repeat_num):
            if i==0:
                if down_sample:
                    self.blocks.append(Residual(in_ch,out_ch,use_1x1conv=True,strides=2))
                else:
                    self.blocks.append(Residual(in_ch,out_ch,use_1x1conv=True,strides=1))
            else:
                self.blocks.append((Residual(out_ch,out_ch,use_1x1conv=False,strides=1)))
    def forward(self,x):
        for op in self.blocks:
            x = op(x)
        return x