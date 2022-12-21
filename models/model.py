import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import math
import torch.nn.functional as F
from models.Component import *




class ResUNET(nn.Module):
    def __init__(self,in_channel=2,out_channel=1,encode_list=[[32,False,2],[64,True,2],[128,True,2],[256,True,2]]):
        super(ResUNET, self).__init__()
        self.encoder = nn.ModuleList()
        in_ch =in_channel
        for encode in encode_list:
            self.encoder.append((ResidualBlock(in_ch,out_ch=encode[0],down_sample=encode[1],repeat_num=encode[2])))
            in_ch = encode[0]
        self.bottleneck=ResidualBlock(in_ch=in_ch,out_ch=in_ch*2,down_sample=True,repeat_num=2)
        self.upsample = nn.ModuleList()
        for i in range(len(encode_list)):
            self.upsample.append(nn.PixelShuffle(2))
        self.decoder = nn.ModuleList()
        for i in reversed(encode_list):
            self.decoder.append(ResidualBlock(in_ch=i[0]+i[0]//2,out_ch=i[0],down_sample=False,repeat_num=i[2]))
        self.final_conv = nn.Conv2d(in_channels=encode_list[0][0],out_channels=out_channel,kernel_size=1,stride=1,padding=0,bias=False)
    def forward(self,x):
        skip_connections = []
        for trans in self.encoder:
            x = trans(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        for index,i in enumerate(reversed(skip_connections)):
            x = self.upsample[index](x)
            x = torch.cat((x,i),dim=1)
            x = self.decoder[index](x)
        x = self.final_conv(x)
        return x


class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class DebulrFPN(nn.Module):
    def __init__(self,in_channel=2,out_channel=1,features=32,num_filter=128,num_mid=64,repeat_num=2):
        super(DebulrFPN, self).__init__()
        self.fpn = FPN(in_channel,features,num_filter,repeat_num)
        self.head1 = FPNHead(num_filter, num_mid, num_mid)
        self.head2 = FPNHead(num_filter, num_mid, num_mid)
        self.head3 = FPNHead(num_filter, num_mid, num_mid)
        self.head4 = FPNHead(num_filter, num_mid, num_mid)
        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_mid,num_filter, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filter//2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filter // 2, out_channel, kernel_size=3, padding=1)


    def forward(self,x):
        map0,map1,map2,map3,map4 = self.fpn(x)
        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        x = self.final(smoothed)
        return x


class FPN(nn.Module):
    def __init__(self,in_channel=2,features=32,num_filter=128,repeat_num=2):
        super(FPN, self).__init__()
        self.features=features
        self.repeat_num=repeat_num
        in_ch=in_channel
        self.encoder0 = ResidualBlock(in_ch=in_channel,out_ch=features,down_sample=False,repeat_num=repeat_num)
        self.encoder1 = ResidualBlock(in_ch=features,out_ch=features*2,down_sample=True,repeat_num=repeat_num)
        self.encoder2 = ResidualBlock(in_ch=features*2,out_ch=features*4,down_sample=True,repeat_num=repeat_num)
        self.encoder3 = ResidualBlock(in_ch=features*4,out_ch=features*8,down_sample=True,repeat_num=repeat_num)
        self.encoder4 = ResidualBlock(in_ch=features*8,out_ch=features*16,down_sample=True,repeat_num=repeat_num)
        self.lateral4 = nn.Conv2d(self.features*16,num_filter,kernel_size=1,stride=1,bias=False)
        self.lateral3 = nn.Conv2d(self.features * 8, num_filter, kernel_size=1, stride=1, bias=False)
        self.lateral2 = nn.Conv2d(self.features * 4, num_filter, kernel_size=1, stride=1, bias=False)
        self.lateral1 = nn.Conv2d(self.features * 2, num_filter, kernel_size=1, stride=1, bias=False)
        self.lateral0 = nn.Conv2d(self.features * 1, num_filter, kernel_size=1, stride=1, bias=False)
        self.td3 = nn.Sequential(
            nn.Conv2d(num_filter,num_filter,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
            )
        self.td2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )
        self.td1 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )
    def _upsample_add(self,x,y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y
    def forward(self,x):
        en0 = self.encoder0(x)
        en1 = self.encoder1(en0)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        lateral4 = self.lateral4(en4)
        lateral3 = self.lateral3(en3)
        lateral2 = self.lateral2(en2)
        lateral1 = self.lateral1(en1)
        lateral0 = self.lateral0(en0)
        de3 = self.td3(self._upsample_add(lateral4,lateral3))
        de2 = self.td2(self._upsample_add(de3,lateral2))
        de1 = self.td1(self._upsample_add(de2,lateral1))
        return lateral0,de1,de2,de3,lateral4

if __name__=='__main__':
    x = torch.rand(size=(1,1,320,320))
    model = DebulrFPN(in_channel=1)
    x = model(x)
    print(x.shape)