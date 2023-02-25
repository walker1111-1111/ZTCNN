import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from coordConv import addCoords, addCoords_1D
# import gpytorch
import math

cwd = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Solenoid_STCNN(nn.Module):
    def __init__(self):
        super(Solenoid_STCNN, self).__init__()

        # 全连接层
        self.add_coords = addCoords_1D()
        self.lin1 = nn.Linear(8, 25)
        self.lin2 = nn.Linear(25, 25)
        self.lin3 = nn.Linear(25, 25)
        self.lin4 = nn.Linear(25, 25)
        self.norm_fc = nn.BatchNorm1d(25)
        # 解码器
        self.tconv1 = nn.ConvTranspose1d(25, 10, 31, 1, 0)
        self.norm_dec1 = nn.BatchNorm1d(10)

        self.tconv2 = nn.ConvTranspose1d(10, 10, 31, 2, 0)
        self.norm_dec2 = nn.BatchNorm1d(10)

        self.tconv3 = nn.ConvTranspose1d(10, 10, 16, 2, 0)
        self.norm_dec3 = nn.BatchNorm1d(10)

        self.tconv4 = nn.ConvTranspose1d(11, 2, 21, 1, 10)

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = latent.tanh()

        latent = self.lin2(latent)
        latent = latent.tanh()

        latent = self.lin3(latent)
        latent = latent.tanh()

        latent = self.lin4(latent)
        z = (self.norm_fc(latent.view(-1, self.lin4.out_features, 1))).tanh()

        return z

    def transposed_conv(self, z):
        latent = self.tconv1(z)
        latent = latent.tanh()
        latent = self.norm_dec1(latent)

        latent = self.tconv2(latent)
        latent = latent.tanh()
        latent = self.norm_dec2(latent)

        latent = self.tconv3(latent)
        latent = latent.tanh()
        latent = self.norm_dec3(latent)

        latent = self.add_coords(latent)
        recons_y = self.tconv4(latent)
        return recons_y

    def forward(self, x):
        z = self.fully_connected(x)
        out = self.transposed_conv(z)
        return out


class Solenoid_STCNN_V2(nn.Module):
    def __init__(self):
        super(Solenoid_STCNN_V2, self).__init__()

        # 全连接层
        self.add_coords = addCoords_1D()
        self.lin1 = nn.Linear(8, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 30)
        self.lin4 = nn.Linear(30, 30)

        # 转置卷积层
        self.tconv1 = nn.ConvTranspose1d(30, 25, 31, 1, 0)  # 输入30维，输出25维，卷积核大小为31，stride为1，padding为0
        self.norm_dec1 = nn.BatchNorm1d(25)

        self.tconv2 = nn.ConvTranspose1d(25, 25, 4, 2, 1)
        self.norm_dec2 = nn.BatchNorm1d(25)

        self.tconv3 = nn.ConvTranspose1d(25, 25, 4, 2, 1)
        self.norm_dec3 = nn.BatchNorm1d(25)

        self.tconv4 = nn.ConvTranspose1d(25, 25, 4, 2, 1)
        self.norm_dec4 = nn.BatchNorm1d(25)

        self.tconv5 = nn.ConvTranspose1d(25, 2, 4, 1, 0)

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = F.elu(latent)

        latent = self.lin2(latent)
        latent = F.elu(latent)

        latent = self.lin3(latent)
        latent = F.elu(latent)

        latent = self.lin4(latent)
        latent = F.elu(latent)
        z = latent.view(-1, self.lin4.out_features, 1)
        return z

    def transposed_conv(self, z):
        latent = self.tconv1(z)
        latent = F.elu(latent)

        latent = self.tconv2(latent)
        latent = F.elu(latent)

        latent = self.tconv3(latent)
        latent = F.elu(latent)

        latent = self.tconv4(latent)
        latent = F.elu(latent)

        recons_y = self.tconv5(latent)
        return recons_y[..., :196]

    def forward(self, x):
        z = self.fully_connected(x)
        out = self.transposed_conv(z)
        return out


class ZTCNN(nn.Module):
    """
        Z-TCNN转置卷积模型
    """

    def __init__(self):
        super(ZTCNN, self).__init__()

        self.activation = nn.Tanh()  # 激活函数
        self.add_coords = addCoords_1D()

        # ZTCNN
        # 全连接层
        self.lin1 = nn.Linear(3, 25)
        self.lin2 = nn.Linear(25, 25)
        self.lin3 = nn.Linear(25, 25)
        self.lin4 = nn.Linear(25, 25)
        self.norm_fc = nn.BatchNorm1d(25)
        # 转置卷积，(in_channel, out_channel, kernel, stride, padding)
        # Lout = (Lin-1)*s -2p + k
        self.tconv1 = nn.ConvTranspose1d(1, 10, 31, 3, 0)  # 25->103
        self.norm_dec1 = nn.BatchNorm1d(10)
        self.tconv2 = nn.ConvTranspose1d(10, 10, 31, 2, 0)  # 103->235
        self.norm_dec2 = nn.BatchNorm1d(10)
        self.tconv3 = nn.ConvTranspose1d(10, 10, 17, 2, 0)  # 235->485
        self.norm_dec3 = nn.BatchNorm1d(10)
        self.tconv4 = nn.ConvTranspose1d(10, 10, 16, 1, 0)  # 485->500
        self.norm_dec4 = nn.BatchNorm1d(10)
        # ZTCNN加入coordconv层后
        self.tconv5 = nn.ConvTranspose1d(11, 1, 21, 1, 10)  # 500->500

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = self.activation(latent)

        latent = self.lin2(latent)
        latent = self.activation(latent)

        latent = self.lin3(latent)
        latent = self.activation(latent)

        latent = self.lin4(latent)
        latent = self.norm_fc(latent)
        latent = self.activation(latent)

        # (N,25)->(N,1,25)  (N,5)->(N,1,5)
        z = latent.unsqueeze(1)
        return z

    def transposed_conv(self, z):
        # (N, Channel, Length)
        latent = self.tconv1(z)
        latent = self.norm_dec1(latent)
        latent = self.activation(latent)

        latent = self.tconv2(latent)
        latent = self.norm_dec2(latent)
        latent = self.activation(latent)

        latent = self.tconv3(latent)
        latent = self.norm_dec3(latent)
        latent = self.activation(latent)

        latent = self.tconv4(latent)
        latent = self.norm_dec4(latent)
        latent = self.activation(latent)

        # 加入coordconv层
        latent = self.add_coords(latent)  # (,11,500)
        latent = self.tconv5(latent)  # (,1,500)
        latent = latent.view(latent.size(0), -1)  # (, 500)
        return latent

    def forward(self, x):
        # ZTCNN提供细节
        z = self.fully_connected(x)
        ztcnn_out = self.transposed_conv(z)
        return ztcnn_out
