'''
EMOTIC CNN: Baseline Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import params
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

''' Model '''
class _NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension: if dimension == 2 : size = B, C, W, H
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class EmoticCNN(nn.Module):
    def __init__(self):
        super(EmoticCNN, self).__init__()

        # Initialize VGG16 Model
        vgg16_1 = torchvision.models.vgg16(pretrained=True)
        vgg16_2 = torchvision.models.vgg16(pretrained=True)

        # Setup Feature Channels
        self.body_channel = vgg16_1.features
        self.image_channel = vgg16_2.features

        # Average Fusion Layers
        self.avg_pool_body = nn.AvgPool2d(4, stride=1)
        self.avg_pool_imag = nn.AvgPool2d(4, stride=16)

        # Feature Flatten Layers
        self.flat_body = Flatten()
        self.flat_imag = Flatten()

        # Fully Connected Layers
        # FIX: Reconsider tensor shape along each transformation.
        self.bn_layer = nn.BatchNorm2d(13312)
        self.fc_layer = nn.Linear(13312, 256)

        # Output Layers
        self.discrete_out = nn.Linear(256, 26)
        self.vad_out = nn.Linear(256, 3)

    def forward(self, body, image):
        '''
        body:  B, 3, 256, 256
        image :  B, 3, 256, 256
        '''
        # VGG16 Feature Extraction Channels
        x_1 = self.body_channel(body)  # B, 12800
        x_2 = self.image_channel(image) # B, 12800

        # Global Average Pooling
        x_1 = self.avg_pool_body(x_1)  # B, 12800
        x_2 = self.avg_pool_imag(x_2)   # B, 12800

        # Flatten Layers
        x_1 = self.flat_body(x_1)
        x_2 = self.flat_imag(x_2)

        # Concat + FC Layer
        out = torch.cat([x_1, x_2], 1)
        #out = self.bn_layer(out)
        out = self.fc_layer(out)

        # Output Layers
        y_1 = F.softmax(self.discrete_out(out))
        y_2 = self.vad_out(out)

        return y_1, y_2

# Auxillary Flatten Function
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def model_summary(model):
    for m in model.modules(): print(m)

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    return torch.load(path)

''' Loss Function '''
class DiscreteLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiscreteLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        if self.weight:
            disc_w = torch.ones(params.NDIM_DISC, 'ONES')
        else:
            sum_class = torch.sum(target, dim=0).float()
            mask = sum_class > 0.5

            if params.USE_CUDA:
                prev_w = torch.FloatTensor(params.NDIM_DISC).cuda() / torch.log(sum_class.data.float() + params.LDISC_C)
            else:
                prev_w = torch.FloatTensor(params.NDIM_DISC) / torch.log(sum_class.data.float() + params.LDISC_C)

            disc_w = mask.data.float() * prev_w

        # Compute Weighted Loss
        N = input.size()[0]
        loss = torch.sum((input.data - target.data.float()) * (input.data - target.data.float()), dim=0) / N
        w_loss = torch.sum(loss * disc_w, dim=0)

        # Return Loss Back as Torch Tensor
        return Variable(w_loss, requires_grad=True)

def disc_weight(input, target, weight=None):
    if weight == 'ONES':
        disc_w = torch.ones(params.NDIM_DISC)
    else:
        sum_class = torch.sum(target, dim=0).float()
        mask = sum_class > 0.5

        if params.USE_CUDA:
            prev_w = torch.FloatTensor(params.NDIM_DISC).cuda() / torch.log(sum_class.data.float() + params.LDISC_C)
        else:
            prev_w = torch.FloatTensor(params.NDIM_DISC) / torch.log(sum_class.data.float() + params.LDISC_C)

        disc_w = mask.data.float() * prev_w

    return disc_w

class ContinuousLoss(nn.Module):
    def __init__(self, weight=None):
        super(ContinuousLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        # Compute Weight Values
        if self.weight == None: self.weight = cont_weight(input, target)

        # Compute Weighted Loss
        N = input.size()[0]
        loss = torch.sum((input.data - target.data.float()).pow(2)) / N
        w_loss = torch.sum(loss * self.weight, dim=0)

        # Return Loss Back as Torch Tensor
        return Variable(w_loss, requires_grad=True)

def cont_weight(input, target, weight=None):
    if weight == 'ONES':
        cont_w = torch.ones(params.NDIM_CONT)
    else:
        diff = torch.sum(torch.abs(input.data - target.data).float(), dim=0) / input.data.size()[0]
        if params.USE_CUDA:
            cont_w = diff > torch.FloatTensor(params.NDIM_CONT).fill_(params.LOSS_CONT_MARGIN).cuda()
        else:
            cont_w = diff > torch.FloatTensor(params.NDIM_CONT).fill_(params.LOSS_CONT_MARGIN)

    return cont_w.float()

if __name__ == '__main__':
    # Discrete Loss Function Test
    '''
    loss = DiscreteLoss()

    y_pred = torch.LongTensor([[1, 0, 1], [0, 1, 1]]).cuda()
    y_real = torch.LongTensor([[0, 1, 1], [1, 1, 0]]).cuda()

    out = loss(y_pred, y_real)
    print(out)
    '''

    # Continuous Loss Function Test
    y_pred = torch.FloatTensor([[3, 7, 5], [2, 7, 8]]).cuda() / 10.0
    y_real = torch.FloatTensor([[4, 2, 3], [3, 4, 9]]).cuda() / 10.0

    loss = ContinuousLoss()

    out = loss(y_pred, y_real)
    print(out)
