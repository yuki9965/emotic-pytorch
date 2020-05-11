from __future__ import print_function
import params
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class basicEncodingStream(nn.Module):
    def __init__(self):
        super(basicEncodingStream, self).__init__()
        self.conv = nn.Conv2d
        self.max_pool_layer = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.bn = nn.BatchNorm2d
        self.relu = nn.ReLU
        self.average_pool_layer = nn.functional.adaptive_avg_pool2d

        self.in_channel = 3
        self.out_channel_1 = 32
        self.out_channel_2 = 64
        self.out_channel_3 = 128
        self.out_channel_4 = 256
        self.out_channel_5 = 256

        self.layer1 = nn.Sequential(
                        self.conv(in_channels=self.in_channel, out_channels=self.out_channel_1,
                            kernel_size=3),
                        self.bn(self.out_channel_1),
                        self.relu(True),
                        self.max_pool_layer
                    )
        self.layer2 = nn.Sequential(
                        self.conv(in_channels=self.out_channel_1, out_channels=self.out_channel_2,
                            kernel_size=3),
                        self.bn(self.out_channel_2),
                        self.relu(True),
                        self.max_pool_layer
                    )
        self.layer3 = nn.Sequential(
                        self.conv(in_channels=self.out_channel_2, out_channels=self.out_channel_3,
                            kernel_size=3),
                        self.bn(self.out_channel_3),
                        self.relu(True),
                        self.max_pool_layer
                    )
        self.layer4 = nn.Sequential(
                        self.conv(in_channels=self.out_channel_3, out_channels=self.out_channel_4,
                            kernel_size=3),
                        self.bn(self.out_channel_4),
                        self.relu(True),
                        self.max_pool_layer
                    )
        self.layer5 = nn.Sequential(
                        self.conv(in_channels=self.out_channel_4, out_channels=self.out_channel_5,
                            kernel_size=3),
                        self.bn(self.out_channel_5),
                        self.relu(True),
                    )

class BodyStream(torchvision.models.Inception3):
    def __init__(self):
        super(BodyStream, self).__init__()
        self.average_pool_layer = F.adaptive_avg_pool2d

    def forward(self, x):
        '''
        body:  B, 3, 299, 299
        '''
        #batch_size = body.size(0)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        #x = self.average_pool_layer(x, (1, 1))
        # N x 2048 x 1 x 1
        #x = F.dropout(x, training=self.training)

        return x    

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
        self.g = self.g.type(torch.cuda.FloatTensor)

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
        
        self.W = self.W.type(torch.cuda.FloatTensor)
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.theta = self.theta.type(torch.cuda.FloatTensor)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.phi = self.phi.type(torch.cuda.FloatTensor)

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
        g_x = self.g(x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)
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

class ContextStream(torchvision.models.Inception3):
    def __init__(self):
        super(ContextStream, self).__init__()
        self.average_pool_layer = F.adaptive_avg_pool2d
    
    def forward(self, x):

        '''
        image:  B, 3, 299, 299
        '''
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        x = _NonLocalBlock(in_channels=2048, dimension=2)(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        #x = self.average_pool_layer(x, (1, 1))
        # N x 2048 x 1 x 1
        #x = F.dropout(x, training=self.training)

        return x

class FusionStream(nn.Module):
    def __init__(self):
        super(FusionStream, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )

        self.layer3 = nn.Sequential(
                        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )
        self.layer5 = nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=26, kernel_size=1),
                        nn.ReLU(),
                        nn.Dropout(0.5)
        )
    
    def forward(self, bodyFeature, contextFeature):
        body_feature_channel = bodyFeature.size()[1]
        context_feature_channel = contextFeature.size()[1]

        assert body_feature_channel == context_feature_channel

        bodyattention = self.layer1(bodyFeature)
        bodyattention = self.layer2(bodyattention)

        contextattention = self.layer1(contextFeature)
        contextattention = self.layer2(contextattention)

        tmp = torch.cat((bodyattention, contextattention), 1)
        tmp = torch.softmax(tmp, 1)

        lamdaf, lamdac = tmp.split([1,1], dim = 1)
        bodyFeature = torch.mul(lamdaf, bodyFeature)
        contextFeature = torch.mul(lamdac, contextFeature)
        output = torch.cat((bodyFeature, contextFeature), dim=1) #512

        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = torch.squeeze(output)
        y = F.softmax(output)

        return y

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class emotic_attention_model(nn.Module):
    def __init__(self, num_classes = 26):
        super(emotic_attention_model, self).__init__()
        self.bodystream = BodyStream()
        self.contextstream = ContextStream()
        self.ffm = FeatureFusionModule(in_chan=4096, out_chan=4096)
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.fc = nn.Linear(256, num_classes)


    
    def forward(self, body , context):
        bodyfeature = self.bodystream(body)
        contextfeature = self.contextstream(context)

        y = self.ffm(bodyfeature, contextfeature) # N,4096,8,8
        y = F.adaptive_avg_pool2d(y, (1,1))  # N, 4096, 1,1
        y = F.dropout(y, training=self.training)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y) # N, 256, 1,1

        y = torch.flatten(y, 1) # N, 256
        y = self.fc(y)
        #y = F.softmax(y)

        return y
        



if __name__ == "__main__":
    body = torch.rand(1, 3, 256, 256).cuda()
    context = torch.rand(1, 3, 256, 256).cuda()

    model = emotic_attention_model().cuda()
    out = model(body, context)
    print(out)

    pass

    




