import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        # self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)

    def forward(self, x):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            # alpha = torch.rand(N, 1, 1)
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap


class SpeRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, idx_swap, y=None):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            if y != None:
                for i in range(len(y.unique())):
                    index = y == y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
        return x


class Generator(nn.Module):
    def __init__(self, mid_channels=64, kernelsize=3, in_channels=48, imsize=[13, 13], device=0):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.imsize = imsize
        self.device = device

        self.l1 = nn.Linear(self.in_channels, self.mid_channels)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 32)

        self.conv_spa1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels,
                                   kernel_size=(1, 1), stride=(1, 1))
        self.spaRandom = SpaRandomization(self.mid_channels, device=device)
        self.conv_spa2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels,
                                   kernel_size=(1, 1), stride=(1, 1))

        self.conv_spe1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32,
                                   kernel_size=(imsize[0], imsize[0]), stride=(1, 1))
        self.speRandom = SpeRandomization(self.mid_channels)
        self.conv_spe2 = nn.ConvTranspose2d(in_channels=self.mid_channels, out_channels=self.mid_channels,
                                            kernel_size=(imsize[0], imsize[0]))
        self.conv1 = nn.Conv2d(self.mid_channels + self.mid_channels, self.mid_channels, kernelsize, 1, 1)
        self.conv2 = nn.Conv2d(self.mid_channels, self.in_channels, kernelsize, 1, 1)

    def forward(self, x, y):  # 256 48 13 13
        x1 = F.relu(self.conv_spa1(x))
        x2, idx_swap = self.spaRandom(x1)
        x_spa = self.conv_spa2(x2)

        x_spe = F.relu(self.conv_spe1(x))
        y1 = F.relu(self.l1(y))
        y2 = F.relu(self.l2(y1))
        y3 = F.relu(self.l3(y2))
        y4 = F.relu(self.l4(y3))
        y5 = y1 + y4
        y6 = F.relu(self.l5(y5))
        y7 = y6.view(y6.size(0), y6.size(1), 1, 1)
        y_spe = torch.cat((x_spe, y7), 1)
        y_spe = self.speRandom(y_spe, idx_swap)
        y_spe = self.conv_spe2(y_spe)
        x_y = torch.cat((x_spa, y_spe), 1)
        x_y = F.relu(self.conv1(x_y))
        x_y = torch.sigmoid(self.conv2(x_y))
        return x_y


class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes):
        super(Discriminator, self).__init__()
        self.conv2d_10 = nn.Sequential(
            nn.Conv2d(inchannel, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_base_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block1 = conv2d_Resblock(64)
        self.conv2d_20 = nn.Sequential(
            nn.Conv2d(inchannel+64, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_base_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block2 = conv2d_Resblock(64)
        self.conv2d_30 = nn.Sequential(
            nn.Conv2d(inchannel+64, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_base_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block3 = conv2d_Resblock(64)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.pooling_feature1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_feature2 = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_feature3 = nn.AdaptiveAvgPool2d((1, 1))

        self.cls_head_src = nn.Linear(192, int(num_classes))
        self.pro_head = nn.Linear(192, outchannel, nn.ReLU())

    def forward(self, x, mode='test'):
        out_2dscale1 = self.conv2d_10(x)
        feature1 = self.pooling_feature1(out_2dscale1)
        feature1 = feature1.reshape(feature1.shape[0], -1)
        out_2dscale1 = self.conv2d_base_1(out_2dscale1)
        out_2dscale1 = self.conv2d_block1(out_2dscale1)

        out_2dscale2 = torch.cat((out_2dscale1, x), 1)
        out_2dscale2 = self.conv2d_20(out_2dscale2)
        feature2 = self.pooling_feature1(out_2dscale2)
        feature2 = feature2.reshape(feature2.shape[0], -1)
        out_2dscale2 = self.conv2d_base_2(out_2dscale2)
        out_2dscale2 = self.conv2d_block2(out_2dscale2)

        out_2dscale3 = torch.cat((out_2dscale2, x), 1)
        out_2dscale3 = self.conv2d_30(out_2dscale3)
        feature3 = self.pooling_feature1(out_2dscale3)
        feature3 = feature3.reshape(feature3.shape[0], -1)
        out_2dscale3 = self.conv2d_base_3(out_2dscale3)
        out_2dscale3 = self.conv2d_block3(out_2dscale3)

        out1 = self.avgpool1(out_2dscale1)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = self.avgpool2(out_2dscale2)
        out2 = out2.reshape(out2.shape[0], -1)

        out3 = self.avgpool3(out_2dscale3)
        out3 = out3.reshape(out3.shape[0], -1)
        out = torch.cat((out1, out2, out3), dim=1)

        if mode == 'test':
            clss = self.cls_head_src(out)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out))
            clss = self.cls_head_src(out)
            return clss, proj, feature3


class conv2d_Resblock(nn.Module):
    def __init__(self, in_channels):
        super(conv2d_Resblock, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_conv2d_1 = self.conv2d_1(x)
        out_conv2d_2 = self.conv2d_2(out_conv2d_1)
        out = x + out_conv2d_2
        return out


class Generator_v12(nn.Module):
    def __init__(self, mid_channels=64, kernelsize=3, in_channels=48, imsize=[13, 13], device=0):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.imsize = imsize
        self.device = device

        self.l1 = nn.Linear(self.in_channels, self.mid_channels)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 32)

        self.conv_spa1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels,
                                   kernel_size=(1, 1), stride=(1, 1))
        self.spaRandom = SpaRandomization(self.mid_channels, device=device)
        self.conv_spa2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels,
                                   kernel_size=(1, 1), stride=(1, 1))

        self.conv_spe1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32,
                                   kernel_size=(imsize[0], imsize[0]), stride=(1, 1))
        self.speRandom = SpeRandomization(self.mid_channels)
        self.conv_spe2 = nn.ConvTranspose2d(in_channels=self.mid_channels, out_channels=self.mid_channels,
                                            kernel_size=(imsize[0], imsize[0]))
        self.conv1 = nn.Conv2d(self.mid_channels + self.mid_channels, self.mid_channels, kernelsize, 1, 1)
        self.conv2 = nn.Conv2d(self.mid_channels, self.in_channels, kernelsize, 1, 1)

    def forward(self, x, y):  # 256 48 13 13
        x1 = F.relu(self.conv_spa1(x))
        x2, idx_swap = self.spaRandom(x1)
        x_spa = self.conv_spa2(x2)

        x_spe = F.relu(self.conv_spe1(x))
        y1 = F.relu(self.l1(y))
        y2 = F.relu(self.l2(y1))
        y3 = F.relu(self.l3(y2))
        y4 = F.relu(self.l4(y3))
        y5 = y1 + y4
        y6 = F.relu(self.l5(y5))
        y7 = y6.view(y6.size(0), y6.size(1), 1, 1)
        y_spe = torch.cat((x_spe, y7), 1)
        y_spe = self.speRandom(y_spe, idx_swap)
        y_spe = self.conv_spe2(y_spe)
        x_y = torch.cat((x_spa, y_spe), 1)
        x_y = F.relu(self.conv1(x_y))
        x_y = torch.sigmoid(self.conv2(x_y))
        return x_y
