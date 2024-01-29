import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
from transformer import LocalFeatureTransformer
from position_encoding import PositionEncodingSine
from einops.einops import rearrange

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    """
    Basic Block for resnet18 or resnet34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.res_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.res_func(x) + self.shortcut(x))

class SiamEncorder(nn.Module):
    def __init__(self, in_channels, block, num_block):
        super(SiamEncorder, self).__init__()
        self.resnet_name = []
        self.channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.channels, out_channels, stride))
            self.channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward_once(self, x):
        conv1 = self.conv1(x)
        temp = self.maxpool(conv1)
        conv2 = self.conv2_x(temp)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        bottle = self.conv5_x(conv4)
        return [conv1, conv2, conv3, conv4, bottle]

    def forward(self, x1, x2):
        x_shape = x1.shape
        x2 = x2.expand_as(x1)
        feature_1 = self.forward_once(x1)
        feature_2 = self.forward_once(x2)
        return feature_1, feature_2, x_shape

    def load_pretrained_weights(self):
        model_dict = self.state_dict()
        resnet34_dict = models.resnet34(True).state_dict()
        count_res = count_my = 1
        reskeys = list(resnet34_dict.keys())
        mykeys = list(model_dict.keys())

        corresp_map = []
        for i in range(215):
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if 'fc' in reskey:
                break
            while reskey.split('.')[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]
            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1
        for k_res, k_my in corresp_map:
            model_dict[k_my] = resnet34_dict[k_res]
            self.resnet_name.append(k_my)
        try:
            self.load_state_dict(model_dict)
            print('Loaded resnet34 weights in mynet')
        except:
            print('Error resnet34 weights in mynet')
            raise

class Match(nn.Module):
    def __init__(self, num = 0.2, temp = 0.1):
        super(Match, self).__init__()
        self.num = num
        self.temperature = temp

    def forward(self, shape, feature1, feature2):
        b, c, h, w = shape
        matrix = torch.einsum("nlc,nsc->nls", feature1, feature2) / self.temperature
        matrix = F.softmax(matrix, 1) * F.softmax(matrix, 2)
        matrix = matrix.cuda()
        mask = matrix > self.num
        # mutual nearest
        mask = mask \
               * (matrix == matrix.max(dim=2, keepdim=True)[0]) \
               * (matrix == matrix.max(dim=1, keepdim=True)[0])

        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        featureout = feature1.clone()
        for j in range(b_ids.size(0)):
            featureout[b_ids[j]][i_ids[j]] = feature1[b_ids[j]][i_ids[j]] - feature2[b_ids[j]][j_ids[j]]

        featureout = rearrange(featureout, 'n (h w) c -> n c h w',h=h,w=w)

        return featureout

class Seghead(nn.Module):
    def __init__(self, out_channels = 1):
        super(Seghead, self).__init__()

        self.conv1 = BasicConv2d(in_channels=64+64, out_channels=64)
        self.conv2 = BasicConv2d(in_channels=128+64, out_channels=64)
        self.conv3 = BasicConv2d(in_channels=128+256, out_channels=128)
        self.conv4 = BasicConv2d(in_channels=512+256, out_channels=256)
        self.conv5 = BasicConv2d(in_channels=512, out_channels=512)

        self.conv = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_out = nn.Conv2d(in_channels=64,out_channels=out_channels,kernel_size=3,padding=1)

    def forward(self, feature2):

        for i in range(5):
            if i ==0:
                out = self.conv[4-i](feature2[4-i])
            else:
                out = self.conv[4-i](torch.cat((self.up(out),feature2[4-i]),dim=1))
        out = self.conv_out(self.up(out))

        return out

class Decorder(nn.Module):
    def __init__(self, out_channels = 1):
        super(Decorder, self).__init__()
        self.pose3 = PositionEncodingSine(d_model=128)
        self.pose5 = PositionEncodingSine(d_model=512)

        self.trans3 = LocalFeatureTransformer(d_model = 128,nhead = 4,layer_names = ['self', 'cross'] * 2,attention = 'linear')
        self.trans5 = LocalFeatureTransformer(d_model = 512,nhead = 8,layer_names = ['self', 'cross'] * 4,attention = 'linear')

        self.match = Match()

        self.seghead = Seghead(out_channels = out_channels)

    def forward(self, feature_1, feature_2, x_shape):
        fea1 = []
        for i in range(5):
            fea1.append(feature_1[i])
        feature_1[2] = rearrange(self.pose3(feature_1[2]), 'n c h w -> n (h w) c')
        feature_2[2] = rearrange(self.pose3(feature_2[2]), 'n c h w -> n (h w) c')
        feature_1[4] = rearrange(self.pose5(feature_1[4]), 'n c h w -> n (h w) c')
        feature_2[4] = rearrange(self.pose5(feature_2[4]), 'n c h w -> n (h w) c')

        feature_1[2], feature_2[2] = self.trans3(feature_1[2],feature_2[2])
        feature_1[4], feature_2[4] = self.trans5(feature_1[4], feature_2[4])

        feature_1[0] = feature_1[0] - feature_2[0]
        feature_1[1] = feature_1[1] - feature_2[1]
        feature_1[2] = self.match(fea1[2].shape,feature_1[2],feature_2[2])
        feature_1[3] = feature_1[3] - feature_2[3]
        feature_1[4] = self.match(fea1[4].shape, feature_1[4], feature_2[4])

        out = self.seghead(feature_1)

        v1 = feature_1[4]

        return out

class BG_Net(nn.Module):

    def __init__(self, in_channels, out_channels, block, num_block):
        super(BG_Net, self).__init__()
        self.siam_encorder = SiamEncorder(in_channels, block, num_block)
        self.siam_encorder.load_pretrained_weights()
        self.decorder = Decorder(out_channels)

    def forward(self, x1, x2):
        feature_1, feature_2, x_shape = self.siam_encorder(x1, x2)
        out = self.decorder(feature_1,feature_2,x_shape)

        return out

    def save_model(self, model_path):
        torch.save(self.siam_encorder.state_dict(), os.path.join(model_path, 'best_siam_encorder.pth'))
        torch.save(self.decorder.state_dict(), os.path.join(model_path, 'best_decorder.pth'))

    def load_model(self, model_path):
        self.siam_encorder.load_state_dict(torch.load(os.path.join(model_path, 'best_siam_encorder.pth')))
        self.decorder.load_state_dict(torch.load(os.path.join(model_path, 'best_decorder.pth')))
        print('loaded trained model')

if __name__ == '__main__':
    device = torch.device('cuda')
    net = BG_Net(6, 1, BasicBlock, [3, 4, 6, 3])
    net.to(device)
    x1, x2 = torch.rand((2, 6, 256, 256)), torch.rand((2, 6, 256, 256))
    x1, x2 = x1.to(device), x2.to(device)
    x = net(x1, x2)
    print(x.shape)

