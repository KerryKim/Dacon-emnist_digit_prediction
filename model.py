## import libraries

import torch
import torch.nn as nn
import torch.nn.functional as F

## 네트워크 구축하기
# Image size = 28*28

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 순서 : Conv → BN → ReLu
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 풀링을 하는 이유는 차원을 줄이기 위함인데 28*28이라 굳이 할 필요 없을듯

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        #self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.res_block = CBR2d(in_channels=128, out_channels=128)

        self.enc3_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc3_2 = CBR2d(in_channels=128, out_channels=128)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        cat = torch.cat((enc1_2, x), dim=1)

        enc2_1 = self.enc2_1(enc1_2)
        enc2_2 = self.enc2_2(enc2_1)
        cat = torch.cat((enc2_2, cat), dim=1)

        for i in range(5):
            res_block = res_block(cat)
            cat = torch.cat((cat, res_block), dim=1)

        enc3_1 = self.enc3_1(cat)
        enc3_2 = self.enc3_2(enc3_1)

        x = x.view(-1, self.num_flat_features(enc3_2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # self.num_flat_features() 메서드는 input 텐서의 총 parameter 갯수이다.
        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

        return x






'''

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # self.num_flat_features() 메서드는 input 텐서의 총 parameter 갯수이다.
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



'''