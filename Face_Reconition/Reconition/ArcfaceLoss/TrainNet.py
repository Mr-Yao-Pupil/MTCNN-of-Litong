import torch
import torch.nn as nn


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super(Arcsoftmax, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_num, cls_num)).cuda()

    def forward(self, x, s, m):
        x_norm = torch.nn.functional.normalize(x, dim=1)
        w_norm = torch.nn.functional.normalize(self.w, dim=0)
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(s * torch.cos(a + m) * 10) / (
                torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(s * cosa * 10) + torch.exp(
            s * torch.cos(a + m) * 10)
        )

        return arcsoftmax


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  # 14 * 14 * 32

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # 7 * 7 * 64

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()  # 4 * 4 * 128
        )

        self.fc_module = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 2)
        )

        self.arcsoftmax = Arcsoftmax(2, 10)

        self.loss = nn.NLLLoss()

    def forward(self, x, s, m):
        x = self.conv_module(x)
        feature = self.fc_module(x.reshape(x.size(0), -1))
        outputs = torch.log(self.arcsoftmax(feature, s, m))

        return feature, outputs

    def getloss(self, out, ys):
        return self.loss(out, ys)