import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, cls_num, feature_num):
        super(CenterLoss, self).__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num, feature_num))

    def forward(self, xs, ys):
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)
        count_exp = count.index_select(dim=0, index=ys.long())

        return torch.sum(torch.div(torch.sqrt(torch.sum((torch.pow(xs - center_exp, 2)), dim=1)), count_exp))


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

        self.out_module = nn.Linear(2, 10)

        self.center_module = CenterLoss(10, 2)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, xs):
        xs = self.conv_module(xs).reshape(xs.size(0), -1)
        features = self.fc_module(xs)
        outputs = self.out_module(features)

        return features, outputs

    def getloss(self, outputs, features, labels):
        loss_cls = self.loss(outputs, labels)
        loss_center = self.center_module(features, labels)
        loss = loss_cls + loss_center

        return loss
