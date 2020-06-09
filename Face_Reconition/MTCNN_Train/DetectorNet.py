"""
此文件为MTCNN的三个网络代码，其网络框架与网上的基本一致
"""
import torch.nn as nn
import torch


# ----------------P网络----------------------
class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(10, 16, 3, 1, 0),
            nn.PReLU(),

            nn.Conv2d(16, 32, 3, 1, 0),
            nn.PReLU()
        )

        self.cond_module = nn.Conv2d(32, 1, 1, 1)
        self.offset_module = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        middle_put = self.sub_module(x)
        cond = nn.functional.sigmoid(self.cond_module(middle_put))
        offsets = self.offset_module(middle_put)

        return cond, offsets


# ----------------R网络----------------------
class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, 1),
            nn.PReLU(),

            nn.MaxPool2d(3, 2),

            nn.Conv2d(28, 48, 3, 1, 0),
            nn.PReLU(),

            nn.MaxPool2d(3, 2),

            nn.Conv2d(48, 64, 2, 1, 0),
            nn.PReLU()
        )

        self.fc_module = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU(),

            nn.Linear(128, 5)
        )

    def forward(self, x):
        middle_put = self.sub_module(x).reshape(x.size(0), -1)
        fc_out = self.fc_module(middle_put)
        cond = nn.functional.sigmoid(fc_out[:, :1])
        offsets = fc_out[:, 1:]

        return cond, offsets


# ----------------O网络----------------------
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.PReLU(),

            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 3, 1, 0),
            nn.PReLU(),

            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 64, 3, 1, 0),
            nn.PReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 2, 1, 0),
            nn.PReLU()
        )

        self.fc_module = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 5)
        )

    def forward(self, x):
        middle_put = self.sub_module(x).reshape(x.size(0), -1)
        fc_out = self.fc_module(middle_put)
        cond = nn.functional.sigmoid(fc_out[:, :1])
        offsets = fc_out[:, 1:]

        return cond, offsets


if __name__ == '__main__':
    x = torch.Tensor(4, 3, 48, 48)
    net = ONet()
    out = net(x)
    print(out[0].shape)
    print(out[1].shape)
