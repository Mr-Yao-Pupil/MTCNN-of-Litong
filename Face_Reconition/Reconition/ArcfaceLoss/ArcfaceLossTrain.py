from Reconition.ArcfaceLoss.TrainNet import *
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

save_path = r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\ArcfaceLoss.pt'

epoch = 0

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root=r"F:\Datasets\MNIST", train=True, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

net = MainNet().cuda()
if os.path.exists(save_path):
    net = torch.load(save_path)

opt = torch.optim.Adam(net.parameters())


def visualize(feat, labels, epoch):
    plt.ion()
    color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()

    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=color[i])

    plt.title(f"epoch:{epoch}")
    plt.draw()
    plt.pause(0.001)


while True:
    feat_loader = []
    label_loader = []

    for i, (x, y) in enumerate(train_loader):
        feature, outputs = net(x.cuda(), 1, 1)
        ys = y.cuda()

        loss = net.getloss(outputs, ys)

        opt.zero_grad()
        loss.backward()
        opt.step()

        feat_loader.append(feature)
        label_loader.append(y)

        if i % 10 == 0:
            print(f'epoch:{epoch}------>loss:{loss}')

    feat = torch.cat(feat_loader, 0)
    labels = torch.cat(label_loader, 0)

    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
    epoch += 1
    torch.save(net, save_path)
