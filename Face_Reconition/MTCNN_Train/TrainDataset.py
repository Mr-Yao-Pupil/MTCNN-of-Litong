"""
此文件用于将图片处理成为能传入神经网络的数据集
"""
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torch
from PIL import Image


class MTCNN_Dataset(Dataset):
    def __init__(self, path):
        """
        数据集初始化
        :param path: 数据集的根路径
        :param train: 是否为训练集，输入为bool值，True为训练，False为测试
        :param size: 请根据网络选择训练集尺寸，p网络输入12，r网络输入24，o网络输入为48
        """
        self.path = path
        self.transforms = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])

        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())  # 打开正样本标签文档，逐行读取，再添加至列表中
        self.dataset.extend(open(os.path.join(path, "negotive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        line = self.dataset[index].strip().split(" ")

        cond = torch.Tensor([int(line[1])])
        offset = torch.Tensor([float(line[2]), float(line[3]), float(line[4]), float(line[5])])

        img_path = os.path.join(self.path, line[0])
        img_data = Image.open(img_path)
        img_data = self.transforms(img_data)

        return img_data, cond, offset


if __name__ == '__main__':
    train_datasets = MTCNN_Dataset(path=r'F:\Datasets\MTCNN', train=True, size=12)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=100)
    for data_x, label_cond, label_offset in train_loader:
        # print(data_x.shape)
        print(label_cond)
        # print(label_offset.shape)
