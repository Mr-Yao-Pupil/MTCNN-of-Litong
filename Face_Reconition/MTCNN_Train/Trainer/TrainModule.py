"""
此文件封装了网络的训练方法
"""
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from MTCNN_Train.TrainDataset import MTCNN_Dataset


class trainer:
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        """
        初始化网络训练数据
        :param net: 训练的网络模块，有三种可选:PNet, RNet, ONet
        :param dataset_path: 使用的数据集路径
        :param save_path: 训练模型存储的路径
        :param dataset_size: 网络训练所使用的图片尺寸，有三种可选：12, 24, 48
        :param device: 训练所使用的硬件，分别为'cuda'或者'cpu'
        """
        # -----------------------传入训练参数------------------------------
        self.net = net
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.isCuda = isCuda

        # ----------------------将网络移动到训练设备中-------------------------
        if self.isCuda:
            self.net.cuda()

        # ----------------------定义损失函数--------------------------------
        self.cond_lossfn = nn.BCELoss()
        self.offset_lossfn = nn.MSELoss()

        # --------------------------定义优化器-----------------------------
        self.opt = torch.optim.Adam(self.net.parameters())

        # ------------------如果模型文件夹中有该文件，加载该文件的参数----------
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))

    def train(self):
        # --------------------载入训练数据集-------------------------------
        train_dataset = MTCNN_Dataset(path=self.dataset_path)
        train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True)

        epoch = 1

        while True:
            for i, (img_data_, category_, offset_) in enumerate(train_loader):
                # -----------------将数据移动至显卡或者cpu中----------------------
                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()

                # --------------------前向计算---------------------------------
                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)  # [512,1]
                output_offset = _output_offset.view(-1, 4)  # [512,4]

                # ------------------根据需求取出相应的标签-------------------------
                # ------------------取出正负样本的标签用于置信度损失计算-----------
                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cond_lossfn(output_category, category)

                # -------------------取出正、部分样本的标签用于偏移量的损失计算----------------
                offset_mask = torch.gt(category_, 0)
                offset_index = torch.nonzero(offset_mask)[:, 0]
                offset = offset_[offset_index]
                output_offset = output_offset[offset_index]
                offset_loss = self.offset_lossfn(output_offset, offset)

                # ----------------计算总损失-------------------------
                loss = cls_loss + offset_loss

                # -------------------------清空梯度、反向传播、更新梯度-----------------------
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # ----------------------------打印训练过程----------------------------------
                print(f'epoch:{epoch}----->times:{i}------->num_loss:{loss}=====>cond_loss:{cls_loss}=====>offset_loss{offset_loss}')

                if (i + 1) % 100 == 0:
                    torch.save(self.net.state_dict(), self.save_path)
                    print('权重保存完成！！！！！！！！')
            epoch += 1
