"""
此文件用于图片人脸检测
"""
import numpy as np
import utils
from MTCNN_Train import DetectorNet
from torchvision import transforms
import time
import os
import torch
from PIL import Image, ImageDraw

# -----------------------------设置网络参数--------------------------------
p_cls = 0.4
p_nms = 0.5

r_cls = 0.5
r_nms = 0.5

o_cls = 0.9
o_nms = 0.5


class ImgDetector:
    def __init__(self, pnet_param=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\PNet.pt',
                 rnet_param=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\RNet.pt',
                 onet_param=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\ONet.pt',
                 isCuda=True):
        """
        初始化训练参数
        :param pnet_param: p网络的权重存储地址
        :param rnet_param: r网络的权重存储地址
        :param onet_param: o网络的权重存储地址
        :param isCuda: 是否使用Cuda进行测试
        """
        self.iscuda = isCuda

        # ------------------------实例化网络---------------------------------------
        self.pnet = DetectorNet.PNet().cuda() if self.iscuda else DetectorNet.PNet()
        self.rnet = DetectorNet.RNet().cuda() if self.iscuda else DetectorNet.RNet()
        self.onet = DetectorNet.ONet().cuda() if self.iscuda else DetectorNet.ONet()

        # ----------------------加载网络参数权重---------------------------------------
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        # ----------------------开启网络的测试模式--------------------------------------
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        # ---------------------设置图片信息------------------------------------
        self.__data_transforms = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])

    def detect(self, image):
        """
        此函数为人脸检测的主函数
        :param image: 原图片
        :return: o网络最后检测的结果
        """
        # ---------------------p网络检测----------------------------------------
        # -------------------开始计时，计算检测时间---------------------------
        start_time = time.time()

        # ------------------------调用p网络检测方法进行检测------------------
        pnet_boxes = self.__pnet_detect(image)

        # ------------------对检测结果进行判断---------------------------------
        if pnet_boxes.shape[0] == 0:
            return np.array([])

        # -------------------结束计时，并对p网络的运算时间进行计算------------------
        end_time = time.time()
        p_time = end_time - start_time

        # ---------------------r网络检测-------------------------------------
        # -------------------开始计时，计算检测时间---------------------------
        start_time = time.time()

        # ------------------------调用r网络检测方法进行检测------------------
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)

        # ------------------对检测结果进行判断---------------------------------
        if rnet_boxes.shape[0] == 0:
            return np.array([])

        # -------------------结束计时，并对p网络的运算时间进行计算------------------
        end_time = time.time()
        r_time = end_time - start_time

        # ---------------------o网络检测 - ------------------------------------
        # -------------------开始计时，计算检测时间---------------------------
        start_time = time.time()

        # ------------------------调用o网络检测方法进行检测------------------
        onet_boxes = self.__onet_detect(image, rnet_boxes)

        # ------------------对检测结果进行判断---------------------------------
        if onet_boxes.shape[0] == 0:
            return np.array([])

        # -------------------结束计时，并对p网络的运算时间进行计算------------------
        end_time = time.time()
        o_time = end_time - start_time
        total_time = p_time + r_time + o_time
        print(f'total:{total_time}---->PNet:{p_time}---->RNet:{r_time}---->ONet:{o_time}')

        return pnet_boxes

    def __pnet_detect(self, image):
        """
        p网络检测方法
        :param image: 传入的图片信息
        :return: p网络检测结束使用非极大值抑制去重后剩下的框信息
        """
        # ------------------初始化检测参数------------------------------
        boxes = np.zeros((0, 5))
        img = image
        w, h = img.size
        min_side = min(w, h)  # 获得最小边长

        scale = 1  # 缩放倍数

        # ---------------------开始检测------------------------------------
        while min_side >= 12:
            img_data = self.__data_transforms(img).cuda() if self.iscuda else self.__data_transforms(img)

            # -------------------图片升维--------------------------------
            img_data.unsqueeze_(0)

            # ---------------传入网络检测-------------------------------
            _cls, _offset = self.pnet(img_data)

            # ----------------将输出结果移动至cpu中----------------------================>numpy只能在cpu中进行运算
            cls, offset = _cls.cpu(), _offset.cpu()

            # -------将新检测的框在图片上反算坐标点并与已经反算借宿的框进行封装-----------------------
            boxes = np.vstack([boxes, self.__box(cls, offset, scale)])
            # boxes = np.array(self.__box(cls, offset, scale))

            # ---------------定义缩放倍数并对图片进行缩放----------------------------------------
            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            min_side = min(_w, _h)

        return utils.nms(boxes, thresh=p_nms)
        # return boxes

    def __box(self, cls, offset, scale, stride=2, side_len=12):
        """
        此方法用于p网络检测结果的框反算
        :param cls: 传入框的置信度，传入的数据结构为[N, cls, H, W]
        :param offset: 传入框的偏移量，传入的数据结构为[N, 4, H, W]
        :param scale: 传入缩放倍数
        :param stride: 传入网络的检测步长，默认值为2
        :param side_len: 传入建议框的边长，默认为12
        :return: 反算图片后框的信息
        """
        # ---------------获得置信度大于阈值的框掩码-------------------------
        mask = torch.gt(cls, p_cls)

        # -----------------获得符合阈值的框信息----------------------------
        cls = cls[mask]

        # --------------对传入数据进行判断，避免程序报错--------------------
        if cls.shape[0] == 0:
            return np.array([[0, 0, 0, 0, 0]])

        # ------------获得符合框在特征图的位置索引--------------------------
        idxs = mask.nonzero()

        # ---------------根据索引反算原图的建议框位置----------------------
        _x1 = (idxs[:, 3].float() * stride) / scale
        _y1 = (idxs[:, 2].float() * stride) / scale
        _x2 = (idxs[:, 3].float() * stride + side_len) / scale
        _y2 = (idxs[:, 2].float() * stride + side_len) / scale

        # -------------------计算建议框在原图上的边长-----------------------
        w = _x2 - _x1

        # ------------------实际框位置计算-----------------------------
        x1 = _x1 + w[0] * offset[:, 0:1, :, :][mask]
        y1 = _y1 + w[0] * offset[:, 1:2, :, :][mask]
        x2 = _x2 + w[0] * offset[:, 2:3, :, :][mask]
        y2 = _y2 + w[0] * offset[:, 3:4, :, :][mask]

        return torch.stack((x1, y1, x2, y2, cls), dim=1).detach().numpy()

    def __rnet_detect(self, image, pnet_boxes):
        """
        r网络检测方法
        :param image: 传入原图片
        :param pnet_boxes: p网络检测的框信息
        :return: r网络检测后的框信息
        """
        _img_dataset = []

        # -------------调用to_square方法对p网络的输出框变成正方形---------------
        _pnet_boxes = torch.from_numpy(utils.to_square(pnet_boxes))

        # --------------计算建议框在原图中的位置--------------------------------
        _x1 = _pnet_boxes[:, 0]
        _y1 = _pnet_boxes[:, 1]
        _x2 = _pnet_boxes[:, 2]
        _y2 = _pnet_boxes[:, 3]

        # --------------对建议框进行封装----------------------------------
        crops_np = np.vstack((_x1, _y1, _x2, _y2)).transpose(1, 0)

        # ------------------抠图并做成数据集----------------------------------------
        for _box in crops_np:
            img = image.crop(_box)
            img = img.resize((24, 24))
            img_data = self.__data_transforms(img)
            _img_dataset.append(img_data)

        # ----------------对数据集进行封装---------------------
        img_dataset = torch.stack(_img_dataset).cuda() if self.iscuda else torch.stack(_img_dataset)

        # -------------传入r网络进行计算-------------------------
        _cls, _offset = self.rnet(img_dataset)

        # -----------------数据移动至cpu----------------------
        cls, offset = _cls.cpu(), _offset.cpu()

        # -------------计算符合阈值数据的掩码和索引-------------------
        mask = cls[:, 0:] > r_cls
        idxs = mask.nonzero()[:, 0]

        # ----------------获得传入数据中符合阈值的数据--------------------
        _box = _pnet_boxes[idxs]

        # -------------获得符合阈值的r网络的输出数据--------------------
        cls = cls[idxs]
        cls = cls.reshape(cls.size(0))
        offset = offset[idxs]

        # --------------------计算建议框位置-----------------------==============>一回儿试试能不能删除掉
        _x1 = _box[:, 0]
        _y1 = _box[:, 1]
        _x2 = _box[:, 2]
        _y2 = _box[:, 3]

        # ----------------------计算建议框的宽和高-------------------------
        w = _x2 - _x1
        h = _y2 - _y1

        # -------------------计算实际框位置------------------------------
        x1 = _x1 + w * offset[:, 0]
        y1 = _y1 + h * offset[:, 1]
        x2 = _x2 + w * offset[:, 2]
        y2 = _y2 + h * offset[:, 3]

        # ----------------------将反算结果进行封装-------------------------
        boxes = torch.stack((x1, y1, x2, y2, cls.to(torch.float64)), dim=1).detach().numpy()

        return utils.nms(boxes, thresh=r_nms)

    def __onet_detect(self, image, rnet_boxes):
        """
        此函数为人脸检测的主函数
        :param image: 原图片
        :param rnet_boxes: r网络的检测结果
        :return: o网络的检测结果
        """
        _img_dataset = []

        # -------------调用to_square方法对p网络的输出框变成正方形---------------
        _rnet_boxes = torch.from_numpy(utils.to_square(rnet_boxes))

        # --------------计算建议框在原图中的位置--------------------------------
        _x1 = _rnet_boxes[:, 0]
        _y1 = _rnet_boxes[:, 1]
        _x2 = _rnet_boxes[:, 2]
        _y2 = _rnet_boxes[:, 3]

        # --------------对建议框进行封装----------------------------------
        crops_np = np.vstack((_x1, _y1, _x2, _y2)).transpose(1, 0)

        # ------------------抠图并做成数据集----------------------------------------
        for _box in crops_np:
            img = image.crop(_box)
            img = img.resize((48, 48))
            img_data = self.__data_transforms(img)
            _img_dataset.append(img_data)

        # ----------------对数据集进行封装---------------------
        _img_dataset = torch.stack(_img_dataset).cuda() if self.iscuda else torch.stack(_img_dataset)

        # -------------传入r网络进行计算-------------------------
        _cls, _offset = self.onet(_img_dataset)

        # -----------------数据移动至cpu----------------------
        cls, offset = _cls.cpu(), _offset.cpu()

        # -------------计算符合阈值数据的掩码和索引-------------------
        mask = cls[:, 0:] > o_cls
        idxs = mask.nonzero()[:, 0]

        # ----------------获得传入数据中符合阈值的数据--------------------
        _box = _rnet_boxes[idxs]

        # -------------获得符合阈值的r网络的输出数据--------------------
        cls = cls[idxs]
        cls = cls.reshape(cls.size(0))
        offset = offset[idxs]

        # --------------------计算建议框位置-----------------------==============>一回儿试试能不能删除掉
        _x1 = _box[:, 0]
        _y1 = _box[:, 1]
        _x2 = _box[:, 2]
        _y2 = _box[:, 3]

        # ----------------------计算建议框的宽和高-------------------------
        w = _x2 - _x1
        h = _y2 - _y1

        # -------------------计算实际框位置------------------------------
        x1 = _x1 + w * offset[:, 0]
        y1 = _y1 + h * offset[:, 1]
        x2 = _x2 + w * offset[:, 2]
        y2 = _y2 + h * offset[:, 3]

        # ----------------------将反算结果进行封装-------------------------
        boxes = torch.stack((x1, y1, x2, y2, cls.to(torch.float64)), dim=1).detach().numpy()

        if boxes.shape[0] > 0:
            return utils.nms(boxes, True, thresh=o_nms)
        else:
            return np.array([[0, 0, 0, 0, 0]])


if __name__ == '__main__':
    img_path = r'F:\Datasets\MTCNN\人脸检测验证'

    for i in os.listdir(img_path):
        detector = ImgDetector()

        with Image.open(os.path.join(img_path, i)) as im:
            print('==================================')
            print(i)

            boxes = detector.detect(im)
            print(f'size:{im.size}')

            draw = ImageDraw.Draw(im)

            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                print(f'cond:{box[4]}')

                draw.rectangle((x1, y1, x2, y2), outline='red')
            im.show()
