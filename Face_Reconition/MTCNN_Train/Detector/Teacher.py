import torch
from PIL import Image
import cv2
import numpy as np
import utils
from  MTCNN_Train.DetectorNet import *
from torchvision import transforms
import time
import os



class Detector:

    def __init__(self,pnet_param=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\PNet.pt',
                 rnet_param=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\RNet.pt',
                 onet_param=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\ONet.pt',):

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pnet.to(self.device)
        self.rnet.to(self.device)
        self.onet.to(self.device)

        self.pnet.load_state_dict(torch.load(pnet_param,map_location="cpu"))
        self.rnet.load_state_dict(torch.load(rnet_param,map_location="cpu"))
        self.onet.load_state_dict(torch.load(onet_param,map_location="cpu"))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self,image,zoom_factor,p_conf,p_nms,r_conf,r_nms,o_conf,o_nms):
        print(torch.cuda.is_available())
        start_time = time.time()
        pnet_boxes = np.array(self.__pnet_detect(image,zoom_factor,p_conf,p_nms))
        # print(pnet_boxes)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # return pnet_boxes

        start_time = time.time()
        rnet_boxes = np.array(self.__rnet_detect(image, pnet_boxes,r_conf,r_nms))
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = np.array(self.__onet_detect(image, rnet_boxes,o_conf,o_nms))
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self,image,zoom_factor,conf,nms):
        bboxes = []
        boxes = []

        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            img_data = img_data.to(self.device)
            img_data.unsqueeze_(0)


            _cls, _offest = self.pnet(img_data)
            # print(_cls)
            # print(_offest)

            cls, offest = _cls[0, 0].cpu().data, _offest[0].cpu().data
            # print(cls)
            # print(offest)
            #得到置信度大于阈值的索引值，就可以返回算同一个索引的置信度和偏移率
            idxs = torch.nonzero(torch.gt(cls, conf))
            boxes = self.__box(idxs, offest, cls[idxs[:,0], idxs[:,1]], scale)

            scale *= zoom_factor
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            # print(min_side_len)
            min_side_len = np.minimum(_w, _h)
            # print(min_side_len)
            # break
            boxes = utils.nms(np.array(boxes), nms)
            bboxes.extend(boxes)
        # return bboxes
        return bboxes

    # 将偏移回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = ((start_index[:,1]).to(torch.float32) * stride+0) / scale#宽，W，x
        _y1 = ((start_index[:,0]).to(torch.float32) * stride+0) / scale#高，H,y
        _x2 = ((start_index[:,1]).to(torch.float32) * stride + side_len) / scale
        _y2 = ((start_index[:,0]).to(torch.float32) * stride + side_len) / scale
        # print(_x1.dtype)
        # ow,oh就等于side_len/scale
        ow = _x2 - _x1  # 等于12/0.6^n
        oh = _y2 - _y1  # 等于12/0.6^n


        _offset = offset[:, start_index[:,0], start_index[:,1]]
        # print(_offset.shape)
        x1 = _x1 + ow * _offset[0,:]
        y1 = _y1 + oh * _offset[1,:]
        x2 = _x2 + ow * _offset[2,:]
        y2 = _y2 + oh * _offset[3,:]
        # print([x1, y1, x2, y2, cls])
        bboxes = torch.stack([x1,y1,x2,y2,cls])
        bboxes = torch.transpose(bboxes,1,0)
        # print(bboxes)
        return bboxes

    def __rnet_detect(self, image, pnet_boxes,conf,nms):

        _img_dataset = []
        _pnet_boxes = utils.to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _offset = self.rnet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        #得到R网络置信度达标的索引
        idxs, _ = np.where(_cls > conf)
        _box = _pnet_boxes[idxs]
        # print(_box)
        # print(_box[:,0])
        #得到一组_x1/_y1/_x2/_y2
        _x1 = (_box[:,0])
        _y1 = (_box[:,1])
        _x2 = (_box[:,2])
        _y2 = (_box[:,3])

        #得到一组ow/oh
        ow = _x2 - _x1
        oh = _y2 - _y1

        # 得到一组x1/y1/x2/y2/cls
        x1 = _x1 + ow * offset[idxs][:,0]
        y1 = _y1 + oh * offset[idxs][:,1]
        x2 = _x2 + ow * offset[idxs][:,2]
        y2 = _y2 + oh * offset[idxs][:,3]
        cls = _cls[idxs][:,0]

        # print(x1)
        # print(cls)
        #合并成一组组的x1,y1,x2,y2
        boxes=[x1, y1, x2, y2, cls]
        # print(np.shape(boxes))
        #nms之前将一组组的x1,y1,x2,y2转换成一组组的box
        boxes = utils.nms(np.array(boxes).T, nms)
        # print(np.shape(boxes))
        return boxes

    def __onet_detect(self, image, rnet_boxes,conf,nms):

        _img_dataset = []
        _rnet_boxes = utils.to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()


        idxs, _ = np.where(_cls > conf)
        _box = _rnet_boxes[idxs]
        # print(_box)
        # print(_box[:,0])
        # 得到一组_x1/_y1/_x2/_y2
        _x1 = (_box[:, 0])
        _y1 = (_box[:, 1])
        _x2 = (_box[:, 2])
        _y2 = (_box[:, 3])

        # 得到一组ow/oh
        ow = _x2 - _x1
        oh = _y2 - _y1

        # 得到一组x1/y1/x2/y2/cls
        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        cls = _cls[idxs][:, 0]

        # 合并成一组组的x1,y1,x2,y2
        boxes = [x1, y1, x2, y2, cls]
        # print(np.shape(boxes))

        # nms之前将一组组的x1,y1,x2,y2转换成一组组的box
        boxes = utils.nms(np.array(boxes).T, nms)
        return boxes

if __name__ == '__main__':
    a = time.time()
    with torch.no_grad() as grad:
        detector = Detector()
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(r"F:\咕泡学院项目实战\Face_Reconition\v0200f880000br7o5i0a2peq2du5t3m0.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的height
        print(w,h)

        while True:
            x = time.time()
            ret, photo = cap.read()
            if ret:
                b, g, r = cv2.split(photo)
                img = cv2.merge([r, g, b])
            else:
                break
            im = Image.fromarray(img, "RGB")

            boxes = detector.detect(im,zoom_factor=0.7,p_conf=0.4,p_nms=0.5,r_conf=0.5,r_nms=0.5,o_conf=0.9,o_nms=0.5)

            for i, box in enumerate(boxes):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(photo,(x1,y1),(x2,y2),(0,0,255),3)#BG

            cv2.imshow("capture", photo)

            y = time.time()
            timelag=y - x
            print(timelag)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()  # 关闭视频
        cv2.destroyAllWindows()  # 关闭窗口
        b = time.time()
        print(b - a)