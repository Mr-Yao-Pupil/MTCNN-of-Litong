"""
此文件为人脸检测和检测是所需要用到的工具文件，包括：
1、IOU计算
2、NMS去重
3、将输出的框变为正方形
4、图形化观察
"""
import numpy as np


def iou(box, boxes, isMin=False):
    """
    此函数用于计算一个框与其他框的IOU计算
    :param box: 需要计算的第一个框，[x1, y1, x2, y2]
    :param boxes: 其他框，数据共两个维度，第一个维度为传入框的数量，第二维度为[x1, y1, x2, y2]
    :param isMin: 是否采用最小框的IOU，默认为False
    :return: 第一个框与其他框的置信度数组，共两个维度，第一维度为其他框的数量，第二维度为第一个框与其他框的IOU
    """
    # 分别计算传入所有框的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 获取重合区域的四个坐标点
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])

    # 计算重合区域的宽、高，如果没有重合区域则取0
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    # 计算重合区域的面积
    inter_area = w * h

    if isMin:
        return np.true_divide(inter_area, np.minimum(box_area, boxes_area))
    else:
        return np.true_divide(inter_area, boxes_area + boxes_area - inter_area)


def nms(boxes, isMin=False, thresh=0.3):
    """
    此函数用于去除重合部分过多的框
    :param boxes: 传入所有的框
    :param isMin: IOU是否采用最小框的IOU， 默认为False
    :param thresh: 设定重合度的阈值， 默认为0.3
    :return: 去重后的所有框
    """
    # 如果没有传入框，返回空数组
    if boxes.shape[0] == 0:
        return np.array([])

    # 按照置信度对数组从小到大进行排序
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    return_boxes = []

    # 取出置信度最大的框， 去掉与该框重合度高于阈值的框。PS：若阈值设定的越大，则留下的两个框越近；反之越远
    while _boxes.shape[0] > 1:
        first_box = _boxes[0]
        other_boxes = _boxes[1:]

        return_boxes.append(first_box)
        index = np.where(iou(first_box, other_boxes, isMin) < thresh)
        _boxes = other_boxes[index]

    if _boxes.shape[0] > 0:
        return_boxes.append(_boxes[0])

    return np.stack(return_boxes)


def to_square(box):
    """
    该函数将输入的框转化为正方形
    :param box: 传入的所有框，其数据包含两个维度，第一维度为传入框的数量，第二维度为[x1, y1, x2, y2]
    :return: 转化后正方形的框，其数据包含两个维度，第一维度为传入框的数量，第二维度为[x1, y1, x2, y2]
    """
    squre_box = box.copy()

    if box.shape[0] == 0:
        return np.array([])

    h = box[:, 3] - box[:, 1]
    w = box[:, 2] - box[:, 0]

    max_side = np.maximum(h, w)

    squre_box[:, 0] = box[:, 0] + 0.5 * w - 0.5 * max_side
    squre_box[:, 1] = box[:, 1] + 0.5 * h - 0.5 * max_side
    squre_box[:, 2] = squre_box[:, 0] + max_side
    squre_box[:, 3] = squre_box[:, 1] + max_side

    return squre_box
