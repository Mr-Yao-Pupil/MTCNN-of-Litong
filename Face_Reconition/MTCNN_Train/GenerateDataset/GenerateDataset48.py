"""
此文件用于生成O网络训练和测试所需的数据集
"""

from PIL import Image
import os
import utils
import numpy as np
import traceback

ann_path = r'F:\Datasets\CelebA\Anno\list_bbox_celeba.txt'
img_path = r'F:\Datasets\CelebA\Img\img_celeba.7z\img_celeba'
save_path = r'F:\Datasets\MTCNN'

for face_size in [48]:
    # --------------------训练和测试的图片文件存储------------------------
    positive_img_dir = os.path.join(save_path, str(face_size), 'positive')
    negotive_img_dir = os.path.join(save_path, str(face_size), 'negotive')
    part_img_dir = os.path.join(save_path, str(face_size), 'part')

    # ------------------如果文件夹不存在就创建文件夹-------------------------
    for dir_path in [positive_img_dir, negotive_img_dir, part_img_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # -----------------------训练和测试的标签文件----------------------------
    positive_anno_dir = os.path.join(save_path, str(face_size), 'positive.txt')
    negotive_anno_dir = os.path.join(save_path, str(face_size), 'negotive.txt')
    part_anno_dir = os.path.join(save_path, str(face_size), 'part.txt')

    # ------------------------------计数---------------------------------
    positive_count = 0
    negative_count = 0
    part_count = 0

    # -----------------------打开写入的标签文件--------------------------
    try:
        positive_anno_file = open(positive_anno_dir, 'w')
        negotive_anno_file = open(negotive_anno_dir, 'w')
        part_anno_file = open(part_anno_dir, 'w')

        # ---------------按行读取标签文件中的信息----------------------
        for i, line in enumerate(open(ann_path)):
            if i < 2:
                continue

            try:
                # -------------将每一行信息按空格进行分割装入列表---------------
                strs = line.strip().split(' ')
                strs = list(filter(bool, strs))

                # ------------分离文件名并生成图片的路径---------------------
                img_filename = strs[0].strip()
                print(img_filename)
                img_file = os.path.join(img_path, img_filename)

                # ----------------打开图片----------------------------------
                with Image.open(img_file) as img:
                    # ----------------获取图片的四个坐标点---------------
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)
                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    # 标注不准确，对人脸坐标进行适量偏移
                    x1 = int(x1 + w * 0.12)
                    y1 = int(y1 + h * 0.1)
                    x2 = int(x1 + w * 0.9)
                    y2 = int(y1 + h * 0.85)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    # ---------------将坐标进行封装----------------
                    boxes = [[x1, y1, x2, y2]]

                    # --------------求取标签的中心点-----------------
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    for _ in range(5):
                        # --------------随机生成偏移量-------------------
                        w_ = np.random.randint(-w * 0.2, w * 0.2)  # 框的横向偏移范围：向左、向右移动了20%
                        h_ = np.random.randint(-h * 0.2, h * 0.2)

                        # ---------------对中心点进行偏移----------------------
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # ---------------随机生成边长--------------------
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                        # ---------------计算偏移后的四个坐标点并形成正方形-----------------
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        # ------------------对生成的框进行封装--------------------------
                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # ------------------对偏移量进行零中心化-----------------------
                        offset_x1 = (x1 - x1_) / side_len  # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # ----------------------将该区域裁剪并resize-------------------
                        face_crop = img.crop(crop_box)  # “抠图”，crop剪下框出的图像
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        # ------------------计算iou----------------------
                        iou = utils.iou(crop_box, np.array(boxes))[0]

                        # --------------------根据IOU进行分类----------------------------
                        # --------------------生成正样本-------------------------
                        if iou > 0.6:  # 正样本；原为0.65
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2))
                            positive_anno_file.flush()  # flush：将缓存区的数据写入文件
                            face_resize.save(os.path.join(positive_img_dir, "{0}.jpg".format(positive_count)))  # 保存
                            positive_count += 1

                        # ------------------生成部分样本训练集---------------------------
                        elif iou > 0.4:  # 部分样本；原为0.4
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,
                                    offset_y2))  # 写入txt文件
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_img_dir, "{0}.jpg".format(part_count)))
                            part_count += 1

                        # ------------------生成负样本-----------------------------
                        if iou < 0.29:
                            negotive_anno_file.write(
                                f'negoyive\\{negative_count}.jpg {0} {0} {0} {0} {0}\n'
                            )
                            negotive_anno_file.flush()
                            face_resize.save(os.path.join(negotive_img_dir, f'{negative_count}.jpg'))
                            negative_count += 1

                        _boxes = np.array(boxes)
                        # --------------------获取标签框---------------------------
                        prop_boxes = np.array(boxes)[0]

                    for _ in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)

                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if x_ + side_len <= prop_boxes[0] or y_ + side_len <= prop_boxes[1] or x_ >= prop_boxes[2
                        ] or y_ > prop_boxes[3]:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negotive_anno_file.write(
                                f'negotive\\{negative_count}.jpg {0} {0} {0} {0} {0}\n'
                            )
                            negotive_anno_file.flush()
                            face_resize.save(os.path.join(negotive_img_dir, f'{negative_count}.jpg'))
                            negative_count += 1


            except Exception:
                traceback.print_exc()

    finally:
        negotive_anno_file.close()
        positive_anno_file.close()
        part_anno_file.close()
