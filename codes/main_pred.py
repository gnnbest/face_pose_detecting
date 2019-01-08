# -*-coding:utf-8 -*-

# 预测人脸朝向
# 输入图片放置在pics目录下
# 输出图片，画出人脸朝向的，放置在output目录下

import sys, os, argparse

import numpy as np
import cv2

from mtcnn_detector import*
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mtcnn_detector = MtcnnDetector(model_folder=os.path.join(WORK_DIR, 'models'),
                               ctx=mx.gpu(1), num_worker = 2 , accurate_landmark = False)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import time
import datasets, hopenet, utils



class Face_Pose_Estimation():

    def __init__(self):

        # 加载网路模型
        self.snapshot_path = os.path.join(WORK_DIR, 'models/hopenet_robust_alpha1.pkl')
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        # Load snapshot
        saved_state_dict = torch.load(self.snapshot_path)
        #saved_state_dict = torch.load(self.snapshot_path, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(saved_state_dict)
        # 设置模型使用gpu or cpu
        self.gpu = 1
        self.model.cuda(self.gpu)
        #self.model.cpu()

        self.model.eval()
        # 图像预处理，送入模型之前的处理
        self.transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(self.gpu)
        #self.idx_tensor = torch.FloatTensor(self.idx_tensor)

    
    def pred_face_pose(self, faces_boxes, faces_points, image):

        cv2_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i, face_box in enumerate(faces_boxes):

            x_min, y_min, x_max, y_max = int(float(face_box[0])), int(float(face_box[1])), \
                                         int(float(face_box[2])), int(float(face_box[3]))
            x_min_tmp = x_min
            x_max_tmp = x_max
            y_min_tmp = y_min
            y_max_tmp = y_max

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            #x_min -= 0.6 * 0.3 * bbox_width
            #x_max += 0.6 * 0.3 * bbox_width
            #y_min -= 2 * 0.3 * bbox_height
            #y_max += 0.6 * 0.3 * bbox_height
            x_min -= 7
            x_max += 7
            y_min -= 7
            y_max += 5
            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(frame.shape[1], x_max))
            y_max = int(min(frame.shape[0], y_max))
            # Crop image
            img = cv2_frame[y_min:y_max, x_min:x_max]

            img = Image.fromarray(img)

            # Transform
            img = self.transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(self.gpu)
            #img = Variable(img)

            yaw, pitch, roll = self.model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            
            yaw_tmp = yaw_predicted.cpu()
            yaw_numpy = yaw_tmp.detach().numpy()
            max_yaw_value = max(yaw_numpy[0].tolist())
            
            # yaw方向得分较低的就不要的，为了减少误判
            if(max_yaw_value <= 0.06):
                continue

            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

            # Print new frame with cube and axis
            l_eye = (faces_points[i][0], faces_points[i][5])
            r_eye = (faces_points[i][1], faces_points[i][6])

            # 画人脸朝向立体框

            cube = utils.plot_pose_cube_draw(image, yaw_predicted, pitch_predicted,
                                                 roll_predicted, (l_eye, r_eye), (x_min_tmp, y_min_tmp),
                                                 ((x_min_tmp + x_max_tmp) / 2, (y_min_tmp + y_max_tmp) / 2),
                                                 cube_length= bbox_width)




if __name__ == '__main__':
    # 待预测的图像
    pics_path = os.path.join(WORK_DIR, 'pics')
    # 画人脸朝向的图像存放目录
    out_dir = os.path.join(WORK_DIR, 'output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    face_pose_detector = Face_Pose_Estimation()

    for root, dirs, files in os.walk(pics_path):

        for pic_name in files:
            # Start processing frame with bounding box
            frame = cv2.imread(os.path.join(root, pic_name))
            print(frame.shape)

            # mtcnn检测人脸
            faces_results = mtcnn_detector.detect_face(frame)
            if faces_results is None:
                continue

            else:
                faces_boxes = faces_results[0]
                faces_points = faces_results[1]
                # 根据mtcnn检测出来的人脸框进行人脸朝向预测
                face_pose_detector.pred_face_pose(faces_boxes, faces_points, frame)

                cv2.imwrite(os.path.join(out_dir, pic_name), frame)






