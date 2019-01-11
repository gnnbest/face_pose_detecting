import sys, os, argparse

import numpy as np
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
import time


WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from mtcnn_detector import*
mtcnn_detector = MtcnnDetector(model_folder=os.path.join(WORK_DIR, 'model'), ctx=mx.gpu(0), num_worker=1,
                                   accurate_landmark=False)


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import datasets, hopenet, utils



class Face_Pose_Estimation():

    def __init__(self, gpu = 1):

        # Load snapshot
        self.gpu = gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        self.snapshot_path = os.path.join(WORK_DIR, 'model/hopenet_robust_alpha1.pkl')
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        saved_state_dict = torch.load(self.snapshot_path) # gpu
        #saved_state_dict = torch.load(self.snapshot_path, map_location=lambda storage, loc: storage) # cpu
        self.model.load_state_dict(saved_state_dict)

        batchsize = 5
        x = Variable(torch.randn(batchsize, 3, 224, 224))
        orch_out = torch.onnx.export_to_pretty_string(self.model, x, "pnas.onnx", export_params=True)

        self.model.cuda(self.gpu)

        self.model.eval()

        self.transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self.idx_tensor = [idx for idx in range(66)]

        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(self.gpu)





    def pred_batch_imgs(self, batch_imgs, faces_points, face_boxes, image):

        batch_imgs = Variable(batch_imgs).cuda(self.gpu)

        yaw, pitch, roll = self.model(batch_imgs)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)

        yaw_tmp = yaw_predicted.cpu()
        yaw_numpy = yaw_tmp.detach().numpy()
        for i, facepoint in enumerate(faces_points):
            max_yaw_value = max(yaw_numpy[i].tolist())
            print(max_yaw_value)
            if(max_yaw_value <= 0.06):
                continue

            yaw_predicted_value = torch.sum(yaw_predicted.data[i] * self.idx_tensor) * 3 - 99
            pitch_predicted_value = torch.sum(pitch_predicted.data[i] * self.idx_tensor) * 3 - 99
            roll_predicted_value = torch.sum(roll_predicted.data[i] * self.idx_tensor) * 3 - 99

            #Print new frame with cube and axis
            l_eye = (facepoint[0], facepoint[5])
            r_eye = (facepoint[1], facepoint[6])

            x_min = face_boxes[i][0]
            x_max = face_boxes[i][1]
            y_min = face_boxes[i][2]
            y_max = face_boxes[i][3]
            utils.plot_pose_cube_pre(image, yaw_predicted_value, pitch_predicted_value, roll_predicted_value, (x_min + x_max) / 2,
                                 (y_min + y_max) / 2, size=face_boxes[i][4])


    def pred_face_pose(self, faces_boxes, faces_points, image):

        cv2_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        batchsize = 5

        batch_num = len(faces_boxes) / batchsize
        left_num = len(faces_boxes) % batchsize

        for j in range(0, batch_num + 1):

            start = j*batchsize
            end = j*batchsize + batchsize

            if(j == batch_num):
                start = j*batchsize
                end = j*batchsize + left_num

            batch_imgs = np.zeros((1,3,224,224))
            face_points_tmp = []
            face_box_tmp = []
            for i in range(start, end):

                face_points_tmp.append(faces_points[i])

                face_box = faces_boxes[i]

                x_min, y_min, x_max, y_max = int(float(face_box[0])), int(float(face_box[1])), \
                                             int(float(face_box[2])), int(float(face_box[3]))

                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                # x_min -= 3 * bbox_width / 4
                # x_max += 3 * bbox_width / 4
                # y_min -= 3 * bbox_height / 4
                # y_max += bbox_height / 4
                x_min -= 7
                x_max += 7
                y_min -= 7
                y_max += 5
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = self.transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

                face_box_tmp.append((x_min, x_max, y_min, y_max, bbox_width))

                img = img.cpu()

                if(i == start):
                    batch_imgs = img.numpy()

                if(i > start):
                    batch_imgs = np.row_stack((batch_imgs, img.numpy()))

                if(i == end - 1):
                    tensor_batch_imgs = torch.from_numpy(batch_imgs)
                    self.pred_batch_imgs(tensor_batch_imgs, face_points_tmp, face_box_tmp, image)







if __name__ == '__main__':

    pics_path = os.path.join(WORK_DIR, 'pics/back_pics')

    out_dir = os.path.join(WORK_DIR, 'output/back_pics_0.06')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    face_pose_detector = Face_Pose_Estimation(0)


    for root, dirs, files in os.walk(pics_path):

        for pic_name in files:
            # Start processing frame with bounding box
            frame = cv2.imread(os.path.join(root, pic_name))
            print(pic_name)

            pre = time.time()
            faces_results = mtcnn_detector.detect_face(frame)

            # mtcnn
            if faces_results is None:
                continue

            else:
                faces_boxes = faces_results[0]
                faces_points = faces_results[1]

                #pre = time.time()
                face_pose_detector.pred_face_pose(faces_boxes, faces_points, frame)
                print('time:', time.time() - pre)

                cv2.imwrite(os.path.join(out_dir, pic_name), frame)








