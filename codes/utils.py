import numpy as np
import torch
from torch.utils.serialization import load_lua
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)


def plot_pose_cube_draw(img, yaw, pitch, roll, (l_eye, r_eye), (left_up_x, left_up_y),
                            (mt_ct_x, mt_ct_y), cube_length=30.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180

    face_x = left_up_x
    face_y = left_up_y

    x1 = cube_length * (cos(y) * cos(r)) + face_x
    y1 = cube_length * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = cube_length * (-cos(y) * sin(r)) + face_x
    y2 = cube_length * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = cube_length * (sin(y)) + face_x
    y3 = cube_length * (-cos(y) * sin(p)) + face_y

    point0_center = (int(face_x), int(face_y))
    point1 = (int(x1),int(y1))
    point2 = (int(x2),int(y2))
    point3 = (int(x2+x1-face_x),int(y2+y1-face_y))

    point4 = (int(x3),int(y3))
    point5 = (int(x1+x3-face_x),int(y1+y3-face_y))
    point6 = (int(x2+x3-face_x),int(y2+y3-face_y))
    point7 = (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y))

    l_eye_end_t = ((point4[0] - (point0_center[0] - l_eye[0])), (point4[1] - (point0_center[1] - l_eye[1])))
    r_eye_end_t = ((point4[0] - (point0_center[0] - r_eye[0])), (point4[1] - (point0_center[1] - r_eye[1])))
    l_eye_end = (int((l_eye[0] + l_eye_end_t[0])/2), int((l_eye[1] + l_eye_end_t[1])/2))
    r_eye_end = (int((r_eye[0] + r_eye_end_t[0])/2), int((r_eye[1] + r_eye_end_t[1])/2))

    center_x = (min(point0_center[0],point1[0],point2[0],point3[0]) +
               max(point0_center[0],point1[0],point2[0],point3[0]))/2
    center_y = (min(point0_center[1],point1[1],point2[1],point3[1]) +
               max(point0_center[1],point1[1],point2[1],point3[1]))/2

    inter_x = mt_ct_x - center_x
    inter_y = mt_ct_y - center_y

    point0_center = (int(face_x) + inter_x, int(face_y) + inter_y)
    point1 = (int(x1) + inter_x, int(y1) + inter_y)
    point2 = (int(x2) + inter_x, int(y2) + inter_y)
    point3 = (int(x2 + x1 - face_x) + inter_x, int(y2 + y1 - face_y)+inter_y)

    point4 = (int(x3) + inter_x, int(y3) + inter_y)
    point5 = (int(x1 + x3 - face_x) + inter_x, int(y1 + y3 - face_y) + inter_y)
    point6 = (int(x2 + x3 - face_x) + inter_x, int(y2 + y3 - face_y) + inter_y)
    point7 = (int(x3 + x1 + x2 - 2 * face_x) + inter_x, int(y3 + y2 + y1 - 2 * face_y) + inter_y)

    # Draw eye lines
    cv2.line(img, l_eye, l_eye_end, (0, 255, 0), 3)
    cv2.line(img, r_eye, r_eye_end, (0, 255, 0), 3)

    # Draw base in red
    cv2.line(img, point0_center, point1, (0,0,255),3)
    cv2.line(img, point0_center, point2,(0,0,255),3)
    cv2.line(img, point1, point3, (0, 0, 255), 3)
    cv2.line(img, point2, point3, (0,0,255), 3)

    # Draw pillars in blue
    cv2.line(img, point0_center, point4, (255,0,0), 2)
    cv2.line(img, point1, point5, (255,0,0), 2)
    cv2.line(img, point2, point6, (255,0,0), 2)
    cv2.line(img, point3, point7, (255,0,0), 2)
    # Draw top in green
    cv2.line(img, point4, point5, (0,255,0), 2)
    cv2.line(img, point4, point6, (0,255,0), 2)
    cv2.line(img, point5, point7, (0,255,0), 2)
    cv2.line(img, point6, point7, (0,255,0), 2)

    return [list(point0_center), list(point1), list(point2), list(point3),
            list(point4), list(point5), list(point6), list(point7),
            list(l_eye), list(l_eye_end), list(r_eye), list(r_eye_end)]



def draw_line(img, pat, raw, total_box):

    x_center = int((total_box[0] + total_box[2]) / 2)
    y_center = int((total_box[1] + total_box[3]) / 2)

    L = 60 * abs(raw) / 90
#    print pat, L, math.pi, type(pat), type(L),type(math.pi)
    y_d = int(math.sin(math.pi * abs(pat) / 180.) * L)
    if pat <= 0:
        y = y_center + y_d
    else:
        y = y_center - y_d
    x_d = int(math.cos(math.pi * abs(pat) / 180.) * L)
    if raw <= 0:
        x = x_center + x_d
    else:
        x = x_center - x_d
    cv2.circle(img, (x_center, y_center), 5, (0, 0, 155), -1)
    cv2.line(img, (x_center, y_center), (int(x), int(y)), (0, 255, 0), 2)




def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
