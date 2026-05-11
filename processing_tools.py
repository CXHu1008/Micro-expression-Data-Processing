import math
import numpy as np
import cv2
import os
import dlib
import pandas as pd

# =========================
# 关键点说明
# 脸周:0-16
# 左眉:17-21
# 右眉:22-26
# 鼻子:27-35
# 左眼:36-41
# 右眼:42-47
# 嘴巴:48-67
# =========================


def align_face(img, img_land, box_enlarge, img_size):

    leftEye0 = (img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] +
                img_land[2 * 40] + img_land[2 * 41] + img_land[2 * 36]) / 6.0

    leftEye1 = (img_land[2 * 37 + 1] + img_land[2 * 38 + 1] +
                img_land[2 * 39 + 1] + img_land[2 * 40 + 1] +
                img_land[2 * 41 + 1] + img_land[2 * 36 + 1]) / 6.0

    rightEye0 = (img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] +
                 img_land[2 * 46] + img_land[2 * 47] + img_land[2 * 42]) / 6.0

    rightEye1 = (img_land[2 * 43 + 1] + img_land[2 * 44 + 1] +
                 img_land[2 * 45 + 1] + img_land[2 * 46 + 1] +
                 img_land[2 * 47 + 1] + img_land[2 * 42 + 1]) / 6.0

    deltaX = float(rightEye0 - leftEye0)
    deltaY = float(rightEye1 - leftEye1)

    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)

    sinVal = deltaY / l
    cosVal = deltaX / l

    mat1 = np.asmatrix([
        [cosVal, sinVal, 0],
        [-sinVal, cosVal, 0],
        [0, 0, 1]
    ])

    mat2 = np.asmatrix([
        [leftEye0.item(), leftEye1.item(), 1],
        [rightEye0.item(), rightEye1.item(), 1],
        [img_land[2 * 30].item(), img_land[2 * 30 + 1].item(), 1],
        [img_land[2 * 48].item(), img_land[2 * 48 + 1].item(), 1],
        [img_land[2 * 54].item(), img_land[2 * 54 + 1].item(), 1]
    ])

    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    ##为了截取更多额头、脸部框选区域上移
    face_w = float(max(mat2[:, 0]) - min(mat2[:, 0]))
    face_h = float(max(mat2[:, 1]) - min(mat2[:, 1]))   
    cy -= 0.13 * face_h

    if float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1])):
        halfSize = 0.5 * box_enlarge * float(max(mat2[:, 0]) - min(mat2[:, 0]))
    else:
        halfSize = 0.5 * box_enlarge * float(max(mat2[:, 1]) - min(mat2[:, 1]))

    scale = (img_size - 1) / 2.0 / halfSize

    mat3 = np.asmatrix([
        [scale, 0, scale * (halfSize - cx)],
        [0, scale, scale * (halfSize - cy)],
        [0, 0, 1]
    ])

    mat = mat3 * mat1

    aligned_img = cv2.warpAffine(
        img,
        mat[0:2, :],
        (img_size, img_size),
        cv2.INTER_LINEAR,
        borderValue=(128, 128, 128)
    )

    land_3d = np.ones((int(len(img_land) / 2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))

    mat_land_3d = np.asmatrix(land_3d)

    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land


# =========================
# 检测关键点 + 对齐
# =========================
def img_pre_dlib(detector, predictor, img_path,
                 box_enlarge=2.5,
                 img_size=128):

    img = cv2.imread(img_path)

    img_dlib = dlib.load_rgb_image(img_path)

    dets = detector(img_dlib, 1)

    shape = predictor(img_dlib, dets[0])

    ldm = np.matrix([[p.x, p.y] for p in shape.parts()])
    ldm = ldm.reshape(136, 1)

    aligned_img, new_land = align_face(
        img,
        ldm,
        box_enlarge,
        img_size
    )

    return aligned_img, new_land




# =========================
# TV-L1 光流
# =========================

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def computeStrain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis=1)
    v_y = v - pd.DataFrame(v).shift(-1, axis=0)
    u_y = u - pd.DataFrame(u).shift(-1, axis=0)
    v_x = v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 0.25 * (u_y + v_x)**2).ffill(axis=1).ffill(axis=0))
    return os

def compute_optical_flow_tvl1(img1, img2, resize=56):
    # frame1 = cv2.imread(img1, 0)
    # frame2 = cv2.imread(img2, 0)
    # frame1= cv2.resize(frame1, (224, 224))
    # frame2 = cv2.resize(frame2, (224, 224))
    if len(img1.shape) == 3:
        frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if len(img2.shape) == 3:
        frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(frame1, frame2, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u, v = pol2cart(magnitude, angle)
    os = computeStrain(u, v)
    
    final_u = cv2.resize(u, (resize, resize))
    final_v = cv2.resize(v, (resize, resize))
    final_os = cv2.resize(os, (resize, resize))
    # final_u = u
    # final_v = v
    # final_os = os


    if ((np.max(final_u) - np.min(final_u))==0):
        normalized_u = final_u.astype(np.uint8)
    else:
        normalized_u = ((final_u - np.min(final_u)) / (np.max(final_u) - np.min(final_u)) * 255).astype(np.uint8)

    if ((np.max(final_v) - np.min(final_v))==0):
        normalized_v = final_v.astype(np.uint8)
    else:
        normalized_v = ((final_v - np.min(final_v)) / (np.max(final_v) - np.min(final_v)) * 255).astype(np.uint8)

    if ((np.max(final_os) - np.min(final_os))==0):
        normalized_os = final_os.astype(np.uint8)
    else:
        normalized_os = ((final_os - np.min(final_os)) / (np.max(final_os) - np.min(final_os)) * 255).astype(np.uint8)


    return np.concatenate(
            (normalized_os.reshape(*normalized_os.shape, 1), normalized_v.reshape(*normalized_v.shape, 1), normalized_u.reshape(*normalized_u.shape, 1)),
            axis=2
        )


# =========================
# 总处理流程
# =========================
def data_processing(onset_path,
                    apex_path,
                    detector,
                    predictor,
                    box_enlarge=2.0,
                    align_size=256,
                    resize=128):

    aligned_onset, land_onset = img_pre_dlib(
        detector,
        predictor,
        onset_path,
        box_enlarge=box_enlarge,
        img_size=align_size
    )

    aligned_apex, _ = img_pre_dlib(
        detector,
        predictor,
        apex_path,
        box_enlarge=box_enlarge,
        img_size=align_size
    )

    flow = compute_optical_flow_tvl1(
        aligned_onset,
        aligned_apex,
        resize=resize
    )

    # flow_resized = cv2.resize(flow, (resize, resize))

    # 同步缩放 landmark 坐标
    scale = resize / align_size
    land_onset_resized = land_onset * scale

    return (
        aligned_onset,
        aligned_apex,
        land_onset_resized,
        flow
    )
