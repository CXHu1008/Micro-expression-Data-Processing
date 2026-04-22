import math
import numpy as np
import cv2
import os
import dlib


##脸周的关键点编号是0-16，左眉毛是17-21，右眉毛是22-26，鼻子是27-35，左眼是36-41，右眼是42-47，嘴巴是48-67
def align_face(img, img_land, box_enlarge, img_size):

    leftEye0 = (img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] + img_land[2 * 40] + img_land[2 * 41] +
                img_land[2 * 36]) / 6.0
    leftEye1 = (img_land[2 * 37 + 1] + img_land[2 * 38 + 1] + img_land[2 * 39 + 1] + img_land[2 * 40 + 1] +
                img_land[2 * 41 + 1] + img_land[2 * 36 + 1]) / 6.0
    rightEye0 = (img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] + img_land[2 * 46] + img_land[2 * 47] +
                 img_land[2 * 42]) / 6.0
    rightEye1 = (img_land[2 * 43 + 1] + img_land[2 * 44 + 1] + img_land[2 * 45 + 1] + img_land[2 * 46 + 1] +
                 img_land[2 * 47 + 1] + img_land[2 * 42 + 1]) / 6.0

    deltaX = float(rightEye0 - leftEye0)
    deltaY = float(rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    #print("pupil distance:",l)
    #print(deltaX,deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.asmatrix([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])

    mat2 = np.asmatrix([[leftEye0.item(), leftEye1.item(), 1], [rightEye0.item(), rightEye1.item(), 1], [img_land[2 * 30].item(), img_land[2 * 30 + 1].item(), 1],
                   [img_land[2 * 48].item(), img_land[2 * 48 + 1].item(), 1], [img_land[2 * 54].item(), img_land[2 * 54 + 1].item(), 1]])
    
    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.asmatrix([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1

    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))

    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.asmatrix(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land



#检测人脸关键点并对齐
def img_pre_dlib(detector, predictor, img_path, box_enlarge=2.5, img_size=128):

    img = cv2.imread(img_path)#[:,80:560]

    img_dlib = dlib.load_rgb_image(img_path)
    dets = detector(img_dlib, 1)
    shape = predictor(img_dlib, dets[0])
    ldm = np.matrix([[p.x, p.y] for p in shape.parts()])
    ldm=ldm.reshape(136,1)

    aligned_img, new_land = align_face(img, ldm, box_enlarge, img_size)
    return aligned_img, new_land



def compute_optical_flow_tvl1(prev_img, next_img, landmark):
    """
    TV-L1 光流 + 完整人脸mask（脸轮廓 + 眉毛，连成一个大轮廓）
    眉毛所有点向上 expand_px 像素，外部光流衰减
    """

    if len(prev_img.shape) == 3:
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_img
        next_gray = next_img

    # TV-L1 光流
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(prev_gray, next_gray, None)

    h, w = prev_img.shape[:2]
    expand_px = int(h / 10)


    # ===================== 构建完整人脸闭合轮廓 =====================
    contour = []
    # 1. 左眉毛：21→20→19→18→17（从左眉内侧回到外侧，闭合轮廓）
    for i in range(21, 16, -1):
        x = int(landmark[2*i])
        y = int(landmark[2*i + 1]) - expand_px  # 整体上移
        contour.append([x, y])

    # 2. 脸部轮廓：0→1→…→16（右侧脸颊 → 下巴 → 左侧脸颊）
    for i in range(0, 17):
        x = int(landmark[2*i])
        y = int(landmark[2*i + 1])
        contour.append([x, y])
    # 3. 右眉毛：22→23→24→25→26（从右眉外侧到内侧）
    for i in range(26, 22, -1):
        x = int(landmark[2*i])
        y = int(landmark[2*i + 1]) - expand_px  # 根据眉毛关键点向上偏移
        contour.append([x, y])

    # 转成轮廓格式
    contour = np.array([contour], dtype=np.int32)

    # 生成mask：内部=255，外部=0
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, contour, 255)

    # 非面部光流抑制
    flow[mask == 0] /= 3

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_img)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    x = np.mean(flow_rgb[mask == 0]) * 3
    flow_rgb[mask == 0] = np.clip(flow_rgb[mask == 0], a_min=None, a_max=x)

    avg_magnitude = np.mean(mag[mask != 0]) if np.any(mask != 0) else 0

    return flow, avg_magnitude, flow_rgb



def data_processing(onset_path, apex_path, detector, predictor, box_enlarge=2.5, align_size=256, resize = 128):
    aligned_onset, land_onset = img_pre_dlib(detector, predictor, onset_path, box_enlarge=box_enlarge, img_size=align_size)
    aligned_apex, _ = img_pre_dlib(detector, predictor, apex_path, box_enlarge=box_enlarge, img_size=align_size)

    flow, avg_magnitude, flow_rgb = compute_optical_flow_tvl1(aligned_onset, aligned_apex, land_onset)

    flow_rgb_resized = cv2.resize(flow_rgb, (resize, resize))

    return aligned_onset, aligned_apex, land_onset, flow_rgb_resized
