from processing_tools import *
# import matplotlib.pyplot as plt
import dlib

def draw_landmarks_on_flow(flow_rgb_resized, land_onset):

    img_draw = flow_rgb_resized.copy()
    h, w = img_draw.shape[:2]
    
    # 关键点总数 68 个
    num_landmarks = 68
    
    point_color = (0, 255, 0)    # 绿色点
    point_radius = 2             # 点大小
    point_thickness = -1         # -1 表示实心圆
    
    for i in range(num_landmarks):
        x = int(land_onset[2 * i])
        y = int(land_onset[2 * i + 1])
        
        # 确保坐标在图像范围内，防止越界报错
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img_draw, (x, y), point_radius, point_color, point_thickness)
    
    return img_draw


onset_path = './test_pic/onset.png'

apex_path = './test_pic/apex.png'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


img_size = 224

resize = 56

aligned_onset, aligned_apex, land_onset, flow_rgb_resized = data_processing(onset_path, apex_path, detector, predictor, box_enlarge=2.1, align_size=img_size, resize=resize)

img_draw = draw_landmarks_on_flow(flow_rgb_resized, land_onset)

cv2.imwrite('onset.jpg', aligned_onset)
cv2.imwrite('apex.jpg', aligned_apex)
cv2.imwrite('oflowxxxx.jpg', flow_rgb_resized)
cv2.imwrite('oflowxxxxxxx.jpg', img_draw)

