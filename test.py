from processing_tools import *

import dlib


onset_path = './test_pic/apex.jpg'

apex_path = './test_pic/apex.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


img_size = 256
resize = 256

aligned_onset, aligned_apex, land_onset, flow_rgb_resized = data_processing(onset_path, apex_path, detector, predictor, box_enlarge=2.4, align_size=img_size, resize=resize)

cv2.imwrite('oflow.jpg', flow_rgb_resized)

