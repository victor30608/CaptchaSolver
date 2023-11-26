import os
from utils import *
from solver import CaptchaSolver
import cv2

c_solver = CaptchaSolver(yolo_train_path='weights/yolov8_train/train.pt',
                         yolo_object_path='weights/yolov8_objects/objects.pt')

# img = cv2.imread('examples/4498_jpg.rf.e0d11fdd57882734a021c68d53e6c1bd.jpg')
# c_solver(img)

path = r'P:\mixed_train_to_the_coordinates_dataset\hard_mode'
labels_path = r'P:\WORK\CaptchaSolver\val_dataset'
for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    answer = find_answer_xy(load_json(os.path.join(labels_path, file.replace('.jpg', '.json'))))
    ans_x, ans_y = answer[0]
    pred_x, pred_y = c_solver(img, label=(int(ans_x), int(ans_y)))

    print(file)
    print(euc_distance(np.array([pred_x, pred_y]), answer[0]))