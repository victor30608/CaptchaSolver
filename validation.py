from glob import glob
from solver import CaptchaSolver
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from utils import *


c_solver = CaptchaSolver(yolo_train_path='weights/yolov8_train/train.pt',
                         yolo_object_path='weights/yolov8_objects/objects_m.pt')

data_path = "./val_dataset"
val_names = [x.split('.')[0] for x in os.listdir(data_path)]
val_names = list(sorted(set(val_names)))
match_counter = 0

for name in tqdm(val_names):
    img = cv2.imread(os.path.join(data_path, name + '.jpg'))
    markup = load_json(os.path.join(data_path, name + '.json'))
    answer = find_answer_xy(markup)[0]



    pred_x, pred_y = c_solver(img, visualization=False)

    if pred_x is None:
        continue
    answer = np.array(answer)

    dist = euc_distance(np.array([pred_x, pred_y]), answer)

    if dist < 50:
        match_counter += 1

print(f"Accuracy = {100 * match_counter / len(val_names):.3f} %")
