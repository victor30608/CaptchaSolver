import cv2
import torch

from yolo import YoloDetector
from img_sim import ImgSim
from label_helper import LabelHelper
from utils import *
import numpy as np


class CaptchaSolver:

    def __init__(self, yolo_train_path, yolo_object_path):
        self.lh = LabelHelper()
        self.yolo = YoloDetector(train_weights_path=yolo_train_path,
                                 objects_weights_path=yolo_object_path)
        self.img_sim = ImgSim()

    def __call__(self, img, visualization=True, label=None):
        first_screen, captcha_screens, prompt, first_screen_w_promt = self.lh(img)

        all_objects = self.yolo.detect_objects(first_screen_w_promt)
        first_screen_bboxes, prompt_bboxes = [], []
        for bbox in all_objects:
            if max(bbox[1], bbox[3]) > 200:
                bbox[1] -=200
                bbox[3] -=200
                prompt_bboxes.append(bbox)
            else:
                first_screen_bboxes.append(bbox)

        first_screen_bboxes = list(sorted(first_screen_bboxes, key=lambda x: x[0]))

        #prompt_bboxes = self.yolo.detect_objects(prompt)

        if len(prompt_bboxes) != 2:
            return None, None

        # first_screen_bboxes = self.yolo.detect_objects(first_screen)
        first_screen_bboxes = [enlarge_bbox(bbox) for bbox in first_screen_bboxes]

        prompt_bboxes = list(sorted(prompt_bboxes, key=lambda x: x[0]))

        train_bboxes = self.yolo.detect_train(captcha_screens)

        p_bboxes_imgs = bboxes2imgs(prompt, prompt_bboxes)

        fs_bboxes_imgs = bboxes2imgs(first_screen, first_screen_bboxes)

        matching_bboxes = self.img_sim.matching(p_bboxes_imgs, fs_bboxes_imgs)

        answer_objects = [x[1] for x in matching_bboxes]
        answer_objects = list(sorted(answer_objects, key=lambda x: x[0]))
        answer_objects = [bbox_center(bbox) for bbox in answer_objects]

        promt_ans_x = max(answer_objects[0][0], answer_objects[1][0])
        promt_ans_y = max(answer_objects[0][1], answer_objects[1][1])

        train_xy = [bbox_center(bbox) for bbox in train_bboxes]
        train_xy = list(sorted(train_xy, key=lambda x: x[0]))
        distances = []
        for ind, center in enumerate(train_xy):
            promt_b = np.array([promt_ans_x + ind * 200, promt_ans_y])
            distances.append(euc_distance(promt_b, np.array(center)))

        ans_patch_number = np.argmin(distances) + 1
        ans_y = 100
        ans_x = (ans_patch_number - 1) * 200 + 100

        if visualization:
            v_img = self.visualization(prompt_bboxes, first_screen_bboxes, train_bboxes, matching_bboxes, img)

            v_img = cv2.circle(v_img, (ans_x, ans_y), 10, (0, 0, 255), 5)
            if label:
                v_img = cv2.circle(v_img, label, 10, (0, 255, 50), 10)

            cv2.imshow('img', v_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ans_x, ans_y

    def visualization(self, promt_bb, fs_bboxes, train_bboxes, matching_bboxes, img):
        v_img = img.copy()
        for bbox in promt_bb:
            bbox[1] += 200
            bbox[3] += 200
            v_img = draw_rectange(v_img, bbox, (255, 255, 0))

        for bbox in fs_bboxes:
            v_img = draw_rectange(v_img, bbox, (255, 255, 0))

        for bbox in train_bboxes:
            v_img = draw_rectange(v_img, bbox, (0, 0, 255))

        matching_colors = [(125, 125, 125), (125, 125, 0)]

        for ind, (p_bb, fs_bb) in enumerate(matching_bboxes):
            v_img = draw_rectange(v_img, p_bb, matching_colors[ind])
            v_img = draw_rectange(v_img, fs_bb, matching_colors[ind])

        return v_img
