import cv2
from PIL import Image
import numpy as np
from utils import create_bbox, xywha2xyxy


class LabelHelper:

    def __init__(self, patch_count=7):
        self.patch_count = patch_count

    def __call__(self, img):
        h, w, _ = img.shape

        captcha_screens = img[:h // 2, :]
        prompt = img[h // 2:, :w // 8]

        first_sceen_w_promt = img.copy()[:, :w // self.patch_count]

        h, w, _ = captcha_screens.shape
        first_screen = captcha_screens.copy()[:, :w // self.patch_count]

        return first_screen, captcha_screens, prompt, first_sceen_w_promt

