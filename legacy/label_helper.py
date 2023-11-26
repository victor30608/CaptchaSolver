import cv2
import torch
from carvekit.api.high import HiInterface
from PIL import Image
import numpy as np
from utils import create_bbox, xywha2xyxy, draw_rectange


class LabelHelper:

    def __init__(self, patch_count=7):
        self.segmentator = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                                       batch_size_seg=5,
                                       batch_size_matting=1,
                                       device='cuda' if torch.cuda.is_available() else 'cpu',
                                       seg_mask_size=1280,  # Use 640 for Tracer B7 and 320 for U2Net
                                       matting_mask_size=2048,
                                       trimap_prob_threshold=231,
                                       trimap_dilation=30,
                                       trimap_erosion_iters=5,
                                       fp16=False)

        self.patch_count = patch_count
        self.maxval = 255
        self.thresh = 150  # threshold for binarization

    def __call__(self, img, use_segmentator=False):
        h, w, _ = img.shape

        captcha_screens = img[:h // 2, :]
        prompt = img[h // 2:, :w // 8]

        first_sceen_w_promt = img.copy()[:, :w // self.patch_count]
        bboxes = None

        if use_segmentator:

            promt_wo_bg = self.__img2imgwobg(prompt.copy())
            mask_binarized = self.__binarization(promt_wo_bg)

            bboxes = self.__mask2bboxes(mask_binarized, prompt.copy())

        h, w, _ = captcha_screens.shape
        first_screen = captcha_screens.copy()[:, :w // self.patch_count]

        return bboxes, first_screen, captcha_screens, prompt, first_sceen_w_promt

    def __img2imgwobg(self, img):
        pil_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img)

        images_without_background = self.segmentator([pil_img])
        cat_wo_bg = images_without_background[0]
        cat_wo_bg = np.asarray(cat_wo_bg)
        return cat_wo_bg

    def __binarization(self, cat_wo_bg, dilatation=True):
        cat_wo_bg = cat_wo_bg[:, :, 0]
        im_bin = (cat_wo_bg > self.thresh) * self.maxval
        mask_binarized = im_bin.astype(np.uint8)
        if dilatation:
            kernel = np.ones((7, 7), np.uint8)
            mask_binarized = cv2.dilate(mask_binarized, kernel, iterations=1)
        return mask_binarized

    def __mask2bboxes(self, mask_binarized, img_part):
        contours, _ = cv2.findContours(mask_binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [xywha2xyxy(create_bbox(x)) for x in contours]
        return bboxes
