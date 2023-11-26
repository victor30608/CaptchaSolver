from legacy.super_resolution.load import load
import cv2
import torch
import torchvision.transforms as transforms


class SR:

    def __init__(self, scale):
        self.model = load(f'super_resolution/weights/scalex{scale}.ckpt', scale)

    def __call__(self, img):
        with torch.no_grad():
            img_SR = self.model(self.__preprocessing(img)[None])
        return self.__postprocessing(img_SR)

    def __preprocessing(self, img, BGR=True):
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return transforms.ToTensor()(img)

    def __postprocessing(self, img):
        img = img.cpu().squeeze().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
