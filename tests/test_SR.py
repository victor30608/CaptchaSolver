from legacy.super_resolution.load import load
import cv2
import torch
import torchvision.transforms as transforms

scale2x = load('super_resolution/weights/scalex2.ckpt', 2)

img = cv2.imread('examples/4498_jpg.rf.e0d11fdd57882734a021c68d53e6c1bd.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
tensorImg = transforms.ToTensor()(img)

with torch.no_grad():
    imgx2_tensor = scale2x(tensorImg[None])

imgx2 = imgx2_tensor.cpu().squeeze().numpy()
imgx2 = cv2.cvtColor(imgx2, cv2.COLOR_RGB2BGR)

cv2.imshow('imgx2', imgx2)
cv2.waitKey(0)
cv2.destroyAllWindows()