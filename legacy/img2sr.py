import cv2
from legacy.super_resolution.SR import SR
import numpy as np

sr = SR(scale=4)
img = cv2.imread('1_temp.png')
imgx2 = (sr(img)*255).astype(np.uint8)
cv2.imshow('x2', imgx2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('label_sr.png', imgx2)