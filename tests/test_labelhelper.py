from label_helper import LabelHelper
import cv2

lh = LabelHelper()
img = cv2.imread('examples/2626_jpg.rf.475971cecf35a6d74a80433c8b0a609b.jpg')

a, b = lh(img)