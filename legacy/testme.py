import cv2
import torch
from carvekit.api.high import HiInterface
from PIL import Image
import numpy as np
from utils import create_bbox, xywha2xyxy, draw_rectange




# Check doc strings for more information
interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=1280,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)


img = cv2.imread('../examples/2660_jpg.rf.42acb038d3d48da15d7cf18210bc9309.jpg')

h, w, _ = img.shape

part1 = img[:h//2,:]
part2 = img[h//2:,:w//8]

pil_img = cv2.cvtColor(part2, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(pil_img)

images_without_background = interface([pil_img])
cat_wo_bg = images_without_background[0]
cat_wo_bg = np.asarray(cat_wo_bg)

maxval = 255
thresh = 150
#
# im_bin = (cat_wo_bg > thresh) * maxval
# print(im_bin.shape)
cat_wo_bg = cat_wo_bg[:,:,0]
im_bin = (cat_wo_bg > thresh) * maxval
mask_binarized = im_bin.astype(np.uint8)

kernel = np.ones((7, 7), np.uint8)
mask_binarized = cv2.dilate(mask_binarized, kernel, iterations=1)

contours, _ = cv2.findContours(mask_binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bboxes = [xywha2xyxy(create_bbox(x)) for x in contours]
draw_img = part2.copy()
for ind, bbox in enumerate(bboxes):
    draw_img = draw_rectange(draw_img, bbox)
    img_bbox = part2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cv2.imwrite(str(ind)+'_temp.png', img_bbox)

h,w, _ = part1.shape
part1 = part1[:,0:w//7]
cv2.imwrite('../examples/part1.png', part1)


cv2.imshow('img', draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

