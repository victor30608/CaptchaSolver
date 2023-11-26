from yolo import YoloDetector
import cv2
from utils import croping, draw_rectange

model = YoloDetector(objects_weights_path='weights/yolov8_objects/objects.pt',
                     train_weights_path='weights/yolov8_train/train.pt')


img = cv2.imread('35013.jpg')

predict_objects = model.detect_objects(img)
#predict_objects = model.detect_train(img)

for bbox in predict_objects:
    img = draw_rectange(img, bbox)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()