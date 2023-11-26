from ultralytics import YOLO


class YoloDetector:

    def __init__(self, train_weights_path, objects_weights_path):
        self.train_model = YOLO(train_weights_path)
        self.object_model = YOLO(objects_weights_path)


    def detect_train(self, img):
        results = self.train_model.predict(img, conf=0.4, iou=0.3, imgsz=1280, verbose=False)

        boxes = results[0].boxes
        if boxes is not None:
            return self.__postprocessing(boxes.xyxy.cpu().detach().numpy())


    def detect_objects(self, img):
        results = self.object_model.predict(img, conf=0.4, iou=0.2, imgsz=640, verbose=False)

        boxes = results[0].boxes
        if boxes is not None:
            return self.__postprocessing(boxes.xyxy.cpu().detach().numpy())


    def __postprocessing(self, bboxes):
        return [list(map(int, b)) for b in bboxes]