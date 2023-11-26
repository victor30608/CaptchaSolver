import cv2
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
import json

def xywha2polygon(x, y, width, height, angle):
    w = width / 2
    h = height / 2
    p = Polygon([(-w, -h), (w, -h), (w, h), (-w, h)])
    return translate(rotate(p, angle), x, y)


def xywha2xyxy(bbox):
    poly_bbox = xywha2polygon(*bbox)

    poly_bbox = list(poly_bbox.exterior.coords)

    return poly2xyxy(poly_bbox)


def draw_polylines(img, box, color):
    box = np.array(box).astype(np.int32).reshape(-1, 2)
    return cv2.polylines(img, [box], True, color=color, thickness=3)


def create_bbox(countur):
    (x, y), (w, h), angle = cv2.minAreaRect(countur)
    if w < 3 or h < 3:
        return
    if angle < -45.0:
        # the `cv.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        angle += 90
        w, h = h, w

    angle = angle if angle else .0
    return [x, y, w, h, angle]


def poly2xyxy(bbox):
    bbox = np.array(bbox)
    left = int(np.min(bbox[:, 0]))
    right = int(np.max(bbox[:, 0]))
    top = int(np.min(bbox[:, 1]))
    bottom = int(np.max(bbox[:, 1]))
    return np.array([left, top, right, bottom])


def draw_rectange(image, bbox, color=(255, 255, 0)):
    startpoint = (bbox[0], bbox[1])
    endpoint = (bbox[2], bbox[3])
    image = cv2.rectangle(image, startpoint, endpoint, color, 3)
    return image


def croping(img, bbox):
    bbox = list(map(int, bbox))
    return img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def bboxes2imgs(img, bboxes):
    result = []
    for bbox in bboxes:
        result.append((bbox, croping(img, bbox)))
    return result


def bbox_center(bbox):
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

def draw_circle(img, center, r=5):
    return cv2.circle(img, center, r, (255, 0, 125), 3)

def enlarge_bbox(bbox, perc=0.05):
    x1, y1, x2, y2 = bbox
    w = int((x2 - x1)*perc)
    x1 = max(x1 - w//2, 0)
    x2 = x2 + w//2
    h = int((y2 - y1) * perc)
    y1 = max(y1 - h // 2, 0)
    y2 = y2 + h // 2
    return [x1, y1, x2, y2]

def euc_distance(a, b):
    return np.linalg.norm(a - b)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_answer_xy(markup):
    for label in markup['shapes']:
        if label['label'] == '+':
            return label['points']