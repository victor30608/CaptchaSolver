import os
import clip
import torch
from PIL import Image
import cv2

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


class ImgSim:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('weights/ViT-L-14.pt', self.device)

    def calc_features(self, img):
        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.detach().cpu().numpy()

    def matching(self, promt, candidates):
        promt_features = []
        for bbox, img in promt:
            promt_features.append((bbox, self.calc_features(img)))

        candidates_features = []
        for bbox, img in candidates:
            candidates_features.append((bbox, self.calc_features(img)))

        result = []

        for ind, (bbox, features) in enumerate(promt_features):
            if ind == 0:
                p_candidates_features = candidates_features[:3]
            else:
                p_candidates_features = candidates_features[3:]
            # best_bbox, conf = self.find_best(features, candidates_features)
            best_bbox, conf = self.find_best(features, p_candidates_features)
            result.append((bbox, best_bbox))

        return result

    def find_best(self, key, candidates):
        best_bbox, conf = None, 0
        for bbox, features in candidates:
            r = features @ key.T
            r = r[0][0]
            if r > conf:
                best_bbox = bbox
                conf = r
        return best_bbox, conf
