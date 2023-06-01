import yolov5
import torch
import cv2
import pandas as pd

class PlateDetector:
    def __init__(self, weights='keremberke/yolov5m-license-plate', device='cpu', confidence_threshold=0.2):
        self.device = device
        self.model = yolov5.load(weights, device)
        # set model parameters
        self.model.conf = confidence_threshold
        #self.model.iou = 0.45  # NMS IoU threshold
        #self.model.agnostic = False  # NMS class-agnostic
        #self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1  # maximum number of detections per image

    def detect(self, img):
        results = self.model(img, size=640)
        return results.pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)
