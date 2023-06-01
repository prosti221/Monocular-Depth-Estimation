import torch
import matplotlib.pyplot as plt
import cv2

class GeneralDetector:
    def __init__(self, device='cpu', confidence_threshold=0.5, classes=[0]):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 
        self.model.conf = confidence_threshold # confidence threshold
        self.model.classes = classes # Only detect persons
        #self.model.iou = 0.45                  # NMS IoU threshold
        #self.model.max_det = 1000              # maximum number of detections per image

        self.current_prediction = None

    def detect(self, inp):
        '''
        Return detections in pandas dataframe containing:
        [xmin, ymin, xmax, ymax, confidence, class, name]
        Return the detections with confidence score higher than threshold
        '''
        pred = self.model(inp)
        self.current_prediction = pred
        return pred.pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)



    def show_detections(self):
        if self.current_prediction is not None:
            self.current_prediction.show()

# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
if __name__ == '__main__':
    detector = VehicleDetector('./yolov5.pt', device='cpu')

    # Run inference
    img_path = './test_images/img5.png'
    detections = detector.detect(img_path)

    print(detections)

    
    


