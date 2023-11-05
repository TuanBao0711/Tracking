import os
import torch
import torch.nn.functional as F
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox, classify_transforms
from yolov5.utils.dataloaders import LoadImages


class Predict_C:
    def __init__(self) :
        self.imgsz = (320, 320)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cuda'
        self.half = False
        self.dnn = False
        self.transforms = classify_transforms(self.imgsz[0])
        
    def _load_model(self):
        # Load model
        # self.device = select_device(self.device)
        if self.device == "cpu":
            arg = "cpu"
        else:
            arg = f"{self.device}"
        self.device = torch.device(arg)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, fp16=self.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size
        
    @torch.no_grad()
    def predict(self, image):
        im = self.transforms(image)  # transforms
        im = torch.Tensor(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        results = self.model(im)
        pred = F.softmax(results, dim=1)  # probabilities
        top5i = pred.argsort(1, descending=True)[:, :5].tolist()  # top 5 indices
        top5_classes = []
        for i, top_indices in enumerate(top5i):
            top_classes = '-'.join([f'{self.names[idx]}: {pred[i][idx].item()}' for idx in top_indices])
            top5_classes.append(top_classes)
        # print(top5_classes)
        top_class_index = top5i[0][0]
        name_classify = self.names[top_class_index]
        return name_classify
        

