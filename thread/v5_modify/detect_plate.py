import os
import torch
import numpy as np
from thread.yolov5.models.common import DetectMultiBackend
from thread.yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from thread.yolov5.utils.augmentations import letterbox
# from ..config import ROOT
import time
# import easyocr

##### DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en','vi']) ### initiating easyocr
OCR_TH = 0.2

class Detection_anpp:
    def __init__(self):
        
        self.imgsz = (320, 320)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cuda'
        self.classes = 0
        # self.stride = None
   
        self.agnostic_nms = True
        self.half = False
        self.dnn = False  # use OpenCV DNN for ONNX inference
  

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
    def detect(self, image):
        bboxes = []
        im = letterbox(image, self.imgsz, stride=self.stride,
                       auto=self.pt)[0]  # resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        for i, det in enumerate(pred):
            if len(det):
                det = det.detach().cpu().numpy()
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), xyxy))
                    bboxes.append([x1, y1, x2, y2, int(cls), float(conf)])
        return bboxes
    
    # function to recognize license plate numbers using Tesseract OCR
    def recognize_plate_easyocr(self, img, coords,reader,region_threshold):
        # separate coordinates from box
        xmin, ymin, xmax, ymax = coords
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
        ocr_result = reader.readtext(nplate)
        text = self.filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
        if len(text) ==1:
            text = text[0].upper()
        return text

    ### to filter out wrong detections 
    def filter_text(self,region, ocr_result, region_threshold):
        rectangle_size = region.shape[0]*region.shape[1]
        plate = [] 
        print(ocr_result)
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        return plate
