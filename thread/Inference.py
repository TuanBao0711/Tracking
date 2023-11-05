import cv2
import numpy as np
import time
from  queue import Queue
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import torch
from PIL import Image
import boxmot 
from pathlib import Path
from ultralytics import YOLO

import sys, os
import cvzone



class Inference(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, queue,trackingAl):
        super().__init__()
        self.detection_threshold = 0.3
        
        self.color = (0, 0, 255)  # BGR
        self.thickness = 2
        self.fontscale = 0.5
        self.threadActive = False
        self.queue_cap = queue
        self.tracker = None
        self.Model = YOLO('thread/model/UAV_v82.pt')

        if trackingAl == 0:
            self.tracker = boxmot.StrongSORT(
                model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
                device='cuda:0',
                fp16=True,
            )
            print('StrongSort')
        elif trackingAl == 1:
            self.tracker = boxmot.BYTETracker()
            print('ByteTrack')
        elif trackingAl == 2:
            self.tracker = boxmot.BoTSORT(
                model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
                device='cuda:0',
                fp16=True,
            )
            print('BoT-Sort')
        elif trackingAl == 3:
            self.tracker = boxmot.OCSORT()
            print('OC-Sort')
        elif trackingAl == 4:
            self.tracker = boxmot.DeepOCSORT(
                model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
                device='cuda:0',
                fp16=True,
            )
            print('Deep-OC-Sort')

        self.threadActive = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("CUDA device name:", torch.cuda.get_device_name(self.device))


        self.fpsReader = cvzone.FPS()



    def run(self):
        print("thread inference start")
        while self.threadActive:
            
            if self.queue_cap.qsize() > 0:
                

                im = self.queue_cap.get()   
                results = self.Model(im)
                for result in results:
                    boxes = result.boxes.xyxy
                    confs = result.boxes.conf
                    cls = result.boxes.cls
                    # convert PyTorch to NumPy
                    boxes_np = boxes.cpu().numpy()
                    confs_np = confs.cpu().numpy()
                    cls_np = cls.cpu().numpy()
                    detection_results = np.column_stack((boxes_np, confs_np, cls_np))
                tracks = self.tracker.update(detection_results, im) # --> (x, y, x, y, id, conf, cls, ind)
                if np.size(tracks) > 0:
                    xyxys = tracks[:, 0:4].astype('int') # float64 to int
                    ids = tracks[:, 4].astype('int') # float64 to int
                    confs = tracks[:, 5]
                    clss = tracks[:, 6].astype('int') # float64 to int
                    inds = tracks[:, 7].astype('int') # float64 to int
                    if tracks.shape[0] != 0:
                        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                            im = cv2.rectangle(
                                im,
                                (xyxy[0], xyxy[1]),
                                (xyxy[2], xyxy[3]),
                                self.color,
                                self.thickness
                            )
                            cv2.putText(
                                im,
                                f'id: {id}, conf: {conf}, c: {cls}',
                                (xyxy[0], xyxy[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.fontscale,
                                self.color,
                                self.thickness
                            )
                fps, im = self.fpsReader.update(im, pos=(100, 100), color=(255, 0, 0), scale=3, thickness=3)
                self.signal.emit(im)
                time.sleep(0.001)
            time.sleep(0.001)
    


    
