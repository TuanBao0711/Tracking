import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal,QWaitCondition
from queue import Queue
import time


class CaptureThread(QThread):
    signalImg = pyqtSignal(np.ndarray)
    signalreplay = pyqtSignal(bool)
    
    def __init__(self, url_video): 
        super().__init__()
        self.url_video = url_video
        self.player = cv2.VideoCapture(self.url_video)
        self.threadActive = False
        self.cap_queue = Queue()

    def run(self):
        self.threadActive = True
        print("Start Capture")
        while self.threadActive:
            ret, frame = self.player.read()
            if not ret:
                self.player = cv2.VideoCapture(self.url_video)
                self.msleep(100)
                continue
            else:    
                if self.cap_queue.qsize() < 4:
                    self.cap_queue.put(frame)
                    # print(f"put--{self.url_video}")
                # print("come here")
            # print("Cap queue size: ", self.cap_queue.qsize())
            time.sleep(0.010)
        # self.player.release()

    def stop(self):
        self.threadActive = False


        

        
    