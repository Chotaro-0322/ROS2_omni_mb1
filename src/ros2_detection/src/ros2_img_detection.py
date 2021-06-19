#!/usr/bin/env python
import os
import sys
import time
import warnings
from collections import defaultdict

import cv2
import time

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import glob
import datetime

# from utils.evalDataset import evalDataset
from utils.mobile_model import create_mobilenetv1_ssd
from utils.evalTransform import EvalAugmentation, AfterAugmentation
# from layers.functions.detection import *
from utils.prior_box import *
from utils.predictor import Predictor
# from layers.modules.multibox_loss import MultiBoxLoss
warnings.filterwarnings("ignore")

from config import mb1_cfg, MEANS, SIZE

class Object_Detection(Node):
    def __init__(self):
        super().__init__("panorama_detection_imp")

        self.net = create_mobilenetv1_ssd(2, is_test=True)
        self.net.eval()
        self.net.to(torch.device("cuda:0"))

        net_weights = torch.load("/home/itolab-chotaro/All_ros_ws/ros2_mobile_ssd/src/ros2_detection/src/weight/mb1-ssd-complete3.pth",
                                map_location={'cuda:0': 'cpu'})

        self.net.load_state_dict(net_weights)

        #print("into Object detection!!!!")
        start = time.time()

        self.transform = EvalAugmentation()
        self.after_transform = AfterAugmentation()

        self.Predictor = Predictor(candidate_size=200)

        self._image_pub = self.create_publisher(Image, 'person_box', 1)
        #print("image_pablish !!!!")
        self._image_sub = self.create_subscription(Image, 'panoramaImage', self.publish_process, 1)
        #print("image_subscriber !!!")
        self._bridge = CvBridge()
        #print("CvBridge!!!!!")
        init_time = time.time() - start
        #print("def __init__ is ", init_time)

    def publish_process(self, data):
        cv_img = self.bridge = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        complete_img = self.detection_process(cv_img)
        self._image_pub.publish(self._bridge.cv2_to_imgmsg(complete_img, "bgr8"))
        pub_end = time.time()


    def detection(self, img, only_front=True):
        boxes_list = []
        score_list = []
        img_ori = img
        img, _, _ = self.transform(img, "", "")
        print("img is", img.size())
        img = img.to(torch.device("cuda:0"))
        scores, boxes = self.net(img)
        img = img.to(torch.device("cpu"))
        # img, _, _ = self.after_transform(img, "", "")
        # print("img is ", img.shape)
        # boxes, labels, score = self.Predictor.predict(score, boxes, 10, 0.4)
        print("boxes is", boxes.size())
        for box, score in zip(boxes, scores):
            # box = box.unsqueeze(0)
            # score = score.unsqueeze(0)
            box, labels, score = self.Predictor.predict(score, box, 100, 0.4)
            # img = img[:, :, [0, 1, 2]]
            boxes_list.append(box)
            score_list.append(score)


        return img_ori, score_list, boxes_list

    def get_coord(self, img, boxes_list, only_front):
        #print(prediction_box)
        #cv2.namedWindow("Image")
        #cv2.imshow("image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(np.shape(img))

        img = np.uint8(img)

        img = cv2.resize(img, (1200, 300)) #よくわからないけど, これがないとエラー
        img = cv2.line(img, (300, 0), (300, 300), (0, 255, 0), 2)
        img = cv2.line(img, (900, 0), (900, 300), (0, 255, 0), 2)
        height, width, _ = img.shape
        print(type(img))
        # print("img is ", img)
        # print("boxes is ", boxes_list)
        for i, boxes in enumerate(boxes_list):
            if (boxes is not None) and (i == 0):
                for box in boxes:
                    img = cv2.rectangle(img,
                                        (np.int(box[0] + i * width/4), np.int(box[1])),
                                        (np.int(box[2] + i * width/4), np.int(box[3])),
                                        (255, 0, 0), 5
                                        )

            elif (boxes is not None) and (i > 0):
                for box in boxes:
                    img = cv2.rectangle(img, (np.int(box[0] + i * width/4), np.int(box[1])), (np.int(box[2] + i * width/4), np.int(box[3])), (255, 0, 0), 5)
        return img



    def detection_process(self, img):
        only_front = True
        img, score, prediction_bbox = self.detection(img)
        # print("score is ", score.size())
        # print("predictbox is ", prediction_bbox.size())
        img = self.get_coord(img, prediction_bbox, only_front=only_front)

        return img
