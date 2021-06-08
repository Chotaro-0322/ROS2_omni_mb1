#!/usr/bin/env python
import os
import sys
import time
import warnings
from collections import defaultdict

import cv2
import numpy as np
# import rospy
import torch
import time
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
#print(sys.path)
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

from utils.ssd_model import SSD
from utils.ssd_predict_show_split8 import SSDPredictShow_split8

warnings.filterwarnings("ignore")

class Object_Detection(Node):
    def __init__(self):
        super().__init__("panorama_detection_imp")

        self.voc_classes = ['person', 'difficult_person']
        ssd_cfg = {
            'num_classes': 3,
            'input_size': 300,
            'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2],[2, 3],[2, 3],[2, 3],[2],[2]],
        }
        self.net = SSD(phase="inference", cfg=ssd_cfg)


        net_weights = torch.load("/home/itolab-chotaro/ros2_detect_ws/src/ros2_detection/src/weight/201111_ssd300_VOTT_split4_201031_1_100000.pth", 
                                map_location={'cuda:0': 'cpu'})

        self.net.load_state_dict(net_weights)

        #print("into Object detection!!!!")
        start = time.time()
        self._image_pub = self.create_publisher(Image, 'person_box', 1)
        #print("image_pablish !!!!")
        self._image_sub = self.create_subscription(Image, 'panoramaImage', self.publish_process, 1)
        #print("image_subscriber !!!")
        self._bridge = CvBridge()
        #print("CvBridge!!!!!")
        init_time = time.time() - start
        #print("def __init__ is ", init_time)

    def publish_process(self, data):
        #print(data)
        #print("into pablich_process !!!")
        #pub_start = time.time()
        cv_img = self.bridge = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        #cv_time = time.time()
        complete_img = self.detection_process(cv_img)
        #detection_time = time.time()
        # print("complete_img is ", complete_img)
        self._image_pub.publish(self._bridge.cv2_to_imgmsg(complete_img, "bgr8"))
        pub_end = time.time()
        #print("cv_time is ", cv_time - pub_start)
        #print("detection_time is ", detection_time - cv_time)
        #print("pub_end is ", pub_end - detection_time)
        #print("all of time is ", pub_end - pub_start, "\n\n\n")


    def detection(self, img, ssd, only_front=True):
        #print(np.shape(img.shape))
        img, predict_bbox, predict_label_index, score = ssd.ssd_panorama_predict(img, only_front=True , data_confidence_level=0.5)
        #print(np.shape(predict_bbox))
        img = img[:, :, [0, 1, 2]]


        return img, predict_bbox

    def get_coord(self, img, prediction_box, only_front):
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
        #print(img)
        if only_front == True:
            for i, boxes in enumerate(prediction_box):
                if boxes and i == 0:
                    for box in boxes:
                        img = cv2.rectangle(img, (np.int(box[0] + 1 * width/4), np.int(box[1])), (np.int(box[2] + 1 * width/4), np.int(box[3])), (255, 0, 0), 5)
                elif boxes and i == 1:
                    for box in boxes:
                        img = cv2.rectangle(img, (np.int(box[0] + 2 * width/4), np.int(box[1])), (np.int(box[2] + 2 * width/4), np.int(box[3])), (255, 0, 0), 5)
        else:
            print ("only_front is False")

        return img



    def detection_process(self, img):
        ssd = SSDPredictShow_split8(eval_categories=self.voc_classes, net=self.net)
        only_front = True
        img, prediction_bbox = self.detection(img, ssd, only_front=only_front)
        print(prediction_bbox)
        img = self.get_coord(img, prediction_bbox, only_front=only_front)

        return img
