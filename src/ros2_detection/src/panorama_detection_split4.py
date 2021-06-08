#!/usr/bin/env python3
import os
import sys

print(sys.path)
import rospy
import time

# -*- coding: utf-8 -*-


print(sys.version)
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from img_detection import Object_Detection

#os.chdir("/home/itolab-chotaro/imgpro_ws/src/object_detection/scripts")
if __name__ == '__main__':
    rospy.init_node('panorama_detection')
    detection = Object_Detection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


