#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# from sensor_msgs import Image

from ros2_img_detection import Object_Detection

#class Listener(Node):
    #def __init__(self):
        #super().__init__("panorama_detection")
        #detection = Object_Detection()


def main(args=None):
    rclpy.init(args=args)
    try:
        listener = Object_Detection()
        rclpy.spin(listener)
    finally:
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
