# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('/opt/ros/melodic/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('/opt/ros/melodic/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in '/home/itolab-chotaro/All_ros_ws/panorama_ws/devel;/home/itolab-chotaro/autoware.ai/install/ymc;/home/itolab-chotaro/autoware.ai/install/xsens_driver;/home/itolab-chotaro/autoware.ai/install/wf_simulator;/home/itolab-chotaro/autoware.ai/install/lattice_planner;/home/itolab-chotaro/autoware.ai/install/waypoint_planner;/home/itolab-chotaro/autoware.ai/install/waypoint_maker;/home/itolab-chotaro/autoware.ai/install/way_planner;/home/itolab-chotaro/autoware.ai/install/vlg22c_cam;/home/itolab-chotaro/autoware.ai/install/vision_ssd_detect;/home/itolab-chotaro/autoware.ai/install/vision_segment_enet_detect;/home/itolab-chotaro/autoware.ai/install/vision_lane_detect;/home/itolab-chotaro/autoware.ai/install/vision_darknet_detect;/home/itolab-chotaro/autoware.ai/install/vision_beyond_track;/home/itolab-chotaro/autoware.ai/install/vel_pose_diff_checker;/home/itolab-chotaro/autoware.ai/install/vehicle_socket;/home/itolab-chotaro/autoware.ai/install/vehicle_sim_model;/home/itolab-chotaro/autoware.ai/install/vehicle_model;/home/itolab-chotaro/autoware.ai/install/vehicle_gazebo_simulation_launcher;/home/itolab-chotaro/autoware.ai/install/vehicle_gazebo_simulation_interface;/home/itolab-chotaro/autoware.ai/install/vehicle_engage_panel;/home/itolab-chotaro/autoware.ai/install/vehicle_description;/home/itolab-chotaro/autoware.ai/install/trafficlight_recognizer;/home/itolab-chotaro/autoware.ai/install/op_utilities;/home/itolab-chotaro/autoware.ai/install/op_simulation_package;/home/itolab-chotaro/autoware.ai/install/op_local_planner;/home/itolab-chotaro/autoware.ai/install/op_global_planner;/home/itolab-chotaro/autoware.ai/install/lidar_kf_contour_track;/home/itolab-chotaro/autoware.ai/install/op_ros_helpers;/home/itolab-chotaro/autoware.ai/install/ff_waypoint_follower;/home/itolab-chotaro/autoware.ai/install/dp_planner;/home/itolab-chotaro/autoware.ai/install/op_simu;/home/itolab-chotaro/autoware.ai/install/op_planner;/home/itolab-chotaro/autoware.ai/install/op_utility;/home/itolab-chotaro/autoware.ai/install/lidar_euclidean_cluster_detect;/home/itolab-chotaro/autoware.ai/install/vector_map_server;/home/itolab-chotaro/autoware.ai/install/road_occupancy_processor;/home/itolab-chotaro/autoware.ai/install/costmap_generator;/home/itolab-chotaro/autoware.ai/install/object_map;/home/itolab-chotaro/autoware.ai/install/naive_motion_predict;/home/itolab-chotaro/autoware.ai/install/lanelet_aisan_converter;/home/itolab-chotaro/autoware.ai/install/map_file;/home/itolab-chotaro/autoware.ai/install/libvectormap;/home/itolab-chotaro/autoware.ai/install/lane_planner;/home/itolab-chotaro/autoware.ai/install/imm_ukf_pda_track;/home/itolab-chotaro/autoware.ai/install/decision_maker;/home/itolab-chotaro/autoware.ai/install/vector_map;/home/itolab-chotaro/autoware.ai/install/vector_map_msgs;/home/itolab-chotaro/autoware.ai/install/vectacam;/home/itolab-chotaro/autoware.ai/install/udon_socket;/home/itolab-chotaro/autoware.ai/install/twist_generator;/home/itolab-chotaro/autoware.ai/install/twist_gate;/home/itolab-chotaro/autoware.ai/install/twist_filter;/home/itolab-chotaro/autoware.ai/install/twist2odom;/home/itolab-chotaro/autoware.ai/install/tablet_socket;/home/itolab-chotaro/autoware.ai/install/runtime_manager;/home/itolab-chotaro/autoware.ai/install/mqtt_socket;/home/itolab-chotaro/autoware.ai/install/tablet_socket_msgs;/home/itolab-chotaro/autoware.ai/install/state_machine_lib;/home/itolab-chotaro/autoware.ai/install/sound_player;/home/itolab-chotaro/autoware.ai/install/sick_lms5xx;/home/itolab-chotaro/autoware.ai/install/sick_ldmrs_tools;/home/itolab-chotaro/autoware.ai/install/sick_ldmrs_driver;/home/itolab-chotaro/autoware.ai/install/sick_ldmrs_msgs;/home/itolab-chotaro/autoware.ai/install/sick_ldmrs_description;/home/itolab-chotaro/autoware.ai/install/points2image;/home/itolab-chotaro/autoware.ai/install/rosinterface;/home/itolab-chotaro/autoware.ai/install/rosbag_controller;/home/itolab-chotaro/autoware.ai/install/pure_pursuit;/home/itolab-chotaro/autoware.ai/install/points_preprocessor;/home/itolab-chotaro/autoware.ai/install/mpc_follower;/home/itolab-chotaro/autoware.ai/install/lidar_localizer;/home/itolab-chotaro/autoware.ai/install/emergency_handler;/home/itolab-chotaro/autoware.ai/install/autoware_health_checker;/home/itolab-chotaro/autoware.ai/install/as;/home/itolab-chotaro/autoware.ai/install/ros_observer;/home/itolab-chotaro/autoware.ai/install/roi_object_filter;/home/itolab-chotaro/autoware.ai/install/range_vision_fusion;/home/itolab-chotaro/autoware.ai/install/pos_db;/home/itolab-chotaro/autoware.ai/install/points_downsampler;/home/itolab-chotaro/autoware.ai/install/pixel_cloud_fusion;/home/itolab-chotaro/autoware.ai/install/pcl_omp_registration;/home/itolab-chotaro/autoware.ai/install/pc2_downsampler;/home/itolab-chotaro/autoware.ai/install/oculus_socket;/home/itolab-chotaro/autoware.ai/install/obj_db;/home/itolab-chotaro/autoware.ai/install/nmea_navsat;/home/itolab-chotaro/autoware.ai/install/ndt_tku;/home/itolab-chotaro/autoware.ai/install/ndt_cpu;/home/itolab-chotaro/autoware.ai/install/multi_lidar_calibrator;/home/itolab-chotaro/autoware.ai/install/mrt_cmake_modules;/home/itolab-chotaro/autoware.ai/install/microstrain_driver;/home/itolab-chotaro/autoware.ai/install/memsic_imu;/home/itolab-chotaro/autoware.ai/install/marker_downsampler;/home/itolab-chotaro/autoware.ai/install/map_tools;/home/itolab-chotaro/autoware.ai/install/map_tf_generator;/home/itolab-chotaro/autoware.ai/install/log_tools;/home/itolab-chotaro/autoware.ai/install/lidar_shape_estimation;/home/itolab-chotaro/autoware.ai/install/lidar_point_pillars;/home/itolab-chotaro/autoware.ai/install/lidar_naive_l_shape_detect;/home/itolab-chotaro/autoware.ai/install/lidar_fake_perception;/home/itolab-chotaro/autoware.ai/install/lidar_apollo_cnn_seg_detect;/home/itolab-chotaro/autoware.ai/install/libwaypoint_follower;/home/itolab-chotaro/autoware.ai/install/lgsvl_simulator_bridge;/home/itolab-chotaro/autoware.ai/install/lanelet2_extension;/home/itolab-chotaro/autoware.ai/install/lanelet2_validation;/home/itolab-chotaro/autoware.ai/install/lanelet2_examples;/home/itolab-chotaro/autoware.ai/install/lanelet2_python;/home/itolab-chotaro/autoware.ai/install/lanelet2_routing;/home/itolab-chotaro/autoware.ai/install/lanelet2_traffic_rules;/home/itolab-chotaro/autoware.ai/install/lanelet2_projection;/home/itolab-chotaro/autoware.ai/install/lanelet2_maps;/home/itolab-chotaro/autoware.ai/install/lanelet2_io;/home/itolab-chotaro/autoware.ai/install/lanelet2_core;/home/itolab-chotaro/autoware.ai/install/kvaser;/home/itolab-chotaro/autoware.ai/install/kitti_launch;/home/itolab-chotaro/autoware.ai/install/kitti_player;/home/itolab-chotaro/autoware.ai/install/kitti_box_publisher;/home/itolab-chotaro/autoware.ai/install/javad_navsat_driver;/home/itolab-chotaro/autoware.ai/install/integrated_viewer;/home/itolab-chotaro/autoware.ai/install/image_processor;/home/itolab-chotaro/autoware.ai/install/hokuyo;/home/itolab-chotaro/autoware.ai/install/graph_tools;/home/itolab-chotaro/autoware.ai/install/gnss_localizer;/home/itolab-chotaro/autoware.ai/install/gnss;/home/itolab-chotaro/autoware.ai/install/glviewer;/home/itolab-chotaro/autoware.ai/install/gazebo_world_description;/home/itolab-chotaro/autoware.ai/install/gazebo_imu_description;/home/itolab-chotaro/autoware.ai/install/gazebo_camera_description;/home/itolab-chotaro/autoware.ai/install/garmin;/home/itolab-chotaro/autoware.ai/install/freespace_planner;/home/itolab-chotaro/autoware.ai/install/fastvirtualscan;/home/itolab-chotaro/autoware.ai/install/ekf_localizer;/home/itolab-chotaro/autoware.ai/install/ds4_msgs;/home/itolab-chotaro/autoware.ai/install/ds4_driver;/home/itolab-chotaro/autoware.ai/install/detected_objects_visualizer;/home/itolab-chotaro/autoware.ai/install/decision_maker_panel;/home/itolab-chotaro/autoware.ai/install/data_preprocessor;/home/itolab-chotaro/autoware.ai/install/custom_msgs;/home/itolab-chotaro/autoware.ai/install/carla_autoware_bridge;/home/itolab-chotaro/autoware.ai/install/calibration_publisher;/home/itolab-chotaro/autoware.ai/install/autoware_system_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_rviz_plugins;/home/itolab-chotaro/autoware.ai/install/autoware_quickstart_examples;/home/itolab-chotaro/autoware.ai/install/autoware_pointgrey_drivers;/home/itolab-chotaro/autoware.ai/install/autoware_driveworks_interface;/home/itolab-chotaro/autoware.ai/install/autoware_connector;/home/itolab-chotaro/autoware.ai/install/autoware_camera_lidar_calibrator;/home/itolab-chotaro/autoware.ai/install/astar_search;/home/itolab-chotaro/autoware.ai/install/amathutils_lib;/home/itolab-chotaro/autoware.ai/install/autoware_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_map_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_launcher_rviz;/home/itolab-chotaro/autoware.ai/install/autoware_launcher;/home/itolab-chotaro/autoware.ai/install/autoware_lanelet2_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_external_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_driveworks_gmsl_interface;/home/itolab-chotaro/autoware.ai/install/autoware_config_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_can_msgs;/home/itolab-chotaro/autoware.ai/install/autoware_build_flags;/home/itolab-chotaro/autoware.ai/install/autoware_bag_tools;/home/itolab-chotaro/autoware.ai/install/adi_driver;/opt/ros/melodic'.split(';'):
        python_path = os.path.join(workspace, 'lib/python2.7/dist-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('/home/itolab-chotaro/All_ros_ws/ros2_detect_ws/build/Project/devel/env.sh')

output_filename = '/home/itolab-chotaro/All_ros_ws/ros2_detect_ws/build/Project/catkin_generated/setup_cached.sh'
with open(output_filename, 'w') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)