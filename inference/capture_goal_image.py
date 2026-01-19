#!/usr/bin/env python3
"""
Capture Goal Image from Drone
=============================

Simple script to capture the current camera image and save it as the goal image.
Run this whenever you want to set a new navigation goal.

Usage (inside Docker container):
    python capture_goal_image.py

    # Or with custom filename:
    python capture_goal_image.py --name my_goal.jpg

    # Or with custom topic:
    python capture_goal_image.py --topic /simple_drone/bottom/image_raw
"""

import os
import sys
import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2

# Default paths - these match run.sh setup
OMNIVLA_ROOT = os.environ.get('OMNIVLA_ROOT', '/root/ros2_ws/OmniVLA')
DEFAULT_SAVE_DIR = os.path.join(OMNIVLA_ROOT, 'inference')
DEFAULT_FILENAME = 'goal_img.jpg'
DEFAULT_TOPIC = '/simple_drone/front/image_raw'


class ImageCapture(Node):
    """Simple node to capture a single image"""

    def __init__(self, topic: str, save_path: str):
        super().__init__('image_capture')
        self.bridge = CvBridge()
        self.save_path = save_path
        self.captured = False

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.get_logger().info(f"Waiting for image on: {topic}")
        self.sub = self.create_subscription(
            RosImage,
            topic,
            self.image_callback,
            sensor_qos
        )

    def image_callback(self, msg: RosImage):
        """Capture and save the image"""
        if self.captured:
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            # Save image
            cv2.imwrite(self.save_path, cv_image)

            self.captured = True
            self.get_logger().info(f"✓ Image saved to: {self.save_path}")
            self.get_logger().info(f"  Resolution: {cv_image.shape[1]}x{cv_image.shape[0]}")

        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")


def main():
    parser = argparse.ArgumentParser(description='Capture goal image from drone camera')
    parser.add_argument('--topic', '-t', default=DEFAULT_TOPIC,
                        help=f'Image topic (default: {DEFAULT_TOPIC})')
    parser.add_argument('--name', '-n', default=DEFAULT_FILENAME,
                        help=f'Output filename (default: {DEFAULT_FILENAME})')
    parser.add_argument('--dir', '-d', default=DEFAULT_SAVE_DIR,
                        help=f'Output directory (default: {DEFAULT_SAVE_DIR})')
    parser.add_argument('--timeout', default=10.0, type=float,
                        help='Timeout in seconds (default: 10)')
    args = parser.parse_args()

    save_path = os.path.join(args.dir, args.name)

    print("=" * 50)
    print("Goal Image Capture")
    print("=" * 50)
    print(f"Topic:  {args.topic}")
    print(f"Output: {save_path}")
    print("=" * 50)

    # Initialize ROS2
    rclpy.init()

    node = ImageCapture(args.topic, save_path)

    # Spin until image is captured or timeout
    start_time = node.get_clock().now()
    timeout = rclpy.duration.Duration(seconds=args.timeout)

    try:
        while rclpy.ok() and not node.captured:
            rclpy.spin_once(node, timeout_sec=0.1)

            elapsed = node.get_clock().now() - start_time
            if elapsed > timeout:
                node.get_logger().error(f"Timeout! No image received in {args.timeout}s")
                node.get_logger().error(f"Check if topic '{args.topic}' is publishing:")
                node.get_logger().error(f"  ros2 topic hz {args.topic}")
                break

    except KeyboardInterrupt:
        print("\nCancelled")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.captured:
        print("\n✓ Goal image captured successfully!")
        print(f"  File: {save_path}")
        print("\nThe image is saved to the mounted volume,")
        print("so it's available both inside and outside the container.")
        return 0
    else:
        print("\n✗ Failed to capture image")
        return 1


if __name__ == "__main__":
    sys.exit(main())