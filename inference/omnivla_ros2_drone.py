#!/usr/bin/env python3
"""
OmniVLA ROS2 Adapter for SJTU Drone - FIXED VERSION
====================================================

Fixed QoS settings to match actual drone topics:
- Image: RELIABLE
- Odom: RELIABLE
- Cmd_vel: BEST_EFFORT (to match drone subscriber)

Usage:
    cd $OMNIVLA_ROOT
    python inference/omnivla_ros2_drone.py
"""

import sys
import os

# Get paths from environment variables (set by run.sh)
OMNIVLA_ROOT = os.environ.get('OMNIVLA_ROOT', '/root/ros2_ws/OmniVLA')
OMNIVLA_CHECKPOINTS = os.environ.get('OMNIVLA_CHECKPOINTS', '/models/omnivla')

# Add OmniVLA to path
sys.path.insert(0, OMNIVLA_ROOT)

import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as RosImage, Imu
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# OmniVLA imports
try:
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.action_heads import L1RegressionActionHead_idcat
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
    from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

    print(f"[OK] OmniVLA modules loaded from: {OMNIVLA_ROOT}")
except ImportError as e:
    print(f"[ERROR] Failed to import OmniVLA modules: {e}")
    print(f"[ERROR] OMNIVLA_ROOT={OMNIVLA_ROOT}")
    print(f"[ERROR] Try: cd $OMNIVLA_ROOT && pip install -e .")
    sys.exit(1)


# ===============================================================
# CONFIGURATION
# ===============================================================
@dataclass
class DroneConfig:
    """Configuration for the drone OmniVLA adapter"""

    # Model paths
    checkpoint_name: str = "omnivla-original"
    resume_step: int = 120000

    # Goal modality (set ONE to True)
    use_pose_goal: bool = False
    use_satellite: bool = False
    use_image_goal: bool = False
    use_language_prompt: bool = True

    # Goal settings
    # ----- Goal Settings -----
    # Example short hospital-navigation prompts (swap into `language_instruction`):
    # 1) "fly down the hallway"
    # 2) "go straight to the nurses station"
    # 3) "navigate to the reception desk"
    # 4) "turn left at the next intersection"
    # 5) "turn right at the next intersection"
    # 6) "follow the corridor to the elevator"
    # 7) "go to the nearest wheelchair"
    # 8) "navigate to the waiting area"
    # 9) "fly to the nearest open doorway"
    # 10) "move toward the end of the hall"
    # 11) "enter the room on the left"
    # 12) "enter the room on the right"
    # 13) "go to the main lobby"
    # 14) "find the nearest hospital bed"
    # 15) "find the waiting area"

    language_instruction: str = "enter the room on the right"
    goal_image_path: str = "inference/goal_img.jpg"

    # Goal position (if using pose goal)
    goal_x: float = 10.0
    goal_y: float = 0.0
    goal_yaw: float = 0.0

    # Control parameters
    tick_rate: float = 3.0
    waypoint_select: int = 4
    metric_waypoint_spacing: float = 0.1
    max_goal_distance: float = 30.0

    # Velocity limits
    max_linear_vel: float = 0.5
    max_angular_vel: float = 0.5

    # ROS2 topics
    image_topic: str = "/simple_drone/front/image_raw"
    odom_topic: str = "/simple_drone/odom"
    imu_topic: str = "/simple_drone/imu/out"
    cmd_vel_topic: str = "/simple_drone/cmd_vel"

    # Visualization
    save_visualizations: bool = True
    viz_output_dir: str = "inference/ros2_output"

    @property
    def vla_path(self) -> str:
        return os.path.join(OMNIVLA_CHECKPOINTS, self.checkpoint_name)

    @property
    def full_goal_image_path(self) -> str:
        if os.path.isabs(self.goal_image_path):
            return self.goal_image_path
        return os.path.join(OMNIVLA_ROOT, self.goal_image_path)

    @property
    def full_viz_output_dir(self) -> str:
        if os.path.isabs(self.viz_output_dir):
            return self.viz_output_dir
        return os.path.join(OMNIVLA_ROOT, self.viz_output_dir)


CONFIG = DroneConfig()


# ===============================================================
# Utility Functions
# ===============================================================
def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(np.clip(sinp, -1, 1))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def remove_ddp_in_checkpoint(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(module_name, path, step, device="cpu"):
    checkpoint_file = f"{module_name}--{step}_checkpoint.pt"
    checkpoint_path = os.path.join(path, checkpoint_file)

    if not os.path.exists(checkpoint_path) and module_name == "pose_projector":
        checkpoint_file = f"proprio_projector--{step}_checkpoint.pt"
        checkpoint_path = os.path.join(path, checkpoint_file)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[OK] Loading: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


# ===============================================================
# OmniVLA Drone Node
# ===============================================================
class OmniVLADroneNode(Node):
    """ROS2 Node for OmniVLA-based drone navigation"""

    def __init__(self, config: DroneConfig):
        super().__init__('omnivla_drone_node')
        self.config = config
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # State variables
        self.current_image: Optional[Image.Image] = None
        self.current_pose: Optional[Tuple[float, float, float, float]] = None
        self.goal_image: Optional[Image.Image] = None
        self.inference_count = 0

        # Track data reception
        self.last_image_time = None
        self.last_odom_time = None
        self.last_imu_time = None
        self.image_count = 0
        self.odom_count = 0
        self.imu_count = 0

        # Model components
        self.vla = None
        self.action_head = None
        self.pose_projector = None
        self.processor = None
        self.action_tokenizer = None
        self.num_patches = None
        self.device = None

        # Load goal image
        self._load_goal_image()

        # Create viz directory
        if config.save_visualizations:
            os.makedirs(config.full_viz_output_dir, exist_ok=True)

        # ============================================================
        # QoS PROFILES - FIXED TO MATCH ACTUAL DRONE TOPICS
        # ============================================================

        # For subscribing to image (publisher uses RELIABLE)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # For subscribing to odom (publisher uses RELIABLE)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # For subscribing to IMU (publisher uses RELIABLE)
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # For publishing cmd_vel (subscriber expects BEST_EFFORT)
        cmd_vel_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Create subscribers
        self.image_sub = self.create_subscription(
            RosImage,
            config.image_topic,
            self._image_callback,
            image_qos
        )
        self.get_logger().info(f"Subscribed to image: {config.image_topic} (RELIABLE)")

        self.odom_sub = self.create_subscription(
            Odometry,
            config.odom_topic,
            self._odom_callback,
            odom_qos
        )
        self.get_logger().info(f"Subscribed to odom: {config.odom_topic} (RELIABLE)")

        self.imu_sub = self.create_subscription(
            Imu,
            config.imu_topic,
            self._imu_callback,
            imu_qos
        )
        self.get_logger().info(f"Subscribed to IMU: {config.imu_topic} (RELIABLE)")

        # Create publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            config.cmd_vel_topic,
            cmd_vel_qos
        )
        self.get_logger().info(f"Publishing to: {config.cmd_vel_topic} (BEST_EFFORT)")

        # Create inference timer
        self.timer = self.create_timer(
            1.0 / config.tick_rate,
            self._inference_callback
        )

        # Create status timer (every 5 seconds)
        self.status_timer = self.create_timer(5.0, self._status_callback)

        self.get_logger().info("OmniVLA Drone Node initialized")

    def _load_goal_image(self):
        path = self.config.full_goal_image_path
        if os.path.exists(path):
            self.goal_image = Image.open(path).convert("RGB")
            self.get_logger().info(f"[OK] Goal image: {path}")
        else:
            self.get_logger().warn(f"[WARN] Goal image not found: {path}")
            self.goal_image = Image.new("RGB", (640, 360), (128, 128, 128))

    def load_models(self):
        """Load OmniVLA models"""
        config = self.config

        self.get_logger().info(f"Loading models from: {config.vla_path}")

        if not os.path.exists(config.vla_path):
            raise FileNotFoundError(f"Model path not found: {config.vla_path}")

        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.get_logger().info(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.get_logger().warn("[WARN] No GPU - will be slow!")

        # Register model classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

        # Load processor
        self.get_logger().info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(config.vla_path, trust_remote_code=True)

        # Load VLA
        self.get_logger().info("Loading VLA model...")
        self.vla = AutoModelForVision2Seq.from_pretrained(
            config.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.vla.vision_backbone.set_num_images_in_input(2)
        self.vla.to(dtype=torch.bfloat16, device=self.device)

        # Load pose projector
        self.get_logger().info("Loading pose projector...")
        self.pose_projector = ProprioProjector(
            llm_dim=self.vla.llm_dim,
            proprio_dim=POSE_DIM
        )
        state_dict = load_checkpoint("pose_projector", config.vla_path, config.resume_step)
        self.pose_projector.load_state_dict(state_dict)
        self.pose_projector.to(self.device)

        # Load action head
        self.get_logger().info("Loading action head...")
        self.action_head = L1RegressionActionHead_idcat(
            input_dim=self.vla.llm_dim,
            hidden_dim=self.vla.llm_dim,
            action_dim=ACTION_DIM
        )
        state_dict = load_checkpoint("action_head", config.vla_path, config.resume_step)
        self.action_head.load_state_dict(state_dict)
        self.action_head.to(torch.bfloat16).to(self.device)

        # Calculate patches
        self.num_patches = self.vla.vision_backbone.get_num_patches() * 2 + 1

        # Action tokenizer
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        self.get_logger().info("=" * 50)
        self.get_logger().info("[OK] Models loaded successfully!")
        self.get_logger().info("=" * 50)

    # -------------------- Callbacks --------------------

    def _image_callback(self, msg: RosImage):
        """Handle incoming images"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            with self.lock:
                self.current_image = Image.fromarray(cv_img)
                self.last_image_time = self.get_clock().now()
                self.image_count += 1
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def _odom_callback(self, msg: Odometry):
        """Handle odometry messages"""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)

        with self.lock:
            self.current_pose = (pos.x, pos.y, pos.z, yaw)
            self.last_odom_time = self.get_clock().now()
            self.odom_count += 1

    def _imu_callback(self, msg: Imu):
        """Handle IMU messages"""
        ori = msg.orientation
        _, _, yaw = euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)

        with self.lock:
            # Update yaw from IMU (more accurate/faster than odom)
            if self.current_pose is not None:
                x, y, z, _ = self.current_pose
                self.current_pose = (x, y, z, yaw)
            self.last_imu_time = self.get_clock().now()
            self.imu_count += 1

    def _status_callback(self):
        """Print status every 5 seconds"""
        with self.lock:
            img_ok = self.current_image is not None
            odom_ok = self.current_pose is not None

        status = []
        status.append(f"Image: {'OK' if img_ok else 'NONE'} ({self.image_count})")
        status.append(f"Odom: {'OK' if odom_ok else 'NONE'} ({self.odom_count})")
        status.append(f"IMU: ({self.imu_count})")
        status.append(f"Inferences: {self.inference_count}")

        self.get_logger().info(f"[STATUS] {' | '.join(status)}")

    def _inference_callback(self):
        """Main inference loop"""
        if self.vla is None:
            return

        # Get current state
        with self.lock:
            if self.current_image is None:
                return
            if self.current_pose is None:
                return

            image = self.current_image.copy()
            pose = self.current_pose

        # Run inference
        try:
            linear_vel, angular_vel = self._run_inference(image, pose)

            # Publish command
            cmd = Twist()
            cmd.linear.x = float(linear_vel)
            cmd.angular.z = float(angular_vel)
            self.cmd_vel_pub.publish(cmd)

            # Log every 10 inferences
            if self.inference_count % 10 == 0:
                x, y, z, yaw = pose
                self.get_logger().info(
                    f"[{self.inference_count}] pos=({x:.1f},{y:.1f}) yaw={math.degrees(yaw):.0f}° "
                    f"→ v={linear_vel:.2f} w={angular_vel:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()

    def _run_inference(self, image: Image.Image, pose: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Run OmniVLA inference"""
        config = self.config
        x, y, z, yaw = pose

        # Calculate goal in robot frame
        goal_x, goal_y = config.goal_x, config.goal_y

        dx = goal_x - x
        dy = goal_y - y
        rel_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        rel_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

        # Clamp distance
        dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
        if dist > config.max_goal_distance:
            scale = config.max_goal_distance / dist
            rel_x *= scale
            rel_y *= scale

        # Goal pose vector
        goal_yaw_rel = config.goal_yaw - yaw
        goal_pose = np.array([
            rel_y / config.metric_waypoint_spacing,
            -rel_x / config.metric_waypoint_spacing,
            np.cos(goal_yaw_rel),
            np.sin(goal_yaw_rel)
        ], dtype=np.float32)

        # Prepare batch
        batch = self._prepare_batch(image, goal_pose)
        modality_id = self._get_modality_id()

        # Forward pass
        actions = self._forward_pass(batch, modality_id)
        waypoints = actions.detach().float().cpu().numpy()[0]

        # Extract velocity
        wp = waypoints[config.waypoint_select].copy()
        wp[:2] *= config.metric_waypoint_spacing
        dx_wp, dy_wp, hx, hy = wp

        DT = 1.0 / config.tick_rate
        EPS = 1e-8

        if abs(dx_wp) < EPS and abs(dy_wp) < EPS:
            linear_vel = 0.0
            angular_vel = np.arctan2(hy, hx) / DT
        elif abs(dx_wp) < EPS:
            linear_vel = 0.0
            angular_vel = np.sign(dy_wp) * np.pi / (2 * DT)
        else:
            linear_vel = dx_wp / DT
            angular_vel = np.arctan(dy_wp / dx_wp) / DT

        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -1.0, 1.0)

        # Apply limits
        linear_vel, angular_vel = self._limit_velocity(linear_vel, angular_vel)

        # Save viz
        if config.save_visualizations:
            self._save_visualization(image, goal_pose, waypoints, linear_vel, angular_vel)

        self.inference_count += 1
        return linear_vel, angular_vel

    def _get_modality_id(self) -> torch.Tensor:
        cfg = self.config
        s, p, i, l = cfg.use_satellite, cfg.use_pose_goal, cfg.use_image_goal, cfg.use_language_prompt

        if s and not l and not p and not i:
            return torch.tensor([0], dtype=torch.float32)
        elif s and not l and p and not i:
            return torch.tensor([1], dtype=torch.float32)
        elif s and not l and not p and i:
            return torch.tensor([2], dtype=torch.float32)
        elif s and not l and p and i:
            return torch.tensor([3], dtype=torch.float32)
        elif not s and not l and p and not i:
            return torch.tensor([4], dtype=torch.float32)
        elif not s and not l and p and i:
            return torch.tensor([5], dtype=torch.float32)
        elif not s and not l and not p and i:
            return torch.tensor([6], dtype=torch.float32)
        elif not s and l and not p and not i:
            return torch.tensor([7], dtype=torch.float32)
        elif not s and l and p and not i:
            return torch.tensor([8], dtype=torch.float32)
        return torch.tensor([6], dtype=torch.float32)

    def _prepare_batch(self, image: Image.Image, goal_pose: np.ndarray) -> dict:
        cfg = self.config
        actions = np.random.rand(8, 4).astype(np.float32)

        inst = cfg.language_instruction if cfg.use_language_prompt else "xxxx"
        action_str = ''.join(self.action_tokenizer(actions))

        if inst == "xxxx":
            prompt = f"No language instruction{action_str}"
        else:
            prompt = f"What action should the robot take to {inst}?{action_str}"

        tokens = self.processor.tokenizer(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens.input_ids])
        labels = input_ids.clone()
        labels[:, :-len(action_str) - 1] = -100
        attention_mask = torch.ones_like(input_ids)

        pv_current = self.processor.image_processor.apply_transform(image)
        pv_goal = self.processor.image_processor.apply_transform(self.goal_image)
        pixel_values = torch.cat([pv_current.unsqueeze(0), pv_goal.unsqueeze(0)], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "goal_pose": torch.tensor([goal_pose]),
            "actions": torch.tensor([actions]),
        }

    def _forward_pass(self, batch: dict, modality_id: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                modality_id=modality_id.to(torch.bfloat16).to(self.device),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self.device),
                proprio_projector=self.pose_projector,
            )

        gt_tokens = batch["labels"][:, 1:].to(self.device)
        cur_mask = get_current_action_mask(gt_tokens)
        next_mask = get_next_actions_mask(gt_tokens)

        hidden = output.hidden_states[-1][:, self.num_patches:-1]
        act_hidden = hidden[cur_mask | next_mask].reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        act_hidden = act_hidden.to(torch.bfloat16)

        return self.action_head.predict_action(
            act_hidden,
            modality_id.to(torch.bfloat16).to(self.device)
        )

    def _limit_velocity(self, linear: float, angular: float) -> Tuple[float, float]:
        maxv = self.config.max_linear_vel
        maxw = self.config.max_angular_vel

        if abs(linear) <= maxv and abs(angular) <= maxw:
            return linear, angular
        elif abs(linear) <= maxv:
            rd = linear / (angular + 1e-8)
            return maxw * np.sign(linear) * abs(rd), maxw * np.sign(angular)
        elif abs(angular) <= 0.001:
            return maxv * np.sign(linear), 0.0
        else:
            rd = linear / angular
            if abs(rd) >= maxv / maxw:
                return maxv * np.sign(linear), maxv * np.sign(angular) / abs(rd)
            return maxw * np.sign(linear) * abs(rd), maxw * np.sign(angular)

    def _save_visualization(self, image: Image.Image, goal_pose: np.ndarray,
                            waypoints: np.ndarray, linear: float, angular: float):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(np.array(image))
        axes[0].set_title("Current View", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(np.array(self.goal_image))
        axes[1].set_title("Goal View", fontsize=12)
        axes[1].axis('off')

        x_seq = waypoints[:, 0]
        y_seq = -waypoints[:, 1]
        axes[2].plot(np.insert(y_seq, 0, 0), np.insert(x_seq, 0, 0),
                     'bo-', linewidth=2, markersize=8, label='Predicted')
        axes[2].plot(0, 0, 'gs', markersize=15, label='Drone')

        if self.config.use_pose_goal:
            axes[2].plot(-goal_pose[1], goal_pose[0], 'r*', markersize=20, label='Goal')

        axes[2].set_xlim(-5, 5)
        axes[2].set_ylim(-1, 15)
        axes[2].set_xlabel('Left/Right')
        axes[2].set_ylabel('Forward')
        axes[2].set_title(f"v={linear:.2f} w={angular:.2f}", fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.full_viz_output_dir, f"{self.inference_count:06d}.jpg")
        plt.savefig(save_path, dpi=80)
        plt.close()


# ===============================================================
# Main
# ===============================================================
def main():
    print("=" * 60)
    print("OmniVLA ROS2 Drone Adapter (FIXED QoS)")
    print("=" * 60)
    print(f"OMNIVLA_ROOT:        {OMNIVLA_ROOT}")
    print(f"OMNIVLA_CHECKPOINTS: {OMNIVLA_CHECKPOINTS}")
    print("=" * 60)

    rclpy.init()

    config = DroneConfig()
    print(f"\nConfig:")
    print(f"  Model: {config.vla_path}")
    print(f"  Goal image: {config.full_goal_image_path}")
    print(f"  Modality: image={config.use_image_goal}, pose={config.use_pose_goal}, lang={config.use_language_prompt}")
    print(f"\nTopics:")
    print(f"  Image: {config.image_topic}")
    print(f"  Odom:  {config.odom_topic}")
    print(f"  IMU:   {config.imu_topic}")
    print(f"  Cmd:   {config.cmd_vel_topic}")
    print()

    node = OmniVLADroneNode(config)

    print("Loading models...")
    try:
        node.load_models()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print("\n" + "=" * 60)
    print("Ready! Waiting for sensor data...")
    print("Status updates every 5 seconds")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())