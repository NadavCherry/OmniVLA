#!/bin/bash
# ============================================
# Quick Goal Image Capture
# ============================================
# Usage: ./capture_goal.sh [optional_name.jpg]
#
# Captures the current drone camera view and
# saves it as the goal image for OmniVLA.
# ============================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use environment variable or default
OMNIVLA_ROOT="${OMNIVLA_ROOT:-/root/ros2_ws/OmniVLA}"
SAVE_DIR="${OMNIVLA_ROOT}/inference"

# Default or custom filename
FILENAME="${1:-goal_img.jpg}"

# Image topic
TOPIC="/simple_drone/front/image_raw"

echo "============================================"
echo "Capturing Goal Image"
echo "============================================"
echo "Topic:  ${TOPIC}"
echo "Output: ${SAVE_DIR}/${FILENAME}"
echo "============================================"

# Source ROS2 if not already sourced
if ! command -v ros2 &> /dev/null; then
    source /opt/ros/humble/setup.bash 2>/dev/null || true
fi

# Check if topic exists
if ! ros2 topic list | grep -q "${TOPIC}"; then
    echo "[ERROR] Topic ${TOPIC} not found!"
    echo "Available image topics:"
    ros2 topic list | grep -i image || echo "  (none found)"
    exit 1
fi

# Run the capture script
python3 "${SCRIPT_DIR}/capture_goal_image.py" \
    --topic "${TOPIC}" \
    --dir "${SAVE_DIR}" \
    --name "${FILENAME}" \
    --timeout 10

echo ""
echo "Done! You can now run OmniVLA with this goal image."