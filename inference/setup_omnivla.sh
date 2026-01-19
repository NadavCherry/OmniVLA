#!/bin/bash
# ============================================
# OmniVLA Setup & Verification Script
# ============================================
# Run this INSIDE the Docker container to:
# 1. Install missing dependencies
# 2. Verify everything is ready
# ============================================

set -e

echo "============================================"
echo "OmniVLA Setup & Verification"
echo "============================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

# Check environment variables
echo -e "\n${YELLOW}[1/6] Checking environment variables...${NC}"
if [[ -z "$OMNIVLA_ROOT" ]]; then
    echo -e "${RED}✗ OMNIVLA_ROOT not set${NC}"
    echo "  Expected: set by run.sh"
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}✓ OMNIVLA_ROOT=$OMNIVLA_ROOT${NC}"
fi

if [[ -z "$OMNIVLA_CHECKPOINTS" ]]; then
    echo -e "${RED}✗ OMNIVLA_CHECKPOINTS not set${NC}"
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}✓ OMNIVLA_CHECKPOINTS=$OMNIVLA_CHECKPOINTS${NC}"
fi

# Check OmniVLA code
echo -e "\n${YELLOW}[2/6] Checking OmniVLA code...${NC}"
if [[ -d "$OMNIVLA_ROOT/prismatic" ]]; then
    echo -e "${GREEN}✓ OmniVLA code found at $OMNIVLA_ROOT${NC}"
else
    echo -e "${RED}✗ OmniVLA code not found${NC}"
    echo "  Run on host: git clone https://github.com/NHirose/OmniVLA.git"
    ERRORS=$((ERRORS+1))
fi

# Check model checkpoints
echo -e "\n${YELLOW}[3/6] Checking model checkpoints...${NC}"
CKPT_DIR="$OMNIVLA_CHECKPOINTS/omnivla-original"
if [[ -f "$CKPT_DIR/config.json" ]]; then
    echo -e "${GREEN}✓ Model config found${NC}"
else
    echo -e "${RED}✗ Model checkpoints not found at $CKPT_DIR${NC}"
    echo "  Run on host:"
    echo "    mkdir -p <workspace>/models/omnivla"
    echo "    cd <workspace>/models/omnivla"
    echo "    git clone https://huggingface.co/NHirose/omnivla-original"
    ERRORS=$((ERRORS+1))
fi

if [[ -f "$CKPT_DIR/action_head--120000_checkpoint.pt" ]]; then
    echo -e "${GREEN}✓ Action head checkpoint found${NC}"
else
    echo -e "${RED}✗ Action head checkpoint missing${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check goal image
echo -e "\n${YELLOW}[4/6] Checking goal image...${NC}"
if [[ -f "$OMNIVLA_ROOT/inference/goal_img.jpg" ]]; then
    echo -e "${GREEN}✓ Goal image found${NC}"
else
    echo -e "${YELLOW}⚠ Goal image not found (will use placeholder)${NC}"
    echo "  Create one with: python capture_goal_image.py"
fi

# Check/install Python dependencies
echo -e "\n${YELLOW}[5/6] Checking Python dependencies...${NC}"

# Function to check and install
check_install() {
    local pkg=$1
    local import_name=${2:-$1}
    if python3 -c "import $import_name" 2>/dev/null; then
        echo -e "${GREEN}✓ $pkg${NC}"
    else
        echo -e "${YELLOW}Installing $pkg...${NC}"
        pip install $pkg --quiet
        if python3 -c "import $import_name" 2>/dev/null; then
            echo -e "${GREEN}✓ $pkg (installed)${NC}"
        else
            echo -e "${RED}✗ Failed to install $pkg${NC}"
            ERRORS=$((ERRORS+1))
        fi
    fi
}

check_install "torch"
check_install "transformers"
check_install "accelerate"
check_install "utm"
check_install "Pillow" "PIL"
check_install "opencv-python" "cv2"
check_install "matplotlib"

# Check cv_bridge (ROS2 package)
if python3 -c "from cv_bridge import CvBridge" 2>/dev/null; then
    echo -e "${GREEN}✓ cv_bridge${NC}"
else
    echo -e "${YELLOW}Installing cv_bridge...${NC}"
    apt-get update -qq && apt-get install -y -qq ros-humble-cv-bridge python3-cv-bridge
fi

# Check CUDA
echo -e "\n${YELLOW}[6/6] Checking GPU...${NC}"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}✓ CUDA available: $GPU_NAME${NC}"
else
    echo -e "${RED}✗ CUDA not available (will be very slow!)${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check ROS2 topics
echo -e "\n${YELLOW}Checking ROS2 topics...${NC}"
if command -v ros2 &> /dev/null; then
    if ros2 topic list 2>/dev/null | grep -q "/simple_drone/front/image_raw"; then
        echo -e "${GREEN}✓ Camera topic available${NC}"
    else
        echo -e "${YELLOW}⚠ Camera topic not found (is simulation running?)${NC}"
    fi

    if ros2 topic list 2>/dev/null | grep -q "/simple_drone/odom"; then
        echo -e "${GREEN}✓ Odometry topic available${NC}"
    else
        echo -e "${YELLOW}⚠ Odometry topic not found (is simulation running?)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ ROS2 not sourced${NC}"
fi

# Summary
echo ""
echo "============================================"
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}All checks passed! Ready to run OmniVLA.${NC}"
    echo ""
    echo "To run:"
    echo "  cd \$OMNIVLA_ROOT"
    echo "  python inference/omnivla_ros2_drone.py"
else
    echo -e "${RED}$ERRORS error(s) found. Please fix before running.${NC}"
fi
echo "============================================"

exit $ERRORS