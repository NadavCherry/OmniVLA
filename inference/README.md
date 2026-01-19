# OmniVLA Drone Navigation - Quick Start

## One-Time Setup (on host machine)

```bash
cd <workspace_root>

# Clone OmniVLA
git clone https://github.com/NHirose/OmniVLA.git

# Download model checkpoints (~14GB)
mkdir -p models/omnivla
cd models/omnivla
git clone https://huggingface.co/NHirose/omnivla-original

# Copy adapter files to OmniVLA
cp omnivla_ros2_drone.py <workspace_root>/OmniVLA/inference/
cp capture_goal_image.py <workspace_root>/OmniVLA/inference/
```

---

## Running OmniVLA

### Step 1: Start Simulation

```bash
cd sjtu_drone
./run.sh hospital.world
```

### Step 2: Open New Terminal & Enter Container

```bash
docker exec -it sjtu_drone_hospital bash
```

### Step 3: Install OmniVLA (first time only)

```bash
cd $OMNIVLA_ROOT
pip install -e .
```

### Step 4: Capture Goal Image

```bash
cd $OMNIVLA_ROOT/inference
python capture_goal_image.py
```

### Step 5: Run OmniVLA

```bash
cd $OMNIVLA_ROOT
python inference/omnivla_ros2_drone.py
```

---

## Goal Modes

By default, **Image Goal** is enabled. The drone navigates toward `goal_img.jpg`.

To change mode, edit `omnivla_ros2_drone.py` around line 70:

| Mode | Settings |
|------|----------|
| Image Goal | `use_image_goal=True` |
| Language | `use_language_prompt=True`, edit `language_instruction` |
| Position | `use_pose_goal=True`, edit `goal_x`, `goal_y` |

---

## Files

| File | Location | Purpose |
|------|----------|---------|
| `omnivla_ros2_drone.py` | `OmniVLA/inference/` | Main adapter |
| `capture_goal_image.py` | `OmniVLA/inference/` | Capture goal images |
| `goal_img.jpg` | `OmniVLA/inference/` | Target image |
| Model checkpoints | `models/omnivla/omnivla-original/` | Neural network weights |