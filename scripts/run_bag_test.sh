#!/usr/bin/env bash

# Source ROS 2 Humble/workspace environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
else
    echo "Warning: workspace setup.bash not found at $WS_DIR/install/setup.bash. Sourcing system ROS only."
fi

# Get ROSBAG path from command line arguments (with default value)
BAG_PATH="${1:-/mnt/d/Coding/Projects/f1tenth/figure8_bags/figure8_with_localization_and_pointcloud}"

if [ ! -d "$BAG_PATH" ]; then
    echo "Error: ROSBAG directory not found at $BAG_PATH"
    exit 1
fi

echo "Using ROSBAG: $BAG_PATH"

# Optional looping: pass `--loop` (anywhere in the args) or set LOOP=1 to replay the bag
# continuously (handy for live tuning so it doesn't stop after one pass). Runs until Ctrl+C.
LOOP_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--loop" ]; then LOOP_FLAG="--loop"; fi
done
if [ "${LOOP:-0}" != "0" ]; then LOOP_FLAG="--loop"; fi
if [ -n "$LOOP_FLAG" ]; then
    echo "Looping enabled: bag will replay continuously until Ctrl+C."
    echo "  (Note: each loop restart jumps /clock backward and briefly clears tf2; the"
    echo "   bag re-publishes /tf_static at loop start, so the static tree recovers.)"
fi

# Only replay the topics the detector + RViz actually need. The full bag carries
# heavy camera/depth/pointcloud/imu streams; replaying all of them starves the
# rosbag2 player queue and delays TF, which causes "extrapolation into the future"
# lookups and RViz dropping every scan. Filtering to these keeps TF in lockstep
# with the scan.
COMMON_TOPICS="/gosling1/tf /gosling1/tf_static /gosling1/lidar/scan_filtered"
COMMON_REMAPS="-m /gosling1/tf:=/tf /gosling1/tf_static:=/tf_static /gosling1/lidar/scan_filtered:=/scan"

# Conditional setup for bags without localization
if [[ "$BAG_PATH" == *"no_localization"* ]]; then
    echo "Detected bag without localization. Remapping VESC odometry to /odom..."
    PLAY_TOPICS="$COMMON_TOPICS /gosling1/vehicle/vesc_odom"
    PLAY_REMAPS="$COMMON_REMAPS /gosling1/vehicle/vesc_odom:=/odom"

    echo "Launching static transform publisher for odom -> base_link..."
    ros2 run tf2_ros static_transform_publisher --frame-id odom --child-frame-id base_link &
else
    PLAY_TOPICS="$COMMON_TOPICS /gosling1/odometry/local"
    PLAY_REMAPS="$COMMON_REMAPS /gosling1/odometry/local:=/odom"
fi

# Trap exits (e.g. Ctrl+C or closing RViz) to clean up all background processes
trap "echo 'Cleaning up and stopping background nodes...'; kill 0" EXIT

# ---------------------------------------------------------------------------
# Start bag playback BEFORE the node and RViz.
#
# With use_sim_time, a node/RViz started before /clock exists runs on wall time
# and then jumps *backwards* to the bag's (older) timestamps when --clock starts.
# tf2 clears its buffer on a backward jump, and because rosbag2 publishes
# /tf_static only ONCE, the static sensor-mount transforms (base_link -> lidar)
# are wiped and never re-sent -> "odom and lidar are not part of the same tree".
#
# Starting the bag first means /clock is already live when the node/RViz come up,
# so they initialize at sim time (forward jump only -> no buffer clear), and the
# latched (transient_local) /tf_static is delivered to them on subscribe.
# ---------------------------------------------------------------------------

echo "Starting ROSBAG play (filtered topics) ..."
ros2 bag play "$BAG_PATH" \
  --clock \
  --read-ahead-queue-size 2000 \
  $LOOP_FLAG \
  --topics $PLAY_TOPICS \
  $PLAY_REMAPS &
BAG_PID=$!

# Give the player a few seconds to open the bag and get /clock flowing before the
# node/RViz come up, so they initialize at sim time (no backward jump). With the
# topic filter above the bag opens quickly, so a short fixed wait is reliable;
# increase WAIT_SECS if you still see a startup TF warning on a slow machine.
# (A `ros2 topic echo /clock` poll was tried and proved flaky under WSL/DDS.)
WAIT_SECS="${CLOCK_WAIT_SECS:-8}"
echo "Waiting ${WAIT_SECS}s for /clock (sim time) to go live before starting nodes..."
sleep "$WAIT_SECS"

echo "Launching Laser Obstacle Detector Node..."
ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py use_cpp_node:=true use_sim_time:=true &

echo "Launching RViz2 with pre-configured visualization layout..."
rviz2 -d "$SCRIPT_DIR/../config/obstacle_detector.rviz" --ros-args -p use_sim_time:=true &

# Block until the bag finishes playing, then let the trap clean up the rest.
wait "$BAG_PID"

echo "Bag playback finished."
