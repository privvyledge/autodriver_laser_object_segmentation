# Build, Test & Run Guide

Commands for building, unit-testing, and running the `autodriver_laser_object_segmentation`
package. Tested on **ROS 2 Humble**.

The node clusters a 2D `LaserScan` into obstacles, fits shapes (circle / box / line / corner),
tracks them with a constant-velocity Kalman filter, and publishes
`derived_object_msgs/ObjectArray` on `obstacles`.

---

## 1. Clone into a workspace

The GitHub repository is the package itself, so it is cloned into a workspace `src/`:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/privvyledge/autodriver_laser_object_segmentation.git
```

---

## 2. Install dependencies

All dependencies (`derived_object_msgs`, `python3-scipy`, `tf2_ros`, etc.) are declared in
`package.xml`, so `rosdep` resolves them:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

---

## 3. Build

```bash
cd ~/ros2_ws
colcon build --packages-select autodriver_laser_object_segmentation --symlink-install
source install/setup.bash
```

---

## 4. Unit tests (no ROS runtime needed)

The suite exercises the Python core directly and runs C++ parity checks via `ctypes` against the
built shared library, so the build (step 3) must run first.

```bash
cd ~/ros2_ws/src/autodriver_laser_object_segmentation

# All tests
python3 -m pytest test/ -v

# A single test
python3 -m pytest test/test_laser_obstacle_detector_core.py::TestLaserObstacleDetectorCore::test_ego_motion -v

# Latency profiling (1000 iterations)
python3 test/profile_latency.py
```

Expected result: `14 passed`.

The same tests can be run through colcon:

```bash
cd ~/ros2_ws
colcon test --packages-select autodriver_laser_object_segmentation
colcon test-result --verbose
```

---

## 5. Run on namespaced topics

Example for a robot publishing under the `gosling1` namespace:

| Role | Topic / frame |
|---|---|
| Scan input | `/gosling1/scan_filtered` |
| Odometry (not subscribed — see note) | `/gosling1/odometry/local` |
| TF | `/gosling1/tf`, `/gosling1/tf_static` |

The node subscribes to the **relative** topic `scan`, and the tf2 listener subscribes to the
**absolute** topics `/tf` and `/tf_static`, so all three must be remapped. The launch file does not
expose remaps, so `ros2 run` is used for the namespaced setup.

> **`tracking_frame`** must be set to the name of the odometry frame **as it appears in the TF
> tree** (often namespaced, e.g. `gosling1/odom`). The node looks up
> `tracking_frame → <scan frame_id>` to remove ego-motion before tracking. If TF is unavailable it
> falls back to tracking in the sensor frame and logs a throttled warning; static objects then show
> phantom velocity, which is the signal that the lookup is failing. The exact frame names can be
> confirmed first:
>
> ```bash
> ros2 topic echo /gosling1/scan_filtered --field header.frame_id --once   # the scan frame
> ros2 run tf2_tools view_frames                                            # dumps the TF tree to a PDF
> ```

### C++ node (production, sub-ms)

```bash
ros2 run autodriver_laser_object_segmentation laser_obstacle_detector_node_exe \
  --ros-args \
  --params-file ~/ros2_ws/src/autodriver_laser_object_segmentation/config/params.yaml \
  -p tracking_frame:=gosling1/odom \
  -r scan:=/gosling1/scan_filtered \
  -r /tf:=/gosling1/tf \
  -r /tf_static:=/gosling1/tf_static
```

### Python node (readable, ~1.4 ms)

```bash
ros2 run autodriver_laser_object_segmentation laser_obstacle_detector.py \
  --ros-args \
  --params-file ~/ros2_ws/src/autodriver_laser_object_segmentation/config/params.yaml \
  -p tracking_frame:=gosling1/odom \
  -r scan:=/gosling1/scan_filtered \
  -r /tf:=/gosling1/tf \
  -r /tf_static:=/gosling1/tf_static
```

> **Note on odometry:** the node does *not* subscribe to `/gosling1/odometry/local`. Ego-motion is
> taken from TF (`tracking_frame → scan frame`). Whatever publishes `/gosling1/odometry/local` must
> also broadcast the corresponding `odom → base_link` transform on `/gosling1/tf`. To track in the
> sensor frame instead (no ego-motion removal), set `-p tracking_frame:=""`.

---

## 6. Default / non-namespaced quickstart (launch file)

For default topics (`scan`, `/tf`), the launch file is simplest:

```bash
ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py use_cpp_node:=true
ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py use_cpp_node:=false
ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py \
  params_file:=/path/to/custom_params.yaml
```

---

## 7. Inspect the output

```bash
# Obstacle list (frame_id is the tracking_frame when TF is healthy)
ros2 topic echo /obstacles

# Confirm publishing rate
ros2 topic hz /obstacles

# Debug topics (sensor frame)
ros2 topic echo /debug_clusters   # sensor_msgs/PointCloud2, colored clusters
ros2 topic echo /debug_markers    # visualization_msgs/MarkerArray
```

In RViz2: set **Fixed Frame** to the `tracking_frame` (e.g. `gosling1/odom`), then add
`PointCloud2` on `/debug_clusters` and `MarkerArray` on `/debug_markers`.

---

## 8. Sanity checklist for the first real run

- [ ] `ros2 topic hz /obstacles` shows roughly the scan rate.
- [ ] No repeating `TF lookup failed ...` warning in the node log. If present, fix `tracking_frame`
      / TF remaps before trusting velocities.
- [ ] A stationary object reports near-zero velocity while the robot drives, confirming ego-motion
      removal. Motion at `-v_ego` indicates TF is not resolving and the sensor-frame fallback is
      active.
- [ ] Object `detection_level` is `0` (`OBJECT_DETECTED`) for new tracks and `1`
      (`OBJECT_TRACKED`) after `min_track_age` (default 3) frames.

---

## 9. Replay a recorded rosbag (F1TENTH test bags)

`scripts/run_bag_test.sh` launches the C++ node + RViz2 with `use_sim_time:=true`, waits for
initialization, then plays a bag with the standard F1TENTH topic/TF remaps (`/gosling1/...` →
`scan`, `/tf`, `/tf_static`, `/odom`). It auto-detects localization: bags whose path contains
`no_localization` get a static `odom → base_link` transform and the VESC odometry remapped to
`/odom`.

```bash
cd ~/ros2_ws
colcon build --symlink-install \
  --cmake-args ' -DCMAKE_BUILD_TYPE=Release' -DPython3_FIND_VIRTUALENV="ONLY" \
  --packages-select autodriver_laser_object_segmentation
source install/setup.bash

# Stationary bag, with localization + pointcloud
./src/autodriver_laser_object_segmentation/scripts/run_bag_test.sh \
  /mnt/d/Coding/Projects/f1tenth/test_bags_05252026/stationary_with_localization_and_pointcloud

# Stationary bag, no localization (static odom->base_link is auto-published)
./src/autodriver_laser_object_segmentation/scripts/run_bag_test.sh \
  /mnt/d/Coding/Projects/f1tenth/test_bags_05252026/stationary_no_localization
```

Stationary bags are the best tuning fixture: with the sensor still, any obstacle velocity, marker
jitter, or shape flip is a detector/tracker artifact rather than ego-motion. To capture a log for
offline tuning while a bag plays:

```bash
ros2 bag record -o /tmp/laser_debug /scan /obstacles /debug_markers /tf /tf_static
```
</content>
