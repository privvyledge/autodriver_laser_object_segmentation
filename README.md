# autodriver_laser_object_segmentation

An optimized, real-time ROS 2 perception package for 2D LiDAR scanners. It provides robust obstacle clustering, multi-model geometric shape fitting, and multi-target tracking. The package is optimized for small-scale autonomous platforms (such as F1/10 vehicles or Jetson Orin Nano) running 2D LiDARs (e.g., YDLIDAR X4) but can be used with and extended to support depth image (and PointCloud2) to laserscan. 

It contains a dual-language implementation with mirror algorithms in both C++ and Python:
- **C++:** High-performance, sub-millisecond component node.
- **Python:** Highly readable script node, ideal for prototyping and verification.

---

## Architecture & Algorithm Pipeline

1. **Preprocessing:** Optionally applies a 1D median filter (window size = 3, toggled by `use_median_filter`) to suppress salt-and-pepper noise, filters points outside sensor/user ranges (`min_range` to `max_range`), and projects valid beams to 2D Cartesian coordinates. All preprocessing, clustering, and shape fitting happen in the **sensor frame**.
2. **Clustering:** Performs range-adaptive **Sequential Jump Distance Clustering (JDC)** based on the Dietmayer formula to segment adjacent points into clusters, handling wrap-around for 360° scans (vectorized).
3. **Shape Fitting:** Fits multiple geometric models to clusters:
   - **Circle:** Least-squares circle fitting (Kasa method).
   - **Oriented Bounding Box (OBB):** Minimum-area box via **rotating calipers** on the convex hull (exact, no angular quantization).
   - **Line / Corner:** Split-and-Merge wall segment extraction.
   - **Convex Hull:** `scipy.spatial.ConvexHull` (Python) / Andrew's monotone chain (C++), both CCW-wound, to generate boundary polygons for arbitrary shapes.
4. **Frame transform:** When a sensor pose is available (from a TF lookup `tracking_frame → scan frame`), detections are transformed from the sensor frame into the **tracking frame** before tracking, so ego-motion is removed and static obstacles report near-zero velocity. Without TF, the package falls back to tracking in the sensor frame.
5. **Multi-Target Tracking:** Associates detections to tracks using a **constant-velocity Kalman Filter** with gating and selectable data association — **Hungarian** (`scipy.optimize.linear_sum_assignment`, globally optimal, default) or **greedy** nearest-neighbor. The real inter-scan `dt` is derived from message timestamps. Tracks are initialized, confirmed, and deleted based on observation age; light EMA smoothing (`shape_smoothing_alpha`) stabilizes shape dimensions.

---

## ROS 2 Topics

### Subscriptions
* **`scan`** ([`sensor_msgs/msg/LaserScan`](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html))
  * Raw 2D LiDAR range data.
* **`/tf`, `/tf_static`** ([`tf2_msgs/msg/TFMessage`](https://docs.ros.org/en/humble/p/tf2_msgs/interfaces/msg/TFMessage.html))
  * Used to look up `tracking_frame → scan frame` for ego-motion-aware tracking. Optional: if unavailable, tracking falls back to the sensor frame.

### Publications
* **`obstacles`** ([`derived_object_msgs/msg/ObjectArray`](https://docs.ros.org/en/noetic/api/derived_object_msgs/html/msg/ObjectArray.html))
  * Tracked obstacles, published in the `tracking_frame` (or the scan frame on TF fallback). Each object contains its tracking ID, 2D pose (position & orientation), linear velocity, shape geometry (primitive box/cylinder dimensions), and polygon boundary points. Confirmed tracks (age ≥ `min_track_age`) are marked `OBJECT_TRACKED`; when `publish_unconfirmed` is true, tentative tracks are also published as `OBJECT_DETECTED`.
* **`debug_clusters`** ([`sensor_msgs/msg/PointCloud2`](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html)) *[Optional]*
  * A color-coded point cloud visualizing cluster assignments. Controlled by `publish_debug_pointcloud`.
* **`debug_markers`** ([`visualization_msgs/msg/MarkerArray`](https://docs.ros.org/en/melodic/api/visualization_msgs/html/msg/MarkerArray.html)) *[Optional]*
  * RViz2 visualization aids showcasing oriented bounding boxes, cylinders, velocity vectors, polygon boundaries, and text labels (Track ID + Velocity). Controlled by `publish_debug_markers`.

---

## Parameters

Configurable parameters are declared in [config/params.yaml](config/params.yaml):

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`min_range`** | `double` | `0.15` | Minimum range filtering threshold (m). Helps ignore vehicle chassis reflection. |
| **`max_range`** | `double` | `8.0` | Maximum range filtering threshold (m) for obstacle detection. |
| **`beta_incidence_deg`** | `double` | `12.0` | Angle of incidence threshold (deg) for Dietmayer JDC. |
| **`sigma_r`** | `double` | `0.02` | Standard deviation of scan range measurements (m) (~2cm for YDLIDAR X4). |
| **`min_jump_distance`** | `double` | `0.15` | Lower bound distance (m) to segment adjacent objects. |
| **`max_jump_distance`** | `double` | `1.0` | Upper bound distance threshold (m) for JDC. |
| **`min_cluster_points`** | `int` | `4` | Minimum number of points per cluster to filter out noise. |
| **`max_cluster_points`** | `int` | `2000` | Maximum points per cluster. Raised so long walls survive; only rejects pathological "whole room as one blob" cases. |
| **`use_median_filter`** | `bool` | `true` | Apply the 1D median prefilter. Disable when the scan is already cleaned (e.g. by `laser_filters`). |
| **`use_convex_hull`** | `bool` | `true` | Enables output of convex hulls for arbitrary/dynamic shapes (ideal for MPC/CBF). |
| **`split_threshold`** | `double` | `0.04` | Threshold (m) for Split-and-Merge line extraction. |
| **`circle_residual_ratio`** | `double` | `0.12` | Max residual/radius ratio to accept a cluster as a circle. |
| **`max_circle_radius`** | `double` | `1.0` | Max radius (m) to accept as a circle. |
| **`corner_angle_min_deg`** | `double` | `65.0` | Lower angle bound (deg) for L-shape corner fitting. |
| **`corner_angle_max_deg`** | `double` | `115.0` | Upper angle bound (deg) for L-shape corner fitting. |
| **`tracking_frame`** | `string` | `"odom"` | Frame for KF tracking and output (poses & velocities). Empty string or TF failure → track in the sensor frame. |
| **`association_method`** | `string` | `"hungarian"` | Data association: `hungarian` (globally optimal, recommended) or `greedy`. |
| **`max_association_distance`** | `double` | `0.8` | Kalman Filter gating threshold (m) for tracking data association. |
| **`min_track_age`** | `int` | `3` | Required consecutive observed frames before a track is confirmed (`OBJECT_TRACKED`). |
| **`max_missed_frames`** | `int` | `5` | Allowable consecutive missed updates before deleting a track. |
| **`publish_unconfirmed`** | `bool` | `true` | Also publish tentative (unconfirmed) tracks as `OBJECT_DETECTED`. |
| **`shape_smoothing_alpha`** | `double` | `0.5` | EMA coefficient on shape dimensions (`1.0` = no smoothing). |
| **`kf_process_noise`** | `double` | `0.1` | Kalman process noise `q`. Lower = smoother tracks and less phantom velocity on static objects; raise if fast dynamic obstacles lag. |
| **`shape_type_hysteresis`** | `int` | `3` | A track must observe a differing shape type this many consecutive frames before switching (`1` = off). Suppresses circle↔box↔line flicker; use `5` for near-zero flips. |
| **`publish_debug_pointcloud`** | `bool` | `true` | Flag to publish colorized point cloud clusters on `debug_clusters`. |
| **`publish_debug_markers`** | `bool` | `true` | Flag to publish RViz visualization markers on `debug_markers`. |

---

## Performance (Latency & Frame Rate)

Latency and frame rate benchmarks were gathered using the built-in profiling script [test/profile_latency.py](test/profile_latency.py) over 1000 iterations on a 360-beam scan containing 5 distinct obstacles (circle, box, line, corner, and hull shapes):

### Python Implementation (Core Library)
Measured on an x86 development host after vectorization of the clustering, shape-fitting, and association hot loops:
* **Average Latency:** `1.377 ms`
* **Minimum Latency:** `1.097 ms`
* **Maximum Latency:** `5.361 ms`
* **Standard Deviation:** `0.306 ms`
* **Equivalent Frame Rate:** `~726 Hz`

### C++ Implementation (Core Library)
* **Execution Latency:** Sub-millisecond (`< 1.0 ms`) on the development host.
* Designed for low-latency control loops (e.g., Model Predictive Control and Control Barrier Functions) on embedded platforms like NVIDIA Jetson Orin Nano.
* **Note:** the sub-millisecond figure is measured on x86. Re-measure on the Jetson (ARM) target before relying on it for a specific control-loop budget.

---

## Building and Running

### Build
To build the package in your ROS 2 workspace:
```bash
colcon build --packages-select autodriver_laser_object_segmentation
source install/setup.bash
```

### Launch
Run the perception system using the provided launch file [launch/obstacle_detector.launch.py](launch/obstacle_detector.launch.py):

* **To run the C++ node (Default):**
  ```bash
  ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py use_cpp_node:=true
  ```

* **To run the Python node:**
  ```bash
  ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py use_cpp_node:=false
  ```

* **To load custom parameters:**
  ```bash
  ros2 launch autodriver_laser_object_segmentation obstacle_detector.launch.py params_file:=/path/to/custom_params.yaml
  ```

### Replay a recorded rosbag
For F1TENTH test bags, [scripts/run_bag_test.sh](scripts/run_bag_test.sh) launches the C++ node + RViz2 with simulated time and the standard `/gosling1/...` → `scan` / `/tf` / `/odom` remaps, then plays the bag. It auto-detects `no_localization` bags and publishes a static `odom → base_link` transform for them:
```bash
./src/autodriver_laser_object_segmentation/scripts/run_bag_test.sh \
  /mnt/d/Coding/Projects/f1tenth/test_bags_05252026/stationary_with_localization_and_pointcloud
```
Stationary bags isolate detector/tracker behavior from ego-motion. See [TESTS.md](TESTS.md) §9 for details.

For cloning, dependency install, unit tests, and running against namespaced topics (with TF remaps for ego-motion-aware tracking), see [TESTS.md](TESTS.md).
