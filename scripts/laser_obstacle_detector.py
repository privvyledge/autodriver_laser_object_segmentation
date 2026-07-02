#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from derived_object_msgs.msg import Object, ObjectArray
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Point, Vector3, Quaternion, Polygon, Point32
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformListener
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange, SetParametersResult
try:
    from nav2_dynamic_msgs.msg import Obstacle, ObstacleArray
    NAV2_DYNAMIC_MSGS_AVAILABLE = True
except ImportError:
    NAV2_DYNAMIC_MSGS_AVAILABLE = False

import numpy as np
import struct
import math

from autodriver_laser_object_segmentation.laser_obstacle_detector_core import LaserObstacleDetectorCore


def _fr(lo, hi, step, desc=''):
    """ParameterDescriptor with a FloatingPointRange (min/max/step) for rqt_reconfigure sliders."""
    return ParameterDescriptor(
        description=desc,
        floating_point_range=[FloatingPointRange(from_value=float(lo), to_value=float(hi), step=float(step))]
    )


def _ir(lo, hi, step=1, desc=''):
    """ParameterDescriptor with an IntegerRange (min/max/step) for rqt_reconfigure sliders."""
    return ParameterDescriptor(
        description=desc,
        integer_range=[IntegerRange(from_value=int(lo), to_value=int(hi), step=int(step))]
    )


class LaserObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('laser_obstacle_detector')
        
        # Declare Parameters
        # Numeric params carry a ParameterDescriptor with min/max/step so rqt_reconfigure
        # renders sliders. Every param is re-read at the top of each scan callback, so
        # changes (via sliders or `ros2 param set`) take effect on the next scan with no
        # restart. Bool/string params get no range (rqt shows a checkbox / text field).
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('publish_object_array', True)
        self.declare_parameter('publish_obstacle_array', False)
        self.declare_parameter('min_range', 0.1, _fr(0.0, 5.0, 0.01))
        self.declare_parameter('max_range', 10.0, _fr(0.0, 30.0, 0.01))
        self.declare_parameter('beta_incidence_deg', 10.0, _fr(0.0, 45.0, 0.1))
        self.declare_parameter('sigma_r', 0.01, _fr(0.0, 0.2, 0.001))
        self.declare_parameter('min_jump_distance', 0.1, _fr(0.0, 2.0, 0.01))
        self.declare_parameter('max_jump_distance', 1.0, _fr(0.0, 5.0, 0.01))
        self.declare_parameter('min_cluster_points', 3, _ir(1, 50, 1))
        self.declare_parameter('max_cluster_points', 2000, _ir(1, 5000, 1))
        self.declare_parameter('use_convex_hull', True)
        self.declare_parameter('split_threshold', 0.05, _fr(0.0, 0.5, 0.001))
        self.declare_parameter('max_association_distance', 1.0, _fr(0.0, 5.0, 0.01))
        self.declare_parameter('min_track_age', 3, _ir(1, 20, 1))
        self.declare_parameter('max_missed_frames', 5, _ir(0, 30, 1))
        self.declare_parameter('publish_debug_pointcloud', True)
        self.declare_parameter('publish_debug_markers', True)
        self.declare_parameter('use_median_filter', True)
        self.declare_parameter('association_method', 'hungarian')
        self.declare_parameter('circle_residual_ratio', 0.12, _fr(0.0, 1.0, 0.01))
        self.declare_parameter('max_circle_radius', 1.0, _fr(0.0, 5.0, 0.01))
        self.declare_parameter('corner_angle_min_deg', 65.0, _fr(0.0, 180.0, 0.5))
        self.declare_parameter('corner_angle_max_deg', 115.0, _fr(0.0, 180.0, 0.5))
        self.declare_parameter('shape_smoothing_alpha', 0.5, _fr(0.0, 1.0, 0.01))
        self.declare_parameter('kf_process_noise', 0.1, _fr(0.0, 2.0, 0.01))
        self.declare_parameter('shape_type_hysteresis', 3, _ir(1, 20, 1))
        self.declare_parameter('tracking_frame', 'odom')
        self.declare_parameter('publish_unconfirmed', True)
        self.declare_parameter('tf_fallback_to_latest', True)
        self.declare_parameter('tf_latest_max_delay', 2.0, _fr(0.0, 10.0, 0.05))
        
        # Initialize Core
        beta_rad = math.radians(self.get_parameter('beta_incidence_deg').value)
        self.core = LaserObstacleDetectorCore(
            min_range=self.get_parameter('min_range').value,
            max_range=self.get_parameter('max_range').value,
            beta_incidence_rad=beta_rad,
            sigma_r=self.get_parameter('sigma_r').value,
            min_jump_distance=self.get_parameter('min_jump_distance').value,
            max_jump_distance=self.get_parameter('max_jump_distance').value,
            min_cluster_points=self.get_parameter('min_cluster_points').value,
            max_cluster_points=self.get_parameter('max_cluster_points').value,
            use_convex_hull=self.get_parameter('use_convex_hull').value,
            split_threshold=self.get_parameter('split_threshold').value,
            max_association_distance=self.get_parameter('max_association_distance').value,
            min_track_age=self.get_parameter('min_track_age').value,
            max_missed_frames=self.get_parameter('max_missed_frames').value,
            dt=0.1, # initial guess, will update dynamically
            use_median_filter=self.get_parameter('use_median_filter').value,
            association_method=self.get_parameter('association_method').value,
            circle_residual_ratio=self.get_parameter('circle_residual_ratio').value,
            max_circle_radius=self.get_parameter('max_circle_radius').value,
            corner_angle_min_deg=self.get_parameter('corner_angle_min_deg').value,
            corner_angle_max_deg=self.get_parameter('corner_angle_max_deg').value,
            shape_smoothing_alpha=self.get_parameter('shape_smoothing_alpha').value,
            kf_process_noise=self.get_parameter('kf_process_noise').value,
            shape_type_hysteresis=self.get_parameter('shape_type_hysteresis').value
        )
        
        self.last_stamp = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        from rclpy.qos import qos_profile_sensor_data
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        # Publishers
        self.obstacles_pub = self.create_publisher(
            ObjectArray,
            'obstacles',
            10
        )
        
        self.publish_pc = self.get_parameter('publish_debug_pointcloud').value
        if self.publish_pc:
            self.pc_pub = self.create_publisher(
                PointCloud2,
                'debug_clusters',
                10
            )
            
        self.publish_markers = self.get_parameter('publish_debug_markers').value
        if self.publish_markers:
            self.marker_pub = self.create_publisher(
                MarkerArray,
                'debug_markers',
                10
            )
            
        # Log parameter changes so live retuning (rqt_reconfigure / `ros2 param set`) gives
        # visible feedback. Registered AFTER declares so it doesn't fire for initial values.
        # The actual application happens in scan_callback, which re-reads every param each scan.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info('Laser Obstacle Detector Node initialized.')
        self._nav2_warned = False

    def _on_set_parameters(self, params):
        for p in params:
            self.get_logger().info(f"Parameter update: {p.name} = {p.value}")
        # Accept: range validation is enforced separately by each param's descriptor.
        return SetParametersResult(successful=True)

    def pack_nav2_obstacle_msg(self, detection):
        import uuid
        shape_type, centroid, dims, polygon = detection
        obstacle = Obstacle()
        obstacle.uuid.uuid = list(uuid.uuid4().bytes)
        obstacle.score = 1.0
        
        obstacle.position.x = float(centroid[0])
        obstacle.position.y = float(centroid[1])
        obstacle.position.z = 0.0
        
        if shape_type == 0:  # CIRCLE
            obstacle.size.x = float(2 * dims[0])
            obstacle.size.y = float(2 * dims[0])
            obstacle.size.z = 0.5
        elif shape_type == 1:  # BOX
            obstacle.size.x = float(dims[0])
            obstacle.size.y = float(dims[1])
            obstacle.size.z = 0.5
        else:  # LINE/CORNER
            pts = np.array(polygon)
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            obstacle.size.x = max(0.1, float(max_pt[0] - min_pt[0]))
            obstacle.size.y = max(0.1, float(max_pt[1] - min_pt[1]))
            obstacle.size.z = 0.5
            
        return obstacle

    def _lookup_tracking_transform(self, tracking_frame, source_frame, stamp, current_stamp):
        """Look up tracking_frame<-source_frame at the exact scan stamp.

        On failure (laggy/gappy TF -> extrapolation errors), optionally retry with the
        latest-available transform (rclpy.time.Time()), accepting it only if it is no more
        stale than `tf_latest_max_delay` seconds. Returns the TransformStamped to use, or
        None (caller then falls back to the sensor frame). Warnings are throttled.
        """
        try:
            return self.tf_buffer.lookup_transform(
                tracking_frame,
                source_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
        except Exception as e_exact:
            if not self.get_parameter('tf_fallback_to_latest').value:
                self.get_logger().warning(
                    f"TF lookup failed from '{tracking_frame}' to '{source_frame}': "
                    f"{e_exact}. Falling back to sensor frame.",
                    throttle_duration_sec=5.0
                )
                return None

        # Exact-stamp lookup failed; try the latest-available transform.
        try:
            latest = self.tf_buffer.lookup_transform(
                tracking_frame, source_frame, rclpy.time.Time()
            )
        except Exception as e_latest:
            self.get_logger().warning(
                f"TF lookup failed from '{tracking_frame}' to '{source_frame}' "
                f"(exact and latest): {e_latest}. Falling back to sensor frame.",
                throttle_duration_sec=5.0
            )
            return None

        tf_time = rclpy.time.Time.from_msg(latest.header.stamp)
        staleness = abs(current_stamp.nanoseconds - tf_time.nanoseconds) / 1e9
        max_delay = self.get_parameter('tf_latest_max_delay').value
        if staleness <= max_delay:
            self.get_logger().warning(
                f"Using latest-available TF '{tracking_frame}'<-'{source_frame}', "
                f"stale by {staleness:.2f}s (exact-stamp lookup failed).",
                throttle_duration_sec=5.0
            )
            return latest

        self.get_logger().warning(
            f"Latest TF '{tracking_frame}'<-'{source_frame}' is {staleness:.2f}s stale "
            f"(> {max_delay:.2f}s); falling back to sensor frame.",
            throttle_duration_sec=5.0
        )
        return None

    def scan_callback(self, msg: LaserScan):
        # Calculate dynamic dt
        dt = self.core.dt
        current_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if self.last_stamp is not None:
            calc_dt = (current_stamp.nanoseconds - self.last_stamp.nanoseconds) / 1e9
            if calc_dt > 0.001:
                dt = calc_dt
        self.last_stamp = current_stamp
        
        # Clamp dt to sane range [1e-3, 1.0]
        dt = max(1e-3, min(dt, 1.0))
        
        # Look up transform for ego-motion-aware tracking
        sensor_pose = None
        tracking_frame = self.get_parameter('tracking_frame').value
        out_frame_id = msg.header.frame_id
        
        if tracking_frame:
            trans = self._lookup_tracking_transform(
                tracking_frame, msg.header.frame_id, msg.header.stamp, current_stamp
            )
            if trans is not None:
                tx = trans.transform.translation.x
                ty = trans.transform.translation.y

                # Quaternion to yaw
                qx = trans.transform.rotation.x
                qy = trans.transform.rotation.y
                qz = trans.transform.rotation.z
                qw = trans.transform.rotation.w

                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                sensor_pose = (tx, ty, yaw)
                out_frame_id = tracking_frame

        # Update core parameters dynamically (if updated via dynamic reconfigure / parameters)
        self.core.min_range = self.get_parameter('min_range').value
        self.core.max_range = self.get_parameter('max_range').value
        self.core.beta = math.radians(self.get_parameter('beta_incidence_deg').value)
        self.core.sigma_r = self.get_parameter('sigma_r').value
        self.core.min_jump_distance = self.get_parameter('min_jump_distance').value
        self.core.max_jump_distance = self.get_parameter('max_jump_distance').value
        self.core.min_cluster_points = self.get_parameter('min_cluster_points').value
        self.core.max_cluster_points = self.get_parameter('max_cluster_points').value
        self.core.use_convex_hull = self.get_parameter('use_convex_hull').value
        self.core.split_threshold = self.get_parameter('split_threshold').value
        self.core.max_association_dist = self.get_parameter('max_association_distance').value
        self.core.min_track_age = self.get_parameter('min_track_age').value
        self.core.max_missed_frames = self.get_parameter('max_missed_frames').value
        self.core.use_median_filter = self.get_parameter('use_median_filter').value
        self.core.association_method = self.get_parameter('association_method').value
        self.core.circle_residual_ratio = self.get_parameter('circle_residual_ratio').value
        self.core.max_circle_radius = self.get_parameter('max_circle_radius').value
        self.core.corner_angle_min_deg = self.get_parameter('corner_angle_min_deg').value
        self.core.corner_angle_max_deg = self.get_parameter('corner_angle_max_deg').value
        self.core.shape_smoothing_alpha = self.get_parameter('shape_smoothing_alpha').value
        self.core.kf_process_noise = self.get_parameter('kf_process_noise').value
        self.core.shape_type_hysteresis = self.get_parameter('shape_type_hysteresis').value

        enable_tracking = self.get_parameter('enable_tracking').value
        publish_object_array = self.get_parameter('publish_object_array').value
        publish_obstacle_array = self.get_parameter('publish_obstacle_array').value

        # Process scan
        active_tracks, detections, clusters = self.core.process(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            dt=dt,
            sensor_pose=sensor_pose,
            enable_tracking=enable_tracking
        )
        
        out_header = Header(stamp=msg.header.stamp, frame_id=out_frame_id)
        
        # Select which tracks to visualize/publish. Defined unconditionally so the
        # debug-marker path is safe even when tracking is disabled (active_tracks is
        # empty in that case, so no markers are produced).
        if self.get_parameter('publish_unconfirmed').value:
            tracks_to_publish = active_tracks
        else:
            tracks_to_publish = [t for t in active_tracks if t.is_confirmed]

        # 1. Publish Obstacles (Tracks)
        if publish_object_array and enable_tracking:
            obj_array = ObjectArray()
            obj_array.header = out_header
            
            for track in tracks_to_publish:
                obj = Object()
                obj.header = out_header
                obj.id = track.id
                obj.detection_level = Object.OBJECT_TRACKED if track.is_confirmed else Object.OBJECT_DETECTED
                obj.object_classified = False
                
                # Position (X, Y from state vector, Z=0)
                obj.pose.position = Point(x=track.x[0], y=track.x[1], z=0.0)
                
                # Orientation
                if track.shape_type == 1: # BOX
                    yaw = track.shape_dims[2]
                    cy = math.cos(yaw * 0.5)
                    sy = math.sin(yaw * 0.5)
                    obj.pose.orientation = Quaternion(x=0.0, y=0.0, z=sy, w=cy)
                else:
                    obj.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    
                # Velocity
                obj.twist.linear = Vector3(x=track.x[2], y=track.x[3], z=0.0)
                obj.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
                
                # Shape
                if track.shape_type == 0: # CIRCLE
                    obj.shape.type = SolidPrimitive.CYLINDER
                    obj.shape.dimensions = [0.5, track.shape_dims[0]]
                elif track.shape_type == 1: # BOX
                    obj.shape.type = SolidPrimitive.BOX
                    obj.shape.dimensions = [track.shape_dims[0], track.shape_dims[1], 0.5]
                    
                # Polygon
                poly = Polygon()
                for pt in track.polygon:
                    poly.points.append(Point32(x=float(pt[0]), y=float(pt[1]), z=0.0))
                obj.polygon = poly
                
                obj_array.objects.append(obj)
                
            self.obstacles_pub.publish(obj_array)

        # 1.5. Publish ObstacleArray (Detections)
        if publish_obstacle_array:
            if NAV2_DYNAMIC_MSGS_AVAILABLE:
                if not hasattr(self, 'detections_pub'):
                    self.detections_pub = self.create_publisher(ObstacleArray, 'detections', 10)
                
                obstacle_array = ObstacleArray()
                obstacle_array.header = out_header
                for det in detections:
                    obstacle = self.pack_nav2_obstacle_msg(det)
                    obstacle_array.obstacles.append(obstacle)
                self.detections_pub.publish(obstacle_array)
            else:
                if not getattr(self, '_nav2_warned', False):
                    self.get_logger().warning("nav2_dynamic_msgs not available; nav2 ObstacleArray output disabled")
                    self._nav2_warned = True
        
        # 2. Publish Debug PointCloud (debug pointcloud stays in the sensor frame)
        if self.publish_pc and len(clusters) > 0:
            pc_msg = self.make_color_pc2(msg.header, clusters)
            self.pc_pub.publish(pc_msg)
            
        # 3. Publish Debug Markers. With tracking on, markers come from tracks; with tracking
        # off there are no tracks, so render the raw per-frame detections instead.
        if self.publish_markers:
            if enable_tracking:
                self.publish_debug_markers(out_header, tracks_to_publish)
            else:
                self.publish_debug_markers_detections(out_header, detections)

    def make_color_pc2(self, header, clusters):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        data = []
        # Predefined distinct colors (8-bit RGB packed into 32-bit UINT)
        colors = [
            0xFFFF0000, # Red
            0xFF00FF00, # Green
            0xFF0000FF, # Blue
            0xFFFF00FF, # Magenta
            0xFF00FFFF, # Cyan
            0xFFFFFF00, # Yellow
            0xFFFF8000, # Orange
            0xFF8000FF, # Purple
            0xFF00FF80, # Lime
            0xFFFF0080, # Deep Pink
            0xFF80FF00, # Yellow-Green
            0xFF0080FF  # Sky Blue
        ]
        
        for c_idx, cluster in enumerate(clusters):
            rgb = colors[c_idx % len(colors)]
            for pt in cluster:
                data.append(struct.pack('fffI', float(pt[0]), float(pt[1]), 0.0, rgb))
                
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(data)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = b''.join(data)
        
        return msg

    def publish_debug_markers(self, header, tracks):
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # ID offsets
        SHAPE_ID = 1000
        VEL_ID = 2000
        TEXT_ID = 3000
        BORDER_ID = 4000
        
        for track in tracks:
            # Color based on track ID
            r = (13 * track.id) % 256 / 255.0
            g = (57 * track.id) % 256 / 255.0
            b = (127 * track.id) % 256 / 255.0
            color = ColorRGBA(r=r, g=g, b=b, a=0.7)
            
            # --- 1. Shape Marker (Cylinder / Box / Polyline Border) ---
            shape_marker = Marker()
            shape_marker.header = header
            shape_marker.ns = "shapes"
            shape_marker.id = SHAPE_ID + track.id
            shape_marker.color = color
            shape_marker.pose.position = Point(x=track.x[0], y=track.x[1], z=0.0)
            
            if track.shape_type == 0: # CIRCLE
                shape_marker.type = Marker.CYLINDER
                shape_marker.scale = Vector3(x=track.shape_dims[0]*2.0, y=track.shape_dims[0]*2.0, z=0.2)
                shape_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                shape_marker.action = Marker.ADD
                marker_array.markers.append(shape_marker)
                
            elif track.shape_type == 1: # BOX
                shape_marker.type = Marker.CUBE
                shape_marker.scale = Vector3(x=track.shape_dims[0], y=track.shape_dims[1], z=0.2)
                yaw = track.shape_dims[2]
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                shape_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=sy, w=cy)
                shape_marker.action = Marker.ADD
                marker_array.markers.append(shape_marker)
                
            # For Line (2) or Corner (3), we visualize using a LINE_STRIP border marker
            
            # --- 2. Outline / Border Marker (Polygon) ---
            border_marker = Marker()
            border_marker.header = header
            border_marker.ns = "outlines"
            border_marker.id = BORDER_ID + track.id
            border_marker.type = Marker.LINE_STRIP
            border_marker.scale.x = 0.03 # line width
            border_marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            border_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            border_marker.action = Marker.ADD

            # For primitive shapes, regenerate the outline from the tracked
            # (KF-smoothed) center + smoothed dims so it coincides with the filled
            # shape marker (the raw per-frame polygon otherwise wobbles off-center).
            # For line/corner/hull, use the raw detection polygon.
            if track.shape_type == 0:  # CIRCLE: ring at smoothed radius around track center
                cx, cy, rad = track.x[0], track.x[1], track.shape_dims[0]
                for k in range(17):
                    a = 2.0 * math.pi * (k % 16) / 16.0
                    border_marker.points.append(Point(x=cx + rad*math.cos(a), y=cy + rad*math.sin(a), z=0.0))
            elif track.shape_type == 1:  # BOX: rectangle from smoothed dims/yaw
                cx, cy = track.x[0], track.x[1]
                hl, hw, yaw = track.shape_dims[0]*0.5, track.shape_dims[1]*0.5, track.shape_dims[2]
                c, s = math.cos(yaw), math.sin(yaw)
                for lx, ly in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]:
                    border_marker.points.append(Point(x=cx + lx*c - ly*s, y=cy + lx*s + ly*c, z=0.0))
            else:  # LINE / CORNER: raw detection polygon
                for pt in track.polygon:
                    border_marker.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))

            if len(border_marker.points) > 1:
                marker_array.markers.append(border_marker)
                
            # --- 3. Velocity Vector Arrow ---
            vel_norm = math.sqrt(track.x[2]**2 + track.x[3]**2)
            if vel_norm > 0.1: # Only draw if velocity is non-negligible
                vel_marker = Marker()
                vel_marker.header = header
                vel_marker.ns = "velocity"
                vel_marker.id = VEL_ID + track.id
                vel_marker.type = Marker.ARROW
                vel_marker.scale = Vector3(x=0.04, y=0.08, z=0.1) # shaft, head, height
                vel_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
                vel_marker.action = Marker.ADD
                
                # Arrow points from centroid to centroid + velocity
                p_start = Point(x=track.x[0], y=track.x[1], z=0.0)
                p_end = Point(x=track.x[0] + track.x[2], y=track.x[1] + track.x[3], z=0.0)
                vel_marker.points = [p_start, p_end]
                
                marker_array.markers.append(vel_marker)
                
            # --- 4. Text Marker (ID & Velocity) ---
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "labels"
            text_marker.id = TEXT_ID + track.id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.scale.z = 0.2 # font size
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.pose.position = Point(x=track.x[0], y=track.x[1], z=0.3) # float text above shape
            text_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            text_marker.action = Marker.ADD
            
            # Label content (compact; velocity only when moving)
            text_marker.text = f"ID:{track.id} {vel_norm:.1f}" if vel_norm > 0.1 else f"ID:{track.id}"
            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

    def publish_debug_markers_detections(self, header, detections):
        """Render per-frame detections (used when enable_tracking=False, so there are no
        tracks). Detections carry no id/velocity: color/id are synthesized from the list
        index and no velocity arrows are drawn. Circle/box outlines are regenerated from
        centroid+dims (matching the tracked-marker path); line/corner use the raw polygon.
        Keep in sync with publish_debug_markers() and the C++ node's equivalent.
        """
        marker_array = MarkerArray()

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        SHAPE_ID = 1000
        BORDER_ID = 4000

        for i, det in enumerate(detections):
            shape_type, centroid, dims, polygon = det
            cx, cy = float(centroid[0]), float(centroid[1])

            # Color based on detection index (no track id available)
            r = (13 * i) % 256 / 255.0
            g = (57 * i) % 256 / 255.0
            b = (127 * i) % 256 / 255.0
            color = ColorRGBA(r=r, g=g, b=b, a=0.7)

            # --- 1. Shape Marker (Cylinder / Box) ---
            shape_marker = Marker()
            shape_marker.header = header
            shape_marker.ns = "shapes"
            shape_marker.id = SHAPE_ID + i
            shape_marker.color = color
            shape_marker.pose.position = Point(x=cx, y=cy, z=0.0)

            if shape_type == 0:  # CIRCLE
                shape_marker.type = Marker.CYLINDER
                shape_marker.scale = Vector3(x=dims[0]*2.0, y=dims[0]*2.0, z=0.2)
                shape_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                shape_marker.action = Marker.ADD
                marker_array.markers.append(shape_marker)
            elif shape_type == 1:  # BOX
                shape_marker.type = Marker.CUBE
                shape_marker.scale = Vector3(x=dims[0], y=dims[1], z=0.2)
                yaw = dims[2]
                cyq = math.cos(yaw * 0.5)
                syq = math.sin(yaw * 0.5)
                shape_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=syq, w=cyq)
                shape_marker.action = Marker.ADD
                marker_array.markers.append(shape_marker)

            # --- 2. Outline / Border Marker (Polygon) ---
            border_marker = Marker()
            border_marker.header = header
            border_marker.ns = "outlines"
            border_marker.id = BORDER_ID + i
            border_marker.type = Marker.LINE_STRIP
            border_marker.scale.x = 0.03
            border_marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            border_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            border_marker.action = Marker.ADD

            if shape_type == 0:  # CIRCLE: ring at radius around centroid
                rad = dims[0]
                for k in range(17):
                    a = 2.0 * math.pi * (k % 16) / 16.0
                    border_marker.points.append(Point(x=cx + rad*math.cos(a), y=cy + rad*math.sin(a), z=0.0))
            elif shape_type == 1:  # BOX: rectangle from dims/yaw
                hl, hw, yaw = dims[0]*0.5, dims[1]*0.5, dims[2]
                c, s = math.cos(yaw), math.sin(yaw)
                for lx, ly in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]:
                    border_marker.points.append(Point(x=cx + lx*c - ly*s, y=cy + lx*s + ly*c, z=0.0))
            else:  # LINE / CORNER: raw detection polygon
                for pt in polygon:
                    border_marker.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))

            if len(border_marker.points) > 1:
                marker_array.markers.append(border_marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = LaserObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
