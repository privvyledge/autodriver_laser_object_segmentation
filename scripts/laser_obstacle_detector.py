#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from derived_object_msgs.msg import Object, ObjectArray
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Point, Vector3, Quaternion, Polygon, Point32
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import numpy as np
import struct
import math

from autodriver_laser_object_segmentation.laser_obstacle_detector_core import LaserObstacleDetectorCore

class LaserObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('laser_obstacle_detector')
        
        # Declare Parameters
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('beta_incidence_deg', 10.0)
        self.declare_parameter('sigma_r', 0.01)
        self.declare_parameter('min_jump_distance', 0.1)
        self.declare_parameter('max_jump_distance', 1.0)
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_cluster_points', 150)
        self.declare_parameter('use_convex_hull', True)
        self.declare_parameter('split_threshold', 0.05)
        self.declare_parameter('max_association_distance', 1.0)
        self.declare_parameter('min_track_age', 3)
        self.declare_parameter('max_missed_frames', 5)
        self.declare_parameter('publish_debug_pointcloud', True)
        self.declare_parameter('publish_debug_markers', True)
        
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
            dt=0.1 # initial guess, will update dynamically
        )
        
        self.last_stamp = None
        
        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
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
            
        self.get_logger().info('Laser Obstacle Detector Node initialized.')

    def scan_callback(self, msg: LaserScan):
        # Calculate dynamic dt
        current_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if self.last_stamp is not None:
            dt = (current_stamp.nanoseconds - self.last_stamp.nanoseconds) / 1e9
            if dt > 0.001:
                self.core.dt = dt
        self.last_stamp = current_stamp
        
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
        
        # Process scan
        confirmed_tracks, clusters = self.core.process(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment
        )
        
        # 1. Publish Obstacles
        obj_array = ObjectArray()
        obj_array.header = msg.header
        
        for track in confirmed_tracks:
            obj = Object()
            obj.header = msg.header
            obj.id = track.id
            obj.detection_level = Object.TRACKED
            obj.object_classified = False
            
            # Position (X, Y from state vector, Z=0)
            obj.pose.position = Point(x=track.x[0], y=track.x[1], z=0.0)
            
            # Orientation
            if track.shape_type == 1: # BOX
                yaw = track.shape_dims[2]
                # Convert yaw to quaternion
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
                # Cylinder dimensions: height, radius
                obj.shape.dimensions = [0.5, track.shape_dims[0]]
            elif track.shape_type == 1: # BOX
                obj.shape.type = SolidPrimitive.BOX
                # Box dimensions: X, Y, Z
                obj.shape.dimensions = [track.shape_dims[0], track.shape_dims[1], 0.5]
            # LINE (2) and CORNER (3) do not have basic SolidPrimitives, we rely on polygon
            
            # Polygon
            poly = Polygon()
            for pt in track.polygon:
                poly.points.append(Point32(x=float(pt[0]), y=float(pt[1]), z=0.0))
            obj.polygon = poly
            
            obj_array.objects.append(obj)
            
        self.obstacles_pub.publish(obj_array)
        
        # 2. Publish Debug PointCloud
        if self.publish_pc and len(clusters) > 0:
            pc_msg = self.make_color_pc2(msg.header, clusters)
            self.pc_pub.publish(pc_msg)
            
        # 3. Publish Debug Markers
        if self.publish_markers:
            self.publish_debug_markers(msg.header, confirmed_tracks)

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
            
            for pt in track.polygon:
                border_marker.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
                
            # If circle, box or convex hull, close the polygon loop
            if track.shape_type in [0, 1] or self.core.use_convex_hull:
                if len(track.polygon) > 0:
                    border_marker.points.append(Point(x=float(track.polygon[0][0]), y=float(track.polygon[0][1]), z=0.0))
                    
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
            
            # Label content
            text_marker.text = f"ID: {track.id} | V: {vel_norm:.1f} m/s"
            marker_array.markers.append(text_marker)
            
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
