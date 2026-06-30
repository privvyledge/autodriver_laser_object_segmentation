import unittest
import numpy as np
import math
import sys
import os

# Include package folder in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodriver_laser_object_segmentation.laser_obstacle_detector_core import LaserObstacleDetectorCore

class TestLaserObstacleDetectorCore(unittest.TestCase):
    def setUp(self):
        # Instantiate detector with typical values
        self.detector = LaserObstacleDetectorCore(
            min_range=0.1,
            max_range=15.0,
            beta_incidence_rad=math.radians(10.0),
            sigma_r=0.01,
            min_jump_distance=0.15,
            max_jump_distance=1.0,
            min_cluster_points=3,
            max_cluster_points=200,
            use_convex_hull=False,
            split_threshold=0.04,
            max_association_distance=1.0,
            min_track_age=1, # 1 for immediate tracking promotion in tests
            max_missed_frames=3,
            dt=0.1
        )

    def test_circle_fitting(self):
        # Generate 20 points along a circle of center [3.0, 4.0] and radius 0.5
        center = np.array([3.0, 4.0])
        r = 0.5
        angles = np.linspace(0, 2 * math.pi, 20)
        pts = np.column_stack((
            center[0] + r * np.cos(angles),
            center[1] + r * np.sin(angles)
        ))
        
        # Test core kasa fit
        fit_c, fit_r = self.detector.fit_circle_kasa(pts)
        self.assertAlmostEqual(fit_c[0], center[0], places=3)
        self.assertAlmostEqual(fit_c[1], center[1], places=3)
        self.assertAlmostEqual(fit_r, r, places=3)
        
        # Test shape classifier
        shape_type, shape_center, shape_dims, poly = self.detector.fit_shape(pts)
        self.assertEqual(shape_type, 0) # CIRCLE
        self.assertAlmostEqual(shape_dims[0], r, places=2)

    def test_obb_fitting(self):
        # Generate points of an oriented box centered at [2.0, 1.0], dims [1.0, 0.4], yaw 30 degrees (0.523 rad)
        center = np.array([2.0, 1.0])
        length, width = 1.0, 0.4
        yaw = math.radians(30.0)
        
        # Box corners in local frame
        hl, hw = length / 2.0, width / 2.0
        local_pts = np.array([
            [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw],
            [0.0, -hw], [0.0, hw], [-hl, 0.0], [hl, 0.0]
        ])
        
        # Rotate and translate
        R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])
        pts = np.dot(local_pts, R.T) + center
        
        # Fit OBB
        fit_c, fit_size, fit_yaw = self.detector.fit_obb(pts)
        self.assertAlmostEqual(fit_c[0], center[0], places=2)
        self.assertAlmostEqual(fit_c[1], center[1], places=2)
        self.assertAlmostEqual(fit_size[0], length, places=2)
        self.assertAlmostEqual(fit_size[1], width, places=2)
        self.assertAlmostEqual(fit_yaw, yaw, places=2)

    def test_line_fitting(self):
        # Straight wall: points from [1.0, 2.0] to [4.0, 2.0]
        pts = np.column_stack((
            np.linspace(1.0, 4.0, 10),
            np.ones(10) * 2.0
        ))
        
        shape_type, shape_center, shape_dims, poly = self.detector.fit_shape(pts)
        self.assertEqual(shape_type, 2) # LINE
        # Polygon should contain 2 vertices (start & end)
        self.assertEqual(len(poly), 2)
        self.assertAlmostEqual(poly[0][0], 1.0, places=3)
        self.assertAlmostEqual(poly[0][1], 2.0, places=3)
        self.assertAlmostEqual(poly[1][0], 4.0, places=3)
        self.assertAlmostEqual(poly[1][1], 2.0, places=3)

    def test_corner_fitting(self):
        # L-shape wall meeting at [2.0, 2.0]
        # Seg 1: [1.0, 2.0] to [2.0, 2.0]
        # Seg 2: [2.0, 2.0] to [2.0, 3.0]
        pts = np.array([
            [1.0, 2.0], [1.5, 2.0], [2.0, 2.0], [2.0, 2.5], [2.0, 3.0]
        ])
        
        shape_type, shape_center, shape_dims, poly = self.detector.fit_shape(pts)
        self.assertEqual(shape_type, 3) # CORNER
        self.assertEqual(len(poly), 3) # 3 vertices: start, corner/vertex, end
        # The corner/vertex should be [2.0, 2.0]
        self.assertAlmostEqual(poly[1][0], 2.0, places=3)
        self.assertAlmostEqual(poly[1][1], 2.0, places=3)

    def test_clustering_and_pipeline(self):
        # Create a synthetic scan representing two distinct obstacles:
        # Obstacle 1: A cylinder at range 3.0, angle 0
        # Obstacle 2: A cylinder at range 5.0, angle 45 degrees
        # Assume angular increment of 1 degree (0.01745 rad)
        angle_inc = math.radians(1.0)
        ranges = [20.0] * 360 # Initialize out of range (max_range=15.0)
        
        # Populate Obstacle 1 (spread of 5 beams around 0 degrees)
        # ranges = dist / cos(angle) - approximate points
        for i in [-2, -1, 0, 1, 2]:
            idx = i % 360
            ranges[idx] = 3.0
            
        # Populate Obstacle 2 (spread of 5 beams around 45 degrees)
        for i in [43, 44, 45, 46, 47]:
            ranges[i] = 5.0
            
        # Run process
        confirmed, clusters = self.detector.process(ranges, -math.pi, angle_inc)
        
        # Check that we found 2 clusters
        self.assertEqual(len(clusters), 2)
        
        # Check that they correspond to the correct coordinates
        c1 = np.mean(clusters[0], axis=0)
        c2 = np.mean(clusters[1], axis=0)
        
        # Sort by distance
        if np.linalg.norm(c1) > np.linalg.norm(c2):
            c1, c2 = c2, c1
            
        # Centroid of c1 should be close to [3.0, 0] in some rotated coordinate frame
        # (since angle_min is -pi, beam 0 is at -pi rad. Wait, let's verify angle map:
        # idx 0: -pi, idx 180: 0, idx 45 is at -pi + 45deg)
        # Regardless of frame orientation, the ranges should be 3.0 and 5.0
        self.assertAlmostEqual(np.linalg.norm(c1), 3.0, delta=0.1)
        self.assertAlmostEqual(np.linalg.norm(c2), 5.0, delta=0.1)

    def test_tracking_kf(self):
        # Test tracking over multiple frames with a moving obstacle
        # Centroid moving from [2.0, 0.0] to [2.9, 0.0] in 10 steps (dt = 0.1, speed = 1.0 m/s)
        dt = 0.1
        
        # Frame 1
        det1 = [(0, np.array([2.0, 0.0]), [0.2], [])]
        self.detector.associate_and_track(det1)
        self.assertEqual(len(self.detector.tracks), 1)
        track = self.detector.tracks[0]
        self.assertAlmostEqual(track.x[0], 2.0, places=2)
        
        # Frame 2
        det2 = [(0, np.array([2.1, 0.0]), [0.2], [])]
        self.detector.associate_and_track(det2)
        track = self.detector.tracks[0]
        # Position should update towards 2.1, and velocity vx should start to become positive
        self.assertTrue(track.x[2] > 0.0)
        
        # Frames 3 to 10
        for x_pos in [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]:
            self.detector.associate_and_track([(0, np.array([x_pos, 0.0]), [0.2], [])])
            
        track = self.detector.tracks[0]
        # Velocity vx should be close to 1.0 m/s after 10 updates
        self.assertAlmostEqual(track.x[2], 1.0, delta=0.2)

if __name__ == '__main__':
    unittest.main()
