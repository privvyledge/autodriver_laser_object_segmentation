import unittest
import numpy as np
import math
import sys
import os
import ctypes
import scipy.optimize
import scipy.spatial

# Include package folder in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodriver_laser_object_segmentation.laser_obstacle_detector_core import LaserObstacleDetectorCore

def load_cpp_lib():
    paths = [
        os.path.expanduser('~/ros2_ws/install/autodriver_laser_object_segmentation/lib/liblaser_obstacle_detector_core_lib.so'),
        '/home/privvyledge/ros2_ws/install/autodriver_laser_object_segmentation/lib/liblaser_obstacle_detector_core_lib.so'
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ctypes.CDLL(p)
            except Exception:
                pass
    return None

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
        confirmed, detections, clusters = self.detector.process(ranges, -math.pi, angle_inc)
        
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

    def test_wall_survival(self):
        # 200 points along a straight line: [1.0, 2.0] to [5.0, 2.0]
        pts = np.column_stack((
            np.linspace(1.0, 5.0, 200),
            np.ones(200) * 2.0
        ))
        shape_type, shape_center, shape_dims, poly = self.detector.fit_shape(pts)
        self.assertEqual(shape_type, 2) # LINE
        self.assertEqual(len(poly), 2)

    def test_vectorized_clustering_parity(self):
        def reference_cluster_points(detector, points, valid_indices, angle_increment):
            N = len(points)
            if N == 0:
                return []
            clusters = []
            current_cluster = [0]
            for i in range(1, N):
                idx_prev = valid_indices[i-1]
                idx_curr = valid_indices[i]
                if idx_curr - idx_prev > 2:
                    if len(current_cluster) >= detector.min_cluster_points:
                        clusters.append(current_cluster)
                    current_cluster = [i]
                    continue
                p_prev = points[i-1]
                p_curr = points[i]
                dist = np.linalg.norm(p_curr - p_prev)
                r_prev = np.linalg.norm(p_prev)
                d_theta = angle_increment * (idx_curr - idx_prev)
                denom = math.sin(detector.beta - d_theta)
                if denom > 0.01:
                    d_th = r_prev * (math.sin(d_theta) / denom) + 3.0 * detector.sigma_r
                else:
                    d_th = detector.min_jump_distance
                d_th = np.clip(d_th, detector.min_jump_distance, detector.max_jump_distance)
                if dist > d_th:
                    if len(current_cluster) >= detector.min_cluster_points:
                        clusters.append(current_cluster)
                    current_cluster = [i]
                else:
                    current_cluster.append(i)
            if len(current_cluster) >= detector.min_cluster_points:
                clusters.append(current_cluster)
            
            filtered = []
            for c in clusters:
                if detector.min_cluster_points <= len(c) <= detector.max_cluster_points:
                    filtered.append(points[c])
            return filtered

        angles = np.linspace(-np.pi, np.pi, 360)
        ranges = np.sin(angles) * 3.0 + 4.0
        points, valid_indices = self.detector.preprocess_scan(ranges, -np.pi, 2*np.pi/360)
        
        ref_clusters = reference_cluster_points(self.detector, points, valid_indices, 2*np.pi/360)
        vec_clusters = self.detector.cluster_points(points, valid_indices, 2*np.pi/360)
        
        self.assertEqual(len(ref_clusters), len(vec_clusters))
        for c_ref, c_vec in zip(ref_clusters, vec_clusters):
            self.assertTrue(np.allclose(c_ref, c_vec))

    def test_hull_parity_and_robustness(self):
        pts_collinear = np.column_stack((
            np.linspace(1.0, 5.0, 10),
            np.ones(10) * 2.0
        ))
        hull = self.detector.convex_hull_jarvis(pts_collinear)
        self.assertTrue(len(hull) <= 2)
        
        np.random.seed(42)
        pts_rand = np.random.rand(20, 2)
        hull_rand = self.detector.convex_hull_jarvis(pts_rand)
        
        for i in range(len(hull_rand)):
            p1 = hull_rand[i]
            p2 = hull_rand[(i + 1) % len(hull_rand)]
            p3 = hull_rand[(i + 2) % len(hull_rand)]
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            self.assertTrue(cross >= -1e-9)
            
        lib = load_cpp_lib()
        if lib is not None:
            lib.test_convex_hull.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int)
            ]
            
            px = (ctypes.c_double * 20)(*pts_rand[:, 0])
            py = (ctypes.c_double * 20)(*pts_rand[:, 1])
            hx = (ctypes.c_double * 20)()
            hy = (ctypes.c_double * 20)()
            hn = ctypes.c_int(0)
            
            lib.test_convex_hull(px, py, 20, hx, hy, ctypes.byref(hn))
            cpp_hull = np.column_stack((list(hx)[:hn.value], list(hy)[:hn.value]))
            
            self.assertEqual(hn.value, len(hull_rand))
            for p in hull_rand:
                dists = np.linalg.norm(cpp_hull - p, axis=1)
                self.assertTrue(np.any(dists < 1e-5))

    def test_obb_accuracy(self):
        center = np.array([3.0, 2.0])
        length, width = 2.0, 0.8
        yaw = math.pi / 4.0
        
        hl, hw = length / 2.0, width / 2.0
        local_pts = np.array([
            [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw],
            [-hl/2, -hw], [hl/2, -hw], [hl/2, hw], [-hl/2, hw]
        ])
        R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])
        pts = np.dot(local_pts, R.T) + center
        
        fit_c, fit_size, fit_yaw = self.detector.fit_obb(pts)
        self.assertAlmostEqual(fit_c[0], center[0], places=2)
        self.assertAlmostEqual(fit_c[1], center[1], places=2)
        self.assertAlmostEqual(fit_size[0], length, places=2)
        self.assertAlmostEqual(fit_size[1], width, places=2)
        self.assertAlmostEqual(fit_yaw, yaw, delta=math.radians(1.0))
        
        lib = load_cpp_lib()
        if lib is not None:
            lib.test_obb.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            px = (ctypes.c_double * len(pts))(*pts[:, 0])
            py = (ctypes.c_double * len(pts))(*pts[:, 1])
            cx, cy, clen, cwid, cyaw = ctypes.c_double(0), ctypes.c_double(0), ctypes.c_double(0), ctypes.c_double(0), ctypes.c_double(0)
            
            lib.test_obb(px, py, len(pts), ctypes.byref(cx), ctypes.byref(cy), ctypes.byref(clen), ctypes.byref(cwid), ctypes.byref(cyaw))
            
            self.assertAlmostEqual(cx.value, fit_c[0], places=5)
            self.assertAlmostEqual(cy.value, fit_c[1], places=5)
            self.assertAlmostEqual(clen.value, fit_size[0], places=5)
            self.assertAlmostEqual(cwid.value, fit_size[1], places=5)
            self.assertAlmostEqual(cyaw.value, fit_yaw, places=5)

    def test_ego_motion(self):
        wall_pts_sensor_frame_1 = np.column_stack((
            np.linspace(1.0, 3.0, 10),
            np.ones(10) * 2.0
        ))
        
        tx, ty, yaw = 0.1, 0.0, math.radians(5.0)
        R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])
        wall_pts_tracking = np.column_stack((
            np.linspace(1.0, 3.0, 10),
            np.ones(10) * 2.0
        ))
        wall_pts_sensor_frame_2 = np.dot(wall_pts_tracking - np.array([tx, ty]), R)
        
        self.detector.tracks = []
        
        orig_preprocess = self.detector.preprocess_scan
        orig_cluster = self.detector.cluster_points
        
        try:
            self.detector.preprocess_scan = lambda r, a_min, a_inc: (wall_pts_sensor_frame_1, np.arange(len(wall_pts_sensor_frame_1)))
            self.detector.cluster_points = lambda p, v_idx, a_inc: [p]
            tracks1, _, _ = self.detector.process([1.0]*10, 0.0, 0.1, dt=0.1, sensor_pose=(0.0, 0.0, 0.0))
            
            self.detector.preprocess_scan = lambda r, a_min, a_inc: (wall_pts_sensor_frame_2, np.arange(len(wall_pts_sensor_frame_2)))
            self.detector.cluster_points = lambda p, v_idx, a_inc: [p]
            tracks2, _, _ = self.detector.process([1.0]*10, 0.0, 0.1, dt=0.1, sensor_pose=(tx, ty, yaw))
            
            self.assertEqual(len(tracks2), 1)
            vx, vy = tracks2[0].x[2], tracks2[0].x[3]
            self.assertAlmostEqual(vx, 0.0, delta=0.05)
            self.assertAlmostEqual(vy, 0.0, delta=0.05)
        finally:
            self.detector.preprocess_scan = orig_preprocess
            self.detector.cluster_points = orig_cluster

    def test_dt_scaling(self):
        self.detector.tracks = []
        self.detector.min_track_age = 1
        
        det1 = [(0, np.array([2.0, 0.0]), [0.2], [])]
        self.detector.associate_and_track(det1, dt=0.1)
        
        det2 = [(0, np.array([2.1, 0.0]), [0.2], [])]
        self.detector.associate_and_track(det2, dt=0.1)
        v1 = self.detector.tracks[0].x[2]
        
        self.detector.tracks = []
        self.detector.associate_and_track(det1, dt=0.05)
        self.detector.associate_and_track(det2, dt=0.05)
        v2 = self.detector.tracks[0].x[2]
        
        self.assertAlmostEqual(v2, v1 * 2.0, delta=0.2)

    def test_association_method(self):
        lib = load_cpp_lib()
        if lib is not None:
            lib.test_hungarian.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)
            ]
            cost = np.array([
                [1.0, 2.0, 0.5],
                [0.2, 1.5, 3.0],
                [2.0, 0.1, 1.0]
            ], dtype=np.float64)
            
            row_py, col_py = scipy.optimize.linear_sum_assignment(cost)
            
            flat_cost = cost.flatten()
            c_cost = (ctypes.c_double * 9)(*flat_cost)
            row_cpp = (ctypes.c_int * 3)()
            col_cpp = (ctypes.c_int * 3)()
            count = ctypes.c_int(0)
            
            lib.test_hungarian(c_cost, 3, 3, row_cpp, col_cpp, ctypes.byref(count))
            
            self.assertEqual(count.value, 3)
            match_py = {r: c for r, c in zip(row_py, col_py)}
            for i in range(3):
                self.assertEqual(col_cpp[i], match_py[row_cpp[i]])

    def test_detection_level(self):
        self.detector.tracks = []
        self.detector.min_track_age = 3
        
        self.detector.process([3.0]*10, 0.0, 0.1, dt=0.1)
        self.assertEqual(len(self.detector.tracks), 1)
        self.assertFalse(self.detector.tracks[0].is_confirmed)
        
        self.detector.process([3.0]*10, 0.0, 0.1, dt=0.1)
        self.assertFalse(self.detector.tracks[0].is_confirmed)
        
        self.detector.process([3.0]*10, 0.0, 0.1, dt=0.1)
        self.assertTrue(self.detector.tracks[0].is_confirmed)

    def test_shape_type_hysteresis(self):
        from autodriver_laser_object_segmentation.laser_obstacle_detector_core import Track
        # hysteresis=3: a differing type must persist 3 consecutive frames to switch.
        tr = Track(1, [0.0, 0.0], 0, [0.5], [], dt=0.1,
                   shape_smoothing_alpha=1.0, kf_process_noise=0.1, shape_type_hysteresis=3)
        # Two transient BOX frames must NOT flip the type away from CIRCLE.
        tr.update([0.0, 0.0], 1, [0.4, 0.3, 0.0], [])
        self.assertEqual(tr.shape_type, 0)
        tr.update([0.0, 0.0], 1, [0.4, 0.3, 0.0], [])
        self.assertEqual(tr.shape_type, 0)
        # Third consecutive BOX frame confirms the switch.
        tr.update([0.0, 0.0], 1, [0.4, 0.3, 0.0], [])
        self.assertEqual(tr.shape_type, 1)
        # A single stray CIRCLE frame is rejected (counter resets, stays BOX).
        tr.update([0.0, 0.0], 0, [0.5], [])
        self.assertEqual(tr.shape_type, 1)

    def test_hysteresis_disabled_by_default(self):
        from autodriver_laser_object_segmentation.laser_obstacle_detector_core import Track
        # Default hysteresis=1 => switch immediately (preserves legacy behavior).
        tr = Track(1, [0.0, 0.0], 0, [0.5], [], dt=0.1)
        tr.update([0.0, 0.0], 1, [0.4, 0.3, 0.0], [])
        self.assertEqual(tr.shape_type, 1)

if __name__ == '__main__':
    unittest.main()
