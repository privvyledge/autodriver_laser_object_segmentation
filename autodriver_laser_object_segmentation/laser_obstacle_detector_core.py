import numpy as np
import math
import scipy.spatial
import scipy.signal
import scipy.optimize


class Track:
    def __init__(self, track_id, position, shape_type, shape_dims, polygon, dt, shape_smoothing_alpha=1.0):
        """
        Keeps track of an individual obstacle using a constant-velocity Kalman Filter.
        State vector x = [px, py, vx, vy]^T
        """
        self.id = track_id
        self.dt = dt
        
        # State vector: [px, py, vx, vy]
        self.x = np.array([position[0], position[1], 0.0, 0.0])
        
        # State covariance
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])
        
        # Process noise parameter
        self.q_proc = 0.5
        
        # Measurement noise covariance
        self.R = np.diag([0.02, 0.02])
        
        # Track properties
        self.shape_type = shape_type     # 0: CIRCLE, 1: BOX, 2: LINE, 3: CORNER
        self.shape_dims = shape_dims     # list: [radius] or [length, width, height]
        self.polygon = polygon           # List of [x, y] coordinates representing shape boundary
        
        self.shape_smoothing_alpha = shape_smoothing_alpha
        self.age = 1
        self.missed_frames = 0
        self.is_confirmed = False

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
            
        # State transition matrix F
        F = np.array([
            [1.0, 0.0,   dt,  0.0],
            [0.0, 1.0,  0.0,   dt],
            [0.0, 0.0,  1.0,  0.0],
            [0.0, 0.0,  0.0,  1.0]
        ])
        
        # Predict state
        self.x = np.dot(F, self.x)
        
        # Process noise covariance Q
        # Continuous white noise approximation
        Q = np.array([
            [dt**3 / 3.0,          0.0, dt**2 / 2.0,          0.0],
            [         0.0, dt**3 / 3.0,          0.0, dt**2 / 2.0],
            [dt**2 / 2.0,          0.0,          dt,          0.0],
            [         0.0, dt**2 / 2.0,          0.0,          dt]
        ]) * self.q_proc
        
        # Predict covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + Q

    def update(self, position, shape_type, shape_dims, polygon):
        # Measurement matrix H
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Measurement vector z
        z = np.array([position[0], position[1]])
        
        # Innovation y
        y = z - np.dot(H, self.x)
        
        # Innovation covariance S
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        
        # Kalman gain K
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # Update state
        self.x = self.x + np.dot(K, y)
        
        # Update covariance
        I = np.eye(4)
        self.P = np.dot(I - np.dot(K, H), self.P)
        
        # Smooth shape properties
        if self.shape_smoothing_alpha < 1.0 and self.shape_type == shape_type and len(self.shape_dims) == len(shape_dims):
            alpha = self.shape_smoothing_alpha
            if shape_type == 0:  # CIRCLE
                r_smooth = alpha * shape_dims[0] + (1.0 - alpha) * self.shape_dims[0]
                self.shape_dims = [r_smooth]
            elif shape_type == 1:  # BOX
                l_smooth = alpha * shape_dims[0] + (1.0 - alpha) * self.shape_dims[0]
                w_smooth = alpha * shape_dims[1] + (1.0 - alpha) * self.shape_dims[1]
                
                # Yaw blending with pi-symmetry
                y_prev = self.shape_dims[2]
                y_curr = shape_dims[2]
                
                cos_prev = math.cos(2.0 * y_prev)
                sin_prev = math.sin(2.0 * y_prev)
                cos_curr = math.cos(2.0 * y_curr)
                sin_curr = math.sin(2.0 * y_curr)
                
                cos_smooth = alpha * cos_curr + (1.0 - alpha) * cos_prev
                sin_smooth = alpha * sin_curr + (1.0 - alpha) * sin_prev
                
                yaw_smooth = 0.5 * math.atan2(sin_smooth, cos_smooth)
                yaw_smooth = yaw_smooth % math.pi
                
                self.shape_dims = [l_smooth, w_smooth, yaw_smooth]
            else:
                self.shape_dims = shape_dims
        else:
            self.shape_dims = shape_dims
            
        self.shape_type = shape_type
        self.polygon = polygon
        
        self.age += 1
        self.missed_frames = 0


class LaserObstacleDetectorCore:
    def __init__(self,
                 min_range=0.1,
                 max_range=10.0,
                 beta_incidence_rad=np.radians(10.0),
                 sigma_r=0.01,
                 min_jump_distance=0.1,
                 max_jump_distance=1.0,
                 min_cluster_points=3,
                 max_cluster_points=2000,
                 use_convex_hull=True,
                 split_threshold=0.05,
                 max_association_distance=1.0,
                 min_track_age=3,
                 max_missed_frames=5,
                 dt=0.1,
                 use_median_filter=True,
                 association_method="hungarian",
                 circle_residual_ratio=0.12,
                 max_circle_radius=1.0,
                 corner_angle_min_deg=65.0,
                 corner_angle_max_deg=115.0,
                 shape_smoothing_alpha=0.5):
        """
        Core LaserScan object detector and tracker library. ROS-independent.
        """
        self.min_range = min_range
        self.max_range = max_range
        self.beta = beta_incidence_rad
        self.sigma_r = sigma_r
        self.min_jump_distance = min_jump_distance
        self.max_jump_distance = max_jump_distance
        self.min_cluster_points = min_cluster_points
        self.max_cluster_points = max_cluster_points
        self.use_convex_hull = use_convex_hull
        self.split_threshold = split_threshold
        self.max_association_dist = max_association_distance
        self.min_track_age = min_track_age
        self.max_missed_frames = max_missed_frames
        self.dt = dt
        self.use_median_filter = use_median_filter
        self.association_method = association_method
        self.circle_residual_ratio = circle_residual_ratio
        self.max_circle_radius = max_circle_radius
        self.corner_angle_min_deg = corner_angle_min_deg
        self.corner_angle_max_deg = corner_angle_max_deg
        self.shape_smoothing_alpha = shape_smoothing_alpha
        
        self.tracks = []
        self.next_track_id = 1

    def preprocess_scan(self, ranges, angle_min, angle_increment):
        """
        Applies an optional median filter and projects valid points to 2D Cartesian space.
        Returns:
            points: N x 2 numpy array of (x, y) coordinates
            valid_indices: array of indices matching the original scan angles
        """
        ranges = np.array(ranges)
        num_beams = len(ranges)
        
        if self.use_median_filter:
            # Mask invalid (inf/nan) before filtering to prevent smearing
            process_ranges = np.copy(ranges)
            invalid_mask = np.isnan(process_ranges) | np.isinf(process_ranges) | (process_ranges < self.min_range) | (process_ranges > self.max_range)
            process_ranges[invalid_mask] = self.max_range * 2.0
            filtered_ranges = scipy.signal.medfilt(process_ranges, kernel_size=3)
        else:
            filtered_ranges = ranges
            
        # Cartesian conversion
        angles = angle_min + np.arange(num_beams) * angle_increment
        
        # Mask out values out of sensor physical limits or user bounds
        valid_mask = (filtered_ranges >= self.min_range) & \
                     (filtered_ranges <= self.max_range) & \
                     (~np.isnan(filtered_ranges)) & \
                     (~np.isinf(filtered_ranges))
                     
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.empty((0, 2)), np.empty(0, dtype=int)
            
        xs = filtered_ranges[valid_indices] * np.cos(angles[valid_indices])
        ys = filtered_ranges[valid_indices] * np.sin(angles[valid_indices])
        points = np.column_stack((xs, ys))
        
        return points, valid_indices

    def cluster_points(self, points, valid_indices, angle_increment):
        """
        Performs sequential Jump Distance Clustering (JDC) with an adaptive threshold.
        Fully vectorized over adjacent valid beam pairs.
        """
        N = len(points)
        if N == 0:
            return []
        if N == 1:
            if self.min_cluster_points <= 1 <= self.max_cluster_points:
                return [points]
            return []
            
        # Vectorized sequential clustering
        seg = points[1:] - points[:-1]
        dist = np.linalg.norm(seg, axis=1)
        idx_gap = valid_indices[1:] - valid_indices[:-1]
        d_theta = angle_increment * idx_gap
        r_prev = np.linalg.norm(points[:-1], axis=1)
        
        denom = np.sin(self.beta - d_theta)
        d_th = np.where(denom > 0.01, r_prev * np.sin(d_theta)/denom + 3.0 * self.sigma_r, self.min_jump_distance)
        d_th = np.clip(d_th, self.min_jump_distance, self.max_jump_distance)
        
        break_mask = (idx_gap > 2) | (dist > d_th)
        split_indices = np.where(break_mask)[0] + 1
        
        clusters = [list(c) for c in np.split(np.arange(N), split_indices)]
        
        # Check wrap-around (link last and first cluster if adjacent in 360 scan)
        if len(clusters) > 1:
            p_first = points[clusters[0][0]]
            p_last = points[clusters[-1][-1]]
            dist_val = np.linalg.norm(p_first - p_last)
            
            idx_first = valid_indices[clusters[0][0]]
            idx_last = valid_indices[clusters[-1][-1]]
            total_beams = int(2.0 * math.pi / angle_increment)
            
            d_idx = (idx_first - idx_last) % total_beams
            if d_idx <= 2: # adjacent
                r_last = np.linalg.norm(p_last)
                d_theta_wrap = angle_increment * d_idx
                denom_wrap = math.sin(self.beta - d_theta_wrap)
                d_th_wrap = r_last * (math.sin(d_theta_wrap) / denom_wrap) + 3.0 * self.sigma_r if denom_wrap > 0.01 else self.min_jump_distance
                d_th_wrap = np.clip(d_th_wrap, self.min_jump_distance, self.max_jump_distance)
                
                if dist_val <= d_th_wrap:
                    clusters[0] = clusters[-1] + clusters[0]
                    clusters.pop()
                    
        # Filter cluster sizes
        filtered_clusters = []
        for c in clusters:
            if self.min_cluster_points <= len(c) <= self.max_cluster_points:
                filtered_clusters.append(points[c])
                
        return filtered_clusters

    @staticmethod
    def fit_circle_kasa(points):
        """
        Least-squares circle fitting (Kasa method).
        Returns: centroid [xc, yc], radius
        """
        N = len(points)
        x = points[:, 0]
        y = points[:, 1]
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        u = x - mean_x
        v = y - mean_y
        z = u*u + v*v
        
        # Design matrix A = [2*u, 2*v, 1]
        A = np.column_stack((2*u, 2*v, np.ones(N)))
        
        try:
            # Solve normal equations: A^T A c = A^T z
            c, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            uc, vc = c[0], c[1]
            R2 = c[2] + uc**2 + vc**2
            if R2 < 0:
                raise ValueError("Radius squared is negative")
            R = math.sqrt(R2)
            return np.array([uc + mean_x, vc + mean_y]), R
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to bounding box centroid and half-diagonal
            centroid = np.mean(points, axis=0)
            max_d = np.max(np.linalg.norm(points - centroid, axis=1))
            return centroid, max(max_d, 0.05)

    @staticmethod
    def fit_obb(points):
        """
        Fits an Oriented Bounding Box (OBB) using rotating calipers on the convex hull.
        Returns: center [xc, yc], size [length, width], orientation yaw angle
        """
        N = len(points)
        if N == 0:
            return np.array([0.0, 0.0]), np.array([0.1, 0.1]), 0.0
        if N == 1:
            return points[0], np.array([0.1, 0.1]), 0.0
            
        try:
            hull = scipy.spatial.ConvexHull(points)
            hull_points = points[hull.vertices]
        except Exception:
            # Collinear / degenerate: fallback to extreme points
            p_min = np.argmin(points[:, 0])
            p_max = np.argmax(points[:, 0])
            if np.allclose(points[p_min], points[p_max]):
                p_max = np.argmax(points[:, 1])
            hull_points = points[[p_min, p_max]]
            
        V = len(hull_points)
        best_area = float('inf')
        best_center = np.mean(points, axis=0)
        best_size = np.array([0.1, 0.1])
        best_angle = 0.0
        
        for i in range(V):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % V]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-6:
                continue
            angle = math.atan2(edge[1], edge[0])
            
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R = np.array([
                [cos_a,  sin_a],
                [-sin_a, cos_a]
            ])
            
            rotated = np.dot(points, R.T)
            min_p = np.min(rotated, axis=0)
            max_p = np.max(rotated, axis=0)
            size = max_p - min_p
            area = size[0] * size[1]
            
            if area < best_area:
                best_area = area
                best_size = size
                best_center = np.dot(R.T, min_p + size / 2.0)
                best_angle = angle
                
        # Ensure length > width for consistency
        if best_size[0] < best_size[1]:
            best_size = np.array([best_size[1], best_size[0]])
            best_angle = (best_angle + math.pi / 2.0) % math.pi
        else:
            best_angle = best_angle % math.pi
            
        return best_center, best_size, best_angle

    @staticmethod
    def distance_to_line(p, p1, p2):
        v = p2 - p1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return np.linalg.norm(p - p1)
        return abs(v[0] * (p1[1] - p[1]) - v[1] * (p1[0] - p[0])) / v_norm

    @staticmethod
    def distance_to_line_vectorized(pts, p1, p2):
        v = p2 - p1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return np.linalg.norm(pts - p1, axis=1)
        return np.abs(v[0] * (p1[1] - pts[:, 1]) - v[1] * (p1[0] - pts[:, 0])) / v_norm

    def split_and_merge(self, points):
        """
        Extracts straight line segments from point clusters.
        """
        if len(points) < 2:
            return []
            
        def split_recursive(pts):
            if len(pts) < 2:
                return []
            p1 = pts[0]
            p2 = pts[-1]
            if len(pts) == 2:
                return [[p1, p2]]
                
            dists = self.distance_to_line_vectorized(pts[1:-1], p1, p2)
            if len(dists) == 0:
                return [[p1, p2]]
                
            max_idx = np.argmax(dists) + 1
            max_dist = dists[max_idx - 1]
            
            if max_dist > self.split_threshold:
                left = split_recursive(pts[:max_idx + 1])
                right = split_recursive(pts[max_idx:])
                return left + right
            else:
                return [[p1, p2]]
                
        return split_recursive(points)

    @staticmethod
    def convex_hull_jarvis(points):
        """
        Wrapper around SciPy ConvexHull. Enforces counterclockwise winding.
        """
        N = len(points)
        if N < 3:
            return points
        try:
            hull = scipy.spatial.ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            # Collinear / degenerate fallback
            p_min = np.argmin(points[:, 0])
            p_max = np.argmax(points[:, 0])
            if np.allclose(points[p_min], points[p_max]):
                p_max = np.argmax(points[:, 1])
            if p_min == p_max:
                return points[[p_min]]
            return points[[p_min, p_max]]

    def fit_shape(self, points):
        """
        Fits the best geometric model to a cluster.
        Returns:
            shape_type: integer (0: CIRCLE, 1: BOX, 2: LINE, 3: CORNER)
            centroid: [x, y]
            dims: list of dimension parameters
            polygon: list of points representing outline vertices
        """
        N = len(points)
        centroid = np.mean(points, axis=0)
        
        # 1. Circle Fitting Test
        circle_center, radius = self.fit_circle_kasa(points)
        distances_to_center = np.linalg.norm(points - circle_center, axis=1)
        circle_residual = np.mean(np.abs(distances_to_center - radius))
        
        # Check if the points fit a circular arc cleanly
        is_circle = (circle_residual / radius < self.circle_residual_ratio) and (radius < self.max_circle_radius)
        
        if is_circle:
            # CIRCLE representation
            poly_points = []
            for angle in np.linspace(0, 2.0 * math.pi, 16, endpoint=False):
                poly_points.append([
                    circle_center[0] + radius * math.cos(angle),
                    circle_center[1] + radius * math.sin(angle)
                ])
            return 0, circle_center, [radius], poly_points
            
        # 2. Split-and-Merge Test (Lines / Corners)
        segments = self.split_and_merge(points)
        
        if len(segments) == 1:
            # Single straight line
            p_start, p_end = segments[0][0], segments[0][1]
            return 2, centroid, [], [p_start.tolist(), p_end.tolist()]
            
        elif len(segments) == 2:
            # Corner / L-shape test
            p_start = segments[0][0]
            p_mid = segments[0][1] # corner vertex
            p_end = segments[1][1]
            
            v1 = p_mid - p_start
            v2 = p_end - p_mid
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle_rad = abs(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                if math.radians(self.corner_angle_min_deg) <= angle_rad <= math.radians(self.corner_angle_max_deg):
                    # CORNER representation
                    return 3, p_mid, [], [p_start.tolist(), p_mid.tolist(), p_end.tolist()]
                    
        # 3. Default to Oriented Bounding Box (BOX)
        box_center, box_size, box_yaw = self.fit_obb(points)
        
        # Calculate box corners
        cos_y = math.cos(box_yaw)
        sin_y = math.sin(box_yaw)
        R = np.array([
            [cos_y, -sin_y],
            [sin_y,  cos_y]
        ])
        
        hw = box_size[1] / 2.0
        hl = box_size[0] / 2.0
        
        local_corners = np.array([
            [-hl, -hw],
            [ hl, -hw],
            [ hl,  hw],
            [-hl,  hw]
        ])
        global_corners = np.dot(local_corners, R.T) + box_center
        
        # Use Convex Hull if requested to represent outer profile
        if self.use_convex_hull:
            hull = self.convex_hull_jarvis(points)
            return 1, box_center, [box_size[0], box_size[1], box_yaw], hull.tolist()
        else:
            return 1, box_center, [box_size[0], box_size[1], box_yaw], global_corners.tolist()

    def associate_and_track(self, detected_obstacles, dt=None):
        """
        Updates trackers with new detections using Greedy or Hungarian association.
        detected_obstacles: list of tuples: (shape_type, centroid, dims, polygon)
        """
        if dt is None:
            dt = self.dt
            
        # Predict all existing tracks
        for track in self.tracks:
            track.predict(dt)
            
        if len(detected_obstacles) == 0:
            active_tracks = []
            for track in self.tracks:
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed_frames:
                    active_tracks.append(track)
            self.tracks = active_tracks
            return
            
        det_centroids = np.array([det[1] for det in detected_obstacles])
        track_positions = np.array([track.x[:2] for track in self.tracks])
        
        matched_detections = set()
        matched_tracks = set()
        
        if len(self.tracks) > 0:
            C = scipy.spatial.distance.cdist(det_centroids, track_positions)
            
            if self.association_method == "hungarian":
                C_gated = np.copy(C)
                gate_mask = C_gated > self.max_association_dist
                C_gated[gate_mask] = 1e9
                
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(C_gated)
                
                for r, c in zip(row_ind, col_ind):
                    if C[r, c] < self.max_association_dist:
                        matched_detections.add(r)
                        matched_tracks.add(c)
                        shape_type, centroid, dims, polygon = detected_obstacles[r]
                        self.tracks[c].update(centroid, shape_type, dims, polygon)
            else:
                # Greedy Association
                associations = []
                for d_idx in range(len(detected_obstacles)):
                    for t_idx in range(len(self.tracks)):
                        dist = C[d_idx, t_idx]
                        if dist < self.max_association_dist:
                            associations.append((dist, d_idx, t_idx))
                            
                associations.sort(key=lambda x: x[0])
                
                for dist, d_idx, t_idx in associations:
                    if d_idx not in matched_detections and t_idx not in matched_tracks:
                        matched_detections.add(d_idx)
                        matched_tracks.add(t_idx)
                        shape_type, centroid, dims, polygon = detected_obstacles[d_idx]
                        self.tracks[t_idx].update(centroid, shape_type, dims, polygon)
                        
        # Manage unmatched tracks
        active_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.missed_frames += 1
            if track.missed_frames <= self.max_missed_frames:
                active_tracks.append(track)
        self.tracks = active_tracks
        
        # Manage unmatched detections (spawn new tracks)
        for d_idx, (shape_type, centroid, dims, polygon) in enumerate(detected_obstacles):
            if d_idx not in matched_detections:
                new_track = Track(
                    self.next_track_id,
                    centroid,
                    shape_type,
                    dims,
                    polygon,
                    dt,
                    self.shape_smoothing_alpha
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

    def process(self, ranges, angle_min, angle_increment, dt=None, sensor_pose=None, enable_tracking=True):
        """
        Main execution pipeline call.
        Returns:
            list of active and confirmed Track objects
            list of shape-fit detections: (shape_type, centroid, dims, polygon)
            list of raw clusters in Cartesian space (for debug output)
        """
        if dt is None:
            dt = self.dt
        else:
            self.dt = dt
            
        # 1. Preprocessing (sensor frame)
        points, valid_indices = self.preprocess_scan(ranges, angle_min, angle_increment)
        
        # 2. Clustering (sensor frame)
        clusters = self.cluster_points(points, valid_indices, angle_increment)
        
        # 3. Shape Fitting (sensor frame)
        detections = []
        for cluster in clusters:
            shape_type, centroid, dims, polygon = self.fit_shape(cluster)
            
            # Transform detections into tracking frame if pose provided
            if sensor_pose is not None:
                tx, ty, yaw = sensor_pose
                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                
                # Transform centroid
                cx = centroid[0] * cos_y - centroid[1] * sin_y + tx
                cy = centroid[0] * sin_y + centroid[1] * cos_y + ty
                centroid = np.array([cx, cy])
                
                # Transform polygon
                transformed_poly = []
                for pt in polygon:
                    px = pt[0] * cos_y - pt[1] * sin_y + tx
                    py = pt[0] * sin_y + pt[1] * cos_y + ty
                    transformed_poly.append([px, py])
                polygon = transformed_poly
                
                # Transform OBB yaw
                if shape_type == 1:
                     box_yaw = (dims[2] + yaw) % math.pi
                     dims = [dims[0], dims[1], box_yaw]
                     
            detections.append((shape_type, centroid, dims, polygon))
            
        # 4. Tracking (tracking frame)
        if enable_tracking:
            self.associate_and_track(detections, dt)
            
            # Update confirmation state
            for track in self.tracks:
                track.is_confirmed = (track.age >= self.min_track_age)
        else:
            self.tracks = []
                
        return self.tracks, detections, clusters
