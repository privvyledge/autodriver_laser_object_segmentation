import numpy as np
import math


class Track:
    def __init__(self, track_id, position, shape_type, shape_dims, polygon, dt):
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
        
        # Update shape properties
        self.shape_type = shape_type
        self.shape_dims = shape_dims
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
                 max_cluster_points=150,
                 use_convex_hull=True,
                 split_threshold=0.05,
                 max_association_distance=1.0,
                 min_track_age=3,
                 max_missed_frames=5,
                 dt=0.1):
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
        
        self.tracks = []
        self.next_track_id = 1

    def preprocess_scan(self, ranges, angle_min, angle_increment):
        """
        Applies a median filter and projects valid points to 2D Cartesian space.
        Returns:
            points: N x 2 numpy array of (x, y) coordinates
            valid_indices: array of indices matching the original scan angles
        """
        ranges = np.array(ranges)
        num_beams = len(ranges)
        
        # 1D Median filter for salt-and-pepper noise (window size = 3)
        filtered_ranges = np.copy(ranges)
        for i in range(1, num_beams - 1):
            filtered_ranges[i] = np.median(ranges[i-1:i+2])
            
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
        Since scan points are ordered radially, adjacencies are analyzed sequentially.
        """
        N = len(points)
        if N == 0:
            return []
            
        clusters = []
        current_cluster = [0]
        
        for i in range(1, N):
            idx_prev = valid_indices[i-1]
            idx_curr = valid_indices[i]
            
            # Check if points are adjacent in terms of scan index
            # In a full scan, they are adjacent if diff == 1
            if idx_curr - idx_prev > 2:  # allow at most 1 skipped beam in between
                # Large gap in angle, start new cluster
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [i]
                continue
                
            # Compute Euclidean distance
            p_prev = points[i-1]
            p_curr = points[i]
            dist = np.linalg.norm(p_curr - p_prev)
            
            # Adaptive Jump Distance (Dietmayer formula)
            r_prev = np.linalg.norm(p_prev)
            d_theta = angle_increment * (idx_curr - idx_prev)
            
            denom = math.sin(self.beta - d_theta)
            if denom > 0.01:
                d_th = r_prev * (math.sin(d_theta) / denom) + 3.0 * self.sigma_r
            else:
                d_th = self.min_jump_distance
                
            # Clamp threshold
            d_th = np.clip(d_th, self.min_jump_distance, self.max_jump_distance)
            
            if dist > d_th:
                # Segment jump! Start new cluster
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [i]
            else:
                current_cluster.append(i)
                
        # Append the last cluster
        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)
            
        # Check wrap-around (link last and first cluster if adjacent in 360 scan)
        if len(clusters) > 1:
            p_first = points[clusters[0][0]]
            p_last = points[clusters[-1][-1]]
            dist = np.linalg.norm(p_first - p_last)
            
            # Compute adaptive threshold for wrap-around
            idx_first = valid_indices[clusters[0][0]]
            idx_last = valid_indices[clusters[-1][-1]]
            total_beams = int(2.0 * math.pi / angle_increment)
            
            # Distance in beams considering wrap-around
            d_idx = (idx_first - idx_last) % total_beams
            if d_idx <= 2: # adjacent
                r_last = np.linalg.norm(p_last)
                d_theta = angle_increment * d_idx
                denom = math.sin(self.beta - d_theta)
                d_th = r_last * (math.sin(d_theta) / denom) + 3.0 * self.sigma_r if denom > 0.01 else self.min_jump_distance
                d_th = np.clip(d_th, self.min_jump_distance, self.max_jump_distance)
                
                if dist <= d_th:
                    # Merge last cluster into first
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
        Fits an Oriented Bounding Box (OBB) using orientation search.
        Returns: center [xc, yc], size [length, width], orientation yaw angle
        """
        best_area = float('inf')
        best_center = np.mean(points, axis=0)
        best_size = np.array([0.1, 0.1])
        best_angle = 0.0
        
        # Search orientations in 2-degree increments from 0 to 90
        angles = np.radians(np.arange(0, 90, 2))
        for angle in angles:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # Rotation matrix (from global to local frame)
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
                # Rotated center
                rot_center = min_p + size / 2.0
                # Transform back to global frame
                best_center = np.dot(R.T, rot_center)
                best_angle = angle
                
        # Ensure length > width for consistency
        if best_size[0] < best_size[1]:
            best_size = np.array([best_size[1], best_size[0]])
            best_angle = (best_angle + math.pi / 2.0) % math.pi
            
        return best_center, best_size, best_angle

    @staticmethod
    def distance_to_line(p, p1, p2):
        v = p2 - p1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return np.linalg.norm(p - p1)
        return abs(v[0] * (p1[1] - p[1]) - v[1] * (p1[0] - p[0])) / v_norm

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
            
            max_dist = -1.0
            max_idx = -1
            
            for i in range(1, len(pts) - 1):
                dist = self.distance_to_line(pts[i], p1, p2)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
                    
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
        Gift wrapping algorithm to find 2D Convex Hull.
        """
        N = len(points)
        if N < 3:
            return points
            
        # Leftmost point
        start_idx = np.argmin(points[:, 0])
        
        hull = []
        p = start_idx
        while True:
            hull.append(p)
            q = (p + 1) % N
            for i in range(N):
                # PQ x PI cross product
                cross = (points[q][0] - points[p][0]) * (points[i][1] - points[p][1]) - \
                        (points[q][1] - points[p][1]) * (points[i][0] - points[p][0])
                if cross > 0:
                    q = i
                elif cross == 0:
                    dist_q = (points[q][0] - points[p][0])**2 + (points[q][1] - points[p][1])**2
                    dist_i = (points[i][0] - points[p][0])**2 + (points[i][1] - points[p][1])**2
                    if dist_i > dist_q:
                        q = i
            p = q
            if p == start_idx:
                break
                
        return points[hull]

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
        mean_dist = np.mean(distances_to_center)
        circle_residual = np.mean(np.abs(distances_to_center - radius))
        
        # Check if the points fit a circular arc cleanly
        is_circle = (circle_residual / radius < 0.12) and (radius < 1.0)
        
        if is_circle:
            # CIRCLE representation
            # We can create a polygon approximation of the circle for controllers
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
                # Angle in radians between segments (dot product of direction vectors)
                # If orthog, cos_theta ~ 0
                angle_rad = abs(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                # A 90 deg corner in lines translates to v1 and v2 being perpendicular.
                # Since v1 = p_mid - p_start, and v2 = p_end - p_mid,
                # if they meet at 90 deg, the angle between their direction vectors is ~ 90 deg.
                if math.radians(65.0) <= angle_rad <= math.radians(115.0):
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

    def associate_and_track(self, detected_obstacles):
        """
        Updates trackers with new detections using Greedy nearest-neighbor association.
        detected_obstacles: list of tuples: (shape_type, centroid, dims, polygon)
        """
        # Predict all existing tracks
        for track in self.tracks:
            track.predict(self.dt)
            
        matched_detections = set()
        matched_tracks = set()
        
        # Greedy Association based on distance
        associations = []
        for d_idx, (_, det_centroid, _, _) in enumerate(detected_obstacles):
            for t_idx, track in enumerate(self.tracks):
                dist = np.linalg.norm(det_centroid - track.x[:2])
                if dist < self.max_association_dist:
                    associations.append((dist, d_idx, t_idx))
                    
        # Sort by distance
        associations.sort(key=lambda x: x[0])
        
        for dist, d_idx, t_idx in associations:
            if d_idx not in matched_detections and t_idx not in matched_tracks:
                matched_detections.add(d_idx)
                matched_tracks.add(t_idx)
                # Update track with this detection
                shape_type, centroid, dims, polygon = detected_obstacles[d_idx]
                self.tracks[t_idx].update(centroid, shape_type, dims, polygon)
                
        # Manage unmatched tracks (increment missed frame counts)
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
                    self.dt
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

    def process(self, ranges, angle_min, angle_increment):
        """
        Main execution pipeline call.
        Returns:
            list of active and confirmed Track objects
            list of raw clusters in Cartesian space (for debug output)
        """
        # 1. Preprocessing
        points, valid_indices = self.preprocess_scan(ranges, angle_min, angle_increment)
        
        # 2. Clustering
        clusters = self.cluster_points(points, valid_indices, angle_increment)
        
        # 3. Shape Fitting
        detections = []
        for cluster in clusters:
            shape_type, centroid, dims, polygon = self.fit_shape(cluster)
            detections.append((shape_type, centroid, dims, polygon))
            
        # 4. Tracking
        self.associate_and_track(detections)
        
        # Filter confirmed tracks for downstream output
        confirmed_tracks = []
        for track in self.tracks:
            # Promote track after min_track_age frames
            if track.age >= self.min_track_age:
                track.is_confirmed = True
                confirmed_tracks.append(track)
                
        return confirmed_tracks, clusters
