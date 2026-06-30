import time
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodriver_laser_object_segmentation.laser_obstacle_detector_core import LaserObstacleDetectorCore

def generate_profile_data():
    angle_inc = math.radians(1.0)
    ranges = [20.0] * 360
    
    # 1. Circle at range 3.0, angle 0 deg
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        ranges[i % 360] = 3.0
        
    # 2. Box at range 5.0, angle 60 deg (60 beams)
    for i in range(57, 64):
        ranges[i] = 5.0
        
    # 3. Straight wall at range 4.0, angle 120 deg (120 beams)
    for i in range(115, 126):
        ranges[i] = 4.0
        
    # 4. Corner at range 6.0, angle 220 deg
    for i in range(215, 226):
        ranges[i] = 6.0
        
    # 5. Trash can/obstacle at range 8.0, angle 300 deg
    for i in range(297, 303):
        ranges[i] = 8.0
        
    return ranges, -math.pi, angle_inc

def run_profile():
    ranges, angle_min, angle_inc = generate_profile_data()
    
    detector = LaserObstacleDetectorCore(
        min_range=0.1,
        max_range=15.0,
        beta_incidence_rad=math.radians(10.0),
        sigma_r=0.01,
        min_jump_distance=0.15,
        max_jump_distance=1.0,
        min_cluster_points=3,
        max_cluster_points=200,
        use_convex_hull=True,
        split_threshold=0.05,
        max_association_distance=1.0,
        min_track_age=3,
        max_missed_frames=5,
        dt=0.1
    )
    
    # Warm-up
    for _ in range(50):
        detector.process(ranges, angle_min, angle_inc)
        
    # Measure execution time
    iterations = 1000
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        detector.process(ranges, angle_min, angle_inc)
        end = time.perf_counter()
        times.append((end - start) * 1000.0) # in ms
        
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print("--------------------------------------------------")
    print("Python Core Library Latency Profiling Results")
    print(f"Dataset: 360-beam scan, 5 distinct obstacles (shapes: circle, box, line, corner, hull)")
    print(f"Iterations: {iterations}")
    print("--------------------------------------------------")
    print(f"Average Execution Time : {avg_time:6.3f} ms")
    print(f"Minimum Execution Time : {min_time:6.3f} ms")
    print(f"Maximum Execution Time : {max_time:6.3f} ms")
    print(f"Standard Deviation     : {std_time:6.3f} ms")
    print(f"Equivalent Frame Rate  : {1000.0 / avg_time:6.1f} Hz")
    print("--------------------------------------------------")

if __name__ == '__main__':
    run_profile()
