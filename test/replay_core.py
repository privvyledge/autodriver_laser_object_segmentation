"""
Offline replay of a recorded /scan through the Python core, with clustering and
tracking quality metrics. Used for parameter tuning without a ROS 2 runtime.

Unlike profile_latency.py (synthetic data, measures speed), this replays real
recorded scans and measures output *quality*: cluster stability, shape flips,
position jitter and phantom velocity.

The baseline configuration is read from config/params.yaml, NOT from the core's
constructor defaults -- the two differ deliberately, and benchmarking the
constructor defaults measures a configuration that never runs on the robot.

Reference data (stationary, no localization):
    /mnt/d/Coding/Projects/f1tenth/laser_jitter_debug

IMPORTANT -- the reference bag is stationary, so it contains no moving objects.
It can only measure false-positive (phantom) velocity, never the cost of
over-smoothing a real one. Any metric of the form "less velocity is better" is
trivially won by trusting measurements less (--kf-r high) or by process noise
-> 0, which scores perfectly here while destroying real dynamic-obstacle
tracking. Do not tune kf_process_noise or the KF measurement noise against this
bag alone; it supplies only one side of the tradeoff.

Usage:
    python test/replay_core.py                                  # baseline
    python test/replay_core.py --inject "x0=2,y0=-2,vx=0,vy=1,r=0.15"
    python test/replay_core.py --sweep beta_incidence_deg=8,10,12,15
    python test/replay_core.py --sweep min_cluster_points=4,6 --sweep sigma_r=0.02,0.04
    python test/replay_core.py --kf-r 0.02,0.05,0.10            # KF measurement-noise sweep

Each --sweep value is applied on its own, so a param that only takes effect
alongside another must have that other one pinned with --fix:

    python test/replay_core.py --fix association_cost=polygon \
        --sweep max_polygon_association_distance=0.05,0.1,0.2,0.3,0.4,0.6

For a bag recorded while the car was driving, the ego pose must be resolved from
the bag's own TF or every static obstacle is tracked at -v_ego:

    python test/replay_core.py --bag /home/privvyledge/bags \
        --topic /gosling1/lidar/scan_filtered \
        --tf-topic /gosling1/tf --tf-static-topic /gosling1/tf_static \
        --tracking-frame odom --movers 10

Metric columns:
    clus   mean clusters per frame        cstd   std of that count
    dN     mean |change in count| between frames (merge/split churn)
    jmp95  95th pct of matched centroid displacement, m (stationary -> want 0)
    jmpmx  max of the same
    churn  mean fraction of clusters unmatched frame-to-frame
    flp/s  per-track shape-type flips per second, median over tracks
    v95    95th pct track speed, m/s (stationary -> want 0)
    vmax   max track speed, m/s (the phantom)
    jitx   mean per-track std of x, m
    trk    confirmed tracks over the run
    life   median confirmed-track lifetime, s (higher = better continuity)
"""
import argparse
import inspect
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodriver_laser_object_segmentation import laser_obstacle_detector_core as core_mod
from autodriver_laser_object_segmentation.laser_obstacle_detector_core import (
    LaserObstacleDetectorCore,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARAMS = os.path.join(REPO, 'config', 'params.yaml')
BAG = '/mnt/d/Coding/Projects/f1tenth/laser_jitter_debug'

INT_PARAMS = {'min_cluster_points', 'max_cluster_points', 'min_track_age',
              'max_missed_frames', 'shape_type_hysteresis'}


def parse_injection(spec):
    """Parse one comma-separated synthetic circle specification."""
    allowed = {'x0', 'y0', 'vx', 'vy', 'r'}
    values = {}
    for item in spec.split(','):
        try:
            name, raw = item.split('=', 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid injection field {item!r}; expected name=value") from exc
        name = name.strip()
        if name not in allowed:
            raise argparse.ArgumentTypeError(f"unknown injection field {name!r}")
        try:
            values[name] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"injection field {name!r} must be numeric") from exc
    missing = allowed - values.keys()
    if missing:
        raise argparse.ArgumentTypeError(
            f"injection is missing: {', '.join(sorted(missing))}")
    if values['r'] <= 0.0:
        raise argparse.ArgumentTypeError("injection radius must be positive")
    return values


def inject_circle(ranges, angle_min, angle_inc, center, radius):
    """Return a scan copy with an occluding circle rasterized into its beams."""
    injected = np.array(ranges, copy=True)
    angles = angle_min + np.arange(len(injected)) * angle_inc
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    along = directions @ np.asarray(center, dtype=float)
    perpendicular_sq = float(np.dot(center, center)) - along * along
    discriminant = radius * radius - perpendicular_sq
    hits = (along > 0.0) & (discriminant >= 0.0)
    if not np.any(hits):
        return injected

    near = along[hits] - np.sqrt(np.maximum(discriminant[hits], 0.0))
    valid_hit = near >= 0.0
    hit_indices = np.flatnonzero(hits)[valid_hit]
    for index, rho in zip(hit_indices, near[valid_hit]):
        injected[index] = min(injected[index], rho)
    return injected


def load_baseline(params_file=PARAMS):
    """
    Build core kwargs from config/params.yaml.

    Only keys the core constructor accepts are kept, so node-only params
    (tracking_frame, publish_*, tf_*) are ignored. beta is declared in degrees
    but the core takes radians.
    """
    with open(params_file) as f:
        declared = yaml.safe_load(f)['/**']['ros__parameters']

    accepted = set(inspect.signature(LaserObstacleDetectorCore.__init__).parameters)
    kwargs = {k: v for k, v in declared.items() if k in accepted}
    if 'beta_incidence_deg' in declared:
        kwargs['beta_incidence_rad'] = math.radians(declared['beta_incidence_deg'])
    return kwargs


def coerce(name, raw):
    """Parse one param value in the core's own type (int, float, or string)."""
    if name in INT_PARAMS:
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def apply(baseline, name, val):
    """Override one param on top of the baseline, in the core's own units."""
    p = dict(baseline)
    if name == 'beta_incidence_deg':
        p['beta_incidence_rad'] = math.radians(val)
    else:
        p[name] = val
    return p


def storage_id_of(bag_path):
    """Read the storage plugin from metadata.yaml (sqlite3 vs mcap)."""
    meta = os.path.join(bag_path, 'metadata.yaml')
    if os.path.isfile(meta):
        with open(meta) as f:
            info = yaml.safe_load(f)['rosbag2_bagfile_information']
        return info.get('storage_identifier', 'sqlite3')
    return 'sqlite3'


def open_reader(bag_path, topics):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id_of(bag_path)),
        rosbag2_py.ConverterOptions('', ''),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def load_scans(bag_path=BAG, topic='/scan'):
    """Read LaserScan messages out of a rosbag2 into plain tuples."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan

    reader = open_reader(bag_path, [topic])

    scans = []
    frame = ''
    while reader.has_next():
        _topic, data, _t = reader.read_next()
        msg = deserialize_message(data, LaserScan)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = msg.header.frame_id
        scans.append((stamp, np.array(msg.ranges), msg.angle_min, msg.angle_increment))
    return scans, frame


def load_sensor_poses(bag_path, scans, sensor_frame, tracking_frame,
                      tf_topic='/tf', tf_static_topic='/tf_static'):
    """
    Resolve (tx, ty, yaw) of the sensor in the tracking frame at every scan stamp.

    Replays the bag's own /tf and /tf_static into a tf2 buffer rather than
    re-implementing the chain, so the composition matches what the node does at
    runtime. Returns a list aligned with `scans`, with None where the lookup
    fails (TF gap, or a stamp outside the recorded span).
    """
    import rclpy.time
    import tf2_ros
    from rclpy.duration import Duration
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage

    span = scans[-1][0] - scans[0][0] if len(scans) > 1 else 60.0
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=span + 60.0))

    reader = open_reader(bag_path, [tf_topic, tf_static_topic])
    while reader.has_next():
        topic, data, _t = reader.read_next()
        for tr in deserialize_message(data, TFMessage).transforms:
            if topic == tf_static_topic:
                buf.set_transform_static(tr, 'bag')
            else:
                buf.set_transform(tr, 'bag')

    poses = []
    for stamp, _r, _amin, _ainc in scans:
        t = rclpy.time.Time(seconds=int(stamp), nanoseconds=int((stamp % 1) * 1e9))
        try:
            tf = buf.lookup_transform(tracking_frame, sensor_frame, t)
        except Exception:
            poses.append(None)
            continue
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        poses.append((tf.transform.translation.x, tf.transform.translation.y, yaw))
    return poses


def match_centroids(prev, cur, gate):
    """
    Match two centroid sets under a distance gate.

    Hungarian rather than greedy so a merge/split isn't masked by the nearest
    partner being stolen. Returns (displacements, n_unmatched_prev, n_unmatched_cur).
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    if len(prev) == 0 or len(cur) == 0:
        return [], len(prev), len(cur)

    cost = cdist(np.asarray(prev), np.asarray(cur))
    rows, cols = linear_sum_assignment(cost)

    disp = []
    matched_prev, matched_cur = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= gate:
            disp.append(cost[r, c])
            matched_prev.add(r)
            matched_cur.add(c)
    return disp, len(prev) - len(matched_prev), len(cur) - len(matched_cur)


def replay(params, scans, gate=0.3, poses=None, injections=None, gt_gate=0.5,
           gt_spinup=1.0):
    """
    Run every scan through a fresh core and collect metrics.

    sensor_pose=None (identity) is correct for a stationary bag: the sensor
    never moves, so sensor-frame tracking equals odom tracking. For a bag with
    a moving ego, pass `poses` (see load_sensor_poses) -- otherwise every static
    obstacle is measured at -v_ego and the velocity metrics are meaningless.

    Cluster centroids are compared in the tracking frame when poses are given,
    so jmp95/churn measure genuine cluster instability rather than ego motion.
    """
    core = LaserObstacleDetectorCore(**params)

    n_clusters, count_deltas, centroid_jumps, churn, speeds = [], [], [], [], []
    prev_shape, track_flips, track_span, track_x = {}, {}, {}, {}
    track_speeds = {}
    prev_centroids = None
    prev_stamp = None
    injections = injections or []
    stamp0 = scans[0][0] if scans else 0.0
    gt_total = 0
    gt_matches = 0
    gt_id_switches = 0
    gt_last_ids = [None] * len(injections)
    gt_pos_errors = []
    gt_v_errors = []

    for i, (stamp, ranges, angle_min, angle_inc) in enumerate(scans):
        # A scan whose ego pose could not be resolved is dropped rather than
        # processed at identity: feeding sensor-frame points into a tracker
        # holding odom-frame state would fabricate a jump the size of the ego
        # offset. The node makes the same call at runtime (falls back / skips).
        if poses is not None and poses[i] is None:
            continue

        dt = (stamp - prev_stamp) if prev_stamp is not None else 0.1
        if dt <= 0 or dt > 1.0:
            dt = 0.1
        prev_stamp = stamp

        pose = poses[i] if poses is not None else None
        scan_ranges = ranges
        elapsed = stamp - stamp0
        gt_centers = []
        gt_visible = []
        for target in injections:
            center_tracking = np.array([
                target['x0'] + target['vx'] * elapsed,
                target['y0'] + target['vy'] * elapsed,
            ])
            gt_centers.append(center_tracking)
            if pose is None:
                center_sensor = center_tracking
            else:
                tx, ty, yaw = pose
                dx, dy = center_tracking - np.array([tx, ty])
                c_, s_ = math.cos(yaw), math.sin(yaw)
                center_sensor = np.array([c_ * dx + s_ * dy,
                                          -s_ * dx + c_ * dy])
            before = scan_ranges
            scan_ranges = inject_circle(
                before, angle_min, angle_inc, center_sensor, target['r'])
            changed = (np.isfinite(scan_ranges)
                       & (~np.isfinite(before) | (scan_ranges < before)))
            gt_visible.append(np.count_nonzero(changed) >= core.min_cluster_points)

        tracks, _detections, clusters = core.process(
            scan_ranges, angle_min, angle_inc, dt=dt, sensor_pose=pose, enable_tracking=True
        )

        confirmed = [track for track in tracks if track.is_confirmed]
        for target_index, (target, center, visible) in enumerate(
                zip(injections, gt_centers, gt_visible)):
            if not visible:
                continue
            gt_total += 1
            if not confirmed:
                continue
            errors = [float(np.linalg.norm(track.x[:2] - center))
                      for track in confirmed]
            matched_index = int(np.argmin(errors))
            if errors[matched_index] > gt_gate:
                continue
            matched = confirmed[matched_index]
            gt_matches += 1
            gt_pos_errors.append(errors[matched_index])
            previous_id = gt_last_ids[target_index]
            if previous_id is not None and previous_id != matched.id:
                gt_id_switches += 1
            gt_last_ids[target_index] = matched.id
            if elapsed >= gt_spinup:
                true_speed = math.hypot(target['vx'], target['vy'])
                gt_v_errors.append(math.hypot(matched.x[2], matched.x[3]) - true_speed)

        n_clusters.append(len(clusters))
        cur_centroids = [np.mean(c, axis=0) for c in clusters]
        if pose is not None:
            tx, ty, yaw = pose
            c_, s_ = math.cos(yaw), math.sin(yaw)
            cur_centroids = [np.array([tx + c_ * p[0] - s_ * p[1],
                                       ty + s_ * p[0] + c_ * p[1]])
                             for p in cur_centroids]
        if prev_centroids is not None:
            disp, unm_prev, unm_cur = match_centroids(prev_centroids, cur_centroids, gate)
            centroid_jumps.extend(disp)
            count_deltas.append(abs(len(cur_centroids) - len(prev_centroids)))
            denom = max(len(prev_centroids), len(cur_centroids), 1)
            churn.append((unm_prev + unm_cur) / (2.0 * denom))
        prev_centroids = cur_centroids

        # Confirmed tracks only: tentative 1-2 frame tracks never reach a
        # planner, and pooling them swamps the per-track statistics.
        for t in tracks:
            if not t.is_confirmed:
                continue
            v = math.hypot(t.x[2], t.x[3])
            speeds.append(v)
            track_speeds.setdefault(t.id, []).append(v)
            if t.id in prev_shape and prev_shape[t.id] != t.shape_type:
                track_flips[t.id] = track_flips.get(t.id, 0) + 1
            prev_shape[t.id] = t.shape_type
            track_flips.setdefault(t.id, 0)
            track_span[t.id] = track_span.get(t.id, 0.0) + dt
            track_x.setdefault(t.id, []).append(t.x[0])

    stds = [np.std(v) for v in track_x.values() if len(v) > 5]
    flip_rates = [track_flips[i] / track_span[i]
                  for i in track_flips if track_span.get(i, 0.0) > 1.0]

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else 0.0

    result = dict(
        n_clusters_mean=float(np.mean(n_clusters)) if n_clusters else 0.0,
        n_clusters_std=float(np.std(n_clusters)) if n_clusters else 0.0,
        count_delta_mean=float(np.mean(count_deltas)) if count_deltas else 0.0,
        centroid_jump_p95=pct(centroid_jumps, 95),
        centroid_jump_max=float(np.max(centroid_jumps)) if centroid_jumps else 0.0,
        churn_rate=float(np.mean(churn)) if churn else 0.0,
        flips_per_s=float(np.median(flip_rates)) if flip_rates else 0.0,
        speed_p95=pct(speeds, 95),
        speed_max=float(np.max(speeds)) if speeds else 0.0,
        jitter_std_x=float(np.mean(stds)) if stds else 0.0,
        n_tracks=len(track_x),
        # Median confirmed-track lifetime. Under ego motion this is the metric
        # that moves: a planner that sees an obstacle re-IDed every 2 s cannot
        # reason about it, even when the per-frame geometry is fine.
        life_p50=float(np.median([v for v in track_span.values() if v > 1.0]))
        if any(v > 1.0 for v in track_span.values()) else 0.0,
        # Per-track speed summary, for spotting a genuinely moving object among
        # a mass of static ones. Short-lived tracks are excluded: a 2-frame
        # track's velocity is initialization noise, not motion.
        movers=sorted(
            ((tid, float(np.mean(v)), float(np.max(v)), track_span.get(tid, 0.0))
             for tid, v in track_speeds.items() if track_span.get(tid, 0.0) > 1.0),
            key=lambda r: -r[1]),
    )
    if injections:
        result.update(
            gt_recall=gt_matches / gt_total if gt_total else 0.0,
            gt_id_switches=gt_id_switches,
            gt_v_rmse=float(np.sqrt(np.mean(np.square(gt_v_errors))))
            if gt_v_errors else 0.0,
            gt_v_mean_err=float(np.mean(gt_v_errors)) if gt_v_errors else 0.0,
            gt_pos_rmse=float(np.sqrt(np.mean(np.square(gt_pos_errors))))
            if gt_pos_errors else 0.0,
        )
    return result


HEADER = (f"{'config':<34} {'clus':>6} {'cstd':>5} {'dN':>5} {'jmp95':>6} {'jmpmx':>6} "
          f"{'churn':>6} {'flp/s':>6} {'v95':>5} {'vmax':>5} {'jitx':>6} {'trk':>4} "
          f"{'life':>5}")


def fmt(label, m):
    return (f"{label:<34} {m['n_clusters_mean']:>6.1f} {m['n_clusters_std']:>5.2f} "
            f"{m['count_delta_mean']:>5.2f} {m['centroid_jump_p95']:>6.3f} "
            f"{m['centroid_jump_max']:>6.3f} {m['churn_rate']:>6.3f} "
            f"{m['flips_per_s']:>6.2f} {m['speed_p95']:>5.2f} {m['speed_max']:>5.2f} "
            f"{m['jitter_std_x']:>6.3f} {m['n_tracks']:>4d} {m['life_p50']:>5.1f}")


GT_HEADER = (f"{'config':<34} {'recall':>7} {'id_sw':>6} {'v_rmse':>7} "
             f"{'v_bias':>7} {'pos_rmse':>8}")


def fmt_gt(label, m):
    return (f"{label:<34} {m['gt_recall']:>7.3f} {m['gt_id_switches']:>6d} "
            f"{m['gt_v_rmse']:>7.3f} {m['gt_v_mean_err']:>7.3f} "
            f"{m['gt_pos_rmse']:>8.3f}")


def patch_kf_r(var):
    """
    Override the KF measurement noise R for a sweep.

    Track.R is hardcoded diag(0.02, 0.02) and is not a ROS parameter, so it
    cannot be swept through the core's constructor. Patching is confined to
    this harness; the shipped core is untouched.
    """
    original = core_mod.Track.update

    def update(self, position, shape_type, shape_dims, polygon):
        self.R = np.diag([var, var])
        return original(self, position, shape_type, shape_dims, polygon)

    return update, original


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', default=BAG)
    ap.add_argument('--topic', default='/scan', help='LaserScan topic in the bag')
    ap.add_argument('--tracking-frame', default='',
                    help='resolve sensor pose per scan into this frame via the '
                         "bag's own /tf (required for a bag with a moving ego)")
    ap.add_argument('--tf-topic', default='/tf')
    ap.add_argument('--tf-static-topic', default='/tf_static')
    ap.add_argument('--movers', type=int, default=0,
                    help='print the N fastest confirmed tracks of the baseline run')
    ap.add_argument('--params', default=PARAMS)
    ap.add_argument('--fix', action='append', default=[],
                    metavar='name=value',
                    help='pin one core param on top of params.yaml for every row, '
                         'including the baseline (repeatable). Needed to sweep a '
                         'param that only takes effect together with another, e.g. '
                         '--fix association_cost=polygon --sweep '
                         'max_polygon_association_distance=0.1,0.2,0.4')
    ap.add_argument('--sweep', action='append', default=[],
                    help='name=v1,v2,v3 (override one core param; repeatable)')
    ap.add_argument('--kf-r', default='',
                    help='comma-separated KF measurement-noise variances to sweep')
    ap.add_argument('--inject', action='append', type=parse_injection, default=[],
                    metavar='x0=...,y0=...,vx=...,vy=...,r=...',
                    help='inject a moving circle defined in the tracking frame; repeatable')
    ap.add_argument('--limit', type=int, default=0, help='only first N scans')
    args = ap.parse_args()

    baseline = load_baseline(args.params)
    baseline_label = 'BASELINE (params.yaml)'
    for spec in args.fix:
        name, raw = spec.split('=', 1)
        baseline = apply(baseline, name, coerce(name, raw))
    if args.fix:
        baseline_label = 'BASELINE (params.yaml + --fix)'

    scans, sensor_frame = load_scans(args.bag, args.topic)
    if args.limit:
        scans = scans[:args.limit]
    print(f'# {len(scans)} scans from {args.bag} [{args.topic}, frame={sensor_frame}]',
          file=sys.stderr)
    print(f'# baseline from {args.params}', file=sys.stderr)
    for spec in args.fix:
        print(f'# pinned on every row: {spec}', file=sys.stderr)

    poses = None
    if args.tracking_frame:
        poses = load_sensor_poses(args.bag, scans, sensor_frame, args.tracking_frame,
                                  args.tf_topic, args.tf_static_topic)
        ok = sum(p is not None for p in poses)
        print(f'# sensor pose in {args.tracking_frame}: {ok}/{len(poses)} scans resolved',
              file=sys.stderr)

    print(HEADER)
    base = replay(baseline, scans, poses=poses, injections=args.inject)
    print(fmt(baseline_label, base))
    gt_rows = [(baseline_label, base)] if args.inject else []

    for spec in args.sweep:
        name, vals = spec.split('=')
        for v in vals.split(','):
            val = coerce(name, v)
            label = f'{name}={val}'
            metrics = replay(apply(baseline, name, val), scans, poses=poses,
                             injections=args.inject)
            print(fmt(label, metrics))
            if args.inject:
                gt_rows.append((label, metrics))

    if args.kf_r:
        for v in args.kf_r.split(','):
            patched, original = patch_kf_r(float(v))
            core_mod.Track.update = patched
            try:
                label = f'kf_R=diag({float(v):.3f})'
                metrics = replay(baseline, scans, poses=poses,
                                 injections=args.inject)
                print(fmt(label, metrics))
                if args.inject:
                    gt_rows.append((label, metrics))
            finally:
                core_mod.Track.update = original

    if gt_rows:
        print(f'\n{GT_HEADER}')
        for label, metrics in gt_rows:
            print(fmt_gt(label, metrics))

    if args.movers:
        print(f"\n{'track':>6} {'v_mean':>7} {'v_max':>7} {'span_s':>7}")
        for tid, vmean, vmax, span in base['movers'][:args.movers]:
            print(f'{tid:>6} {vmean:>7.2f} {vmax:>7.2f} {span:>7.1f}')


if __name__ == '__main__':
    main()
