#include "autodriver_laser_object_segmentation/laser_obstacle_detector_core.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>
#include <iostream>

namespace autodriver_laser_object_segmentation
{

// 4x4 matrix helpers
static std::array<double, 16> mat_mul_4x4(const std::array<double, 16>& A, const std::array<double, 16>& B)
{
    std::array<double, 16> C{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            double val = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                val += A[r * 4 + k] * B[k * 4 + c];
            }
            C[r * 4 + c] = val;
        }
    }
    return C;
}

static std::array<double, 16> mat_transpose_4x4(const std::array<double, 16>& A)
{
    std::array<double, 16> AT{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            AT[c * 4 + r] = A[r * 4 + c];
        }
    }
    return AT;
}

LaserObstacleDetectorCore::LaserObstacleDetectorCore(
    double min_range,
    double max_range,
    double beta_incidence_rad,
    double sigma_r,
    double min_jump_distance,
    double max_jump_distance,
    uint32_t min_cluster_points,
    uint32_t max_cluster_points,
    bool use_convex_hull,
    double split_threshold,
    double max_association_distance,
    uint32_t min_track_age,
    uint32_t max_missed_frames,
    double dt
) : min_range(min_range),
    max_range(max_range),
    beta(beta_incidence_rad),
    sigma_r(sigma_r),
    min_jump_distance(min_jump_distance),
    max_jump_distance(max_jump_distance),
    min_cluster_points(min_cluster_points),
    max_cluster_points(max_cluster_points),
    use_convex_hull(use_convex_hull),
    split_threshold(split_threshold),
    max_association_dist(max_association_distance),
    min_track_age(min_track_age),
    max_missed_frames(max_missed_frames),
    dt(dt),
    next_track_id_(1)
{
}

std::pair<std::vector<Track>, std::vector<std::vector<Point2D>>> LaserObstacleDetectorCore::process(
    const std::vector<float>& ranges,
    double angle_min,
    double angle_increment
)
{
    // 1. Preprocessing
    auto [points, valid_indices] = preprocess_scan(ranges, angle_min, angle_increment);

    // 2. Clustering
    auto clusters = cluster_points(points, valid_indices, angle_increment);

    // 3. Shape Fitting
    std::vector<std::tuple<uint8_t, Point2D, std::vector<double>, std::vector<Point2D>>> detections;
    for (const auto& cluster : clusters)
    {
        uint8_t shape_type;
        Point2D centroid;
        std::vector<double> dims;
        std::vector<Point2D> polygon;
        fit_shape(cluster, shape_type, centroid, dims, polygon);
        detections.push_back({shape_type, centroid, dims, polygon});
    }

    // 4. Tracking
    associate_and_update(detections);

    // Filter and return confirmed tracks
    std::vector<Track> confirmed_tracks;
    for (auto& track : tracks_)
    {
        if (track.age >= min_track_age)
        {
            track.is_confirmed = true;
            confirmed_tracks.push_back(track);
        }
    }

    return {confirmed_tracks, clusters};
}

std::pair<std::vector<Point2D>, std::vector<size_t>> LaserObstacleDetectorCore::preprocess_scan(
    const std::vector<float>& ranges,
    double angle_min,
    double angle_increment
)
{
    size_t num_beams = ranges.size();
    if (num_beams == 0)
    {
        return {{}, {}};
    }

    // Median filter
    std::vector<float> filtered_ranges = ranges;
    for (size_t i = 1; i < num_beams - 1; ++i)
    {
        std::array<float, 3> window = {ranges[i-1], ranges[i], ranges[i+1]};
        std::sort(window.begin(), window.end());
        filtered_ranges[i] = window[1];
    }

    std::vector<Point2D> points;
    std::vector<size_t> valid_indices;

    for (size_t i = 0; i < num_beams; ++i)
    {
        float r = filtered_ranges[i];
        if (std::isnan(r) || std::isinf(r) || r < min_range || r > max_range)
        {
            continue;
        }

        double angle = angle_min + i * angle_increment;
        Point2D p{r * std::cos(angle), r * std::sin(angle)};
        points.push_back(p);
        valid_indices.push_back(i);
    }

    return {points, valid_indices};
}

std::vector<std::vector<Point2D>> LaserObstacleDetectorCore::cluster_points(
    const std::vector<Point2D>& points,
    const std::vector<size_t>& valid_indices,
    double angle_increment
)
{
    size_t N = points.size();
    if (N == 0)
    {
        return {};
    }

    std::vector<std::vector<size_t>> clusters;
    std::vector<size_t> current_cluster = {0};

    for (size_t i = 1; i < N; ++i)
    {
        size_t idx_prev = valid_indices[i-1];
        size_t idx_curr = valid_indices[i];

        if (idx_curr - idx_prev > 2)
        {
            // Large angular gap
            if (current_cluster.size() >= min_cluster_points)
            {
                clusters.push_back(current_cluster);
            }
            current_cluster = {i};
            continue;
        }

        double dx = points[i].x - points[i-1].x;
        double dy = points[i].y - points[i-1].y;
        double dist = std::sqrt(dx*dx + dy*dy);

        double r_prev = std::sqrt(points[i-1].x * points[i-1].x + points[i-1].y * points[i-1].y);
        double d_theta = angle_increment * (idx_curr - idx_prev);

        double denom = std::sin(beta - d_theta);
        double d_th;
        if (denom > 0.01)
        {
            d_th = r_prev * (std::sin(d_theta) / denom) + 3.0 * sigma_r;
        }
        else
        {
            d_th = min_jump_distance;
        }

        // Clamp jump distance threshold
        d_th = std::max(min_jump_distance, std::min(d_th, max_jump_distance));

        if (dist > d_th)
        {
            if (current_cluster.size() >= min_cluster_points)
            {
                clusters.push_back(current_cluster);
            }
            current_cluster = {i};
        }
        else
        {
            current_cluster.push_back(i);
        }
    }

    if (current_cluster.size() >= min_cluster_points)
    {
        clusters.push_back(current_cluster);
    }

    // Wrap around merge check for 360 degree scans
    if (clusters.size() > 1)
    {
        Point2D p_first = points[clusters.front().front()];
        Point2D p_last = points[clusters.back().back()];

        double dx = p_first.x - p_last.x;
        double dy = p_first.y - p_last.y;
        double dist = std::sqrt(dx*dx + dy*dy);

        size_t idx_first = valid_indices[clusters.front().front()];
        size_t idx_last = valid_indices[clusters.back().back()];
        size_t total_beams = static_cast<size_t>(2.0 * M_PI / angle_increment);

        size_t d_idx = (idx_first - idx_last + total_beams) % total_beams;
        if (d_idx <= 2)
        {
            double r_last = std::sqrt(p_last.x * p_last.x + p_last.y * p_last.y);
            double d_theta = angle_increment * d_idx;
            double denom = std::sin(beta - d_theta);
            double d_th = denom > 0.01 ? (r_last * (std::sin(d_theta) / denom) + 3.0 * sigma_r) : min_jump_distance;
            d_th = std::max(min_jump_distance, std::min(d_th, max_jump_distance));

            if (dist <= d_th)
            {
                // Merge last into first
                clusters.front().insert(clusters.front().begin(), clusters.back().begin(), clusters.back().end());
                clusters.pop_back();
            }
        }
    }

    // Convert indices to Points and filter sizes
    std::vector<std::vector<Point2D>> filtered_clusters;
    for (const auto& c_indices : clusters)
    {
        if (c_indices.size() >= min_cluster_points && c_indices.size() <= max_cluster_points)
        {
            std::vector<Point2D> c_pts;
            c_pts.reserve(c_indices.size());
            for (size_t idx : c_indices)
            {
                c_pts.push_back(points[idx]);
            }
            filtered_clusters.push_back(c_pts);
        }
    }

    return filtered_clusters;
}

void LaserObstacleDetectorCore::fit_circle_kasa(const std::vector<Point2D>& points, Point2D& center, double& radius)
{
    size_t N = points.size();
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& p : points)
    {
        sum_x += p.x;
        sum_y += p.y;
    }
    double mean_x = sum_x / N;
    double mean_y = sum_y / N;

    // Shift coordinates for stability
    std::vector<double> u(N), v(N), z(N);
    double sum_u = 0, sum_v = 0, sum_z = 0;
    for (size_t i = 0; i < N; ++i)
    {
        u[i] = points[i].x - mean_x;
        v[i] = points[i].y - mean_y;
        z[i] = u[i]*u[i] + v[i]*v[i];
    }

    // Set up least squares: A^T A c = A^T z
    // Matrix A is N x 3: [2*u, 2*v, 1]
    // Vector z is N x 1: [u^2 + v^2]
    // Compute A^T A (3x3 symmetric matrix) and A^T z (3x1 vector)
    double ATA[3][3] = {0};
    double ATz[3] = {0};

    for (size_t i = 0; i < N; ++i)
    {
        double ui = u[i];
        double vi = v[i];
        double zi = z[i];

        ATA[0][0] += 4.0 * ui * ui;
        ATA[0][1] += 4.0 * ui * vi;
        ATA[0][2] += 2.0 * ui;

        ATA[1][1] += 4.0 * vi * vi;
        ATA[1][2] += 2.0 * vi;

        ATA[2][2] += 1.0;

        ATz[0] += 2.0 * ui * zi;
        ATz[1] += 2.0 * vi * zi;
        ATz[2] += zi;
    }
    ATA[1][0] = ATA[0][1];
    ATA[2][0] = ATA[0][2];
    ATA[2][1] = ATA[1][2];

    // Solve using Cramer's rule for 3x3 matrix
    double det = ATA[0][0] * (ATA[1][1]*ATA[2][2] - ATA[1][2]*ATA[2][1]) -
                 ATA[0][1] * (ATA[1][0]*ATA[2][2] - ATA[1][2]*ATA[2][0]) +
                 ATA[0][2] * (ATA[1][0]*ATA[2][1] - ATA[1][1]*ATA[2][0]);

    if (std::abs(det) < 1e-9)
    {
        // Fallback
        center.x = mean_x;
        center.y = mean_y;
        radius = 0.1;
        return;
    }

    double det0 = ATz[0] * (ATA[1][1]*ATA[2][2] - ATA[1][2]*ATA[2][1]) -
                  ATA[0][1] * (ATz[1]*ATA[2][2] - ATA[1][2]*ATz[2]) +
                  ATA[0][2] * (ATz[1]*ATA[2][1] - ATA[1][1]*ATz[2]);

    double det1 = ATA[0][0] * (ATz[1]*ATA[2][2] - ATA[1][2]*ATz[2]) -
                  ATz[0] * (ATA[1][0]*ATA[2][2] - ATA[1][2]*ATA[2][0]) +
                  ATA[0][2] * (ATA[1][0]*ATz[2] - ATz[1]*ATA[2][0]);

    double det2 = ATA[0][0] * (ATA[1][1]*ATz[2] - ATA[1][2]*ATz[1]) -
                  ATA[0][1] * (ATA[1][0]*ATz[2] - ATA[1][2]*ATz[0]) +
                  ATz[0] * (ATA[1][0]*ATz[1] - ATA[1][1]*ATA[2][0]);

    double uc = det0 / det;
    double vc = det1 / det;
    double w  = det2 / det;

    double R2 = w + uc*uc + vc*vc;
    if (R2 < 0)
    {
        center.x = mean_x;
        center.y = mean_y;
        radius = 0.1;
    }
    else
    {
        center.x = uc + mean_x;
        center.y = vc + mean_y;
        radius = std::sqrt(R2);
    }
}

void LaserObstacleDetectorCore::fit_obb(const std::vector<Point2D>& points, Point2D& center, double& length, double& width, double& yaw)
{
    double best_area = std::numeric_limits<double>::max();
    double best_yaw = 0.0;
    Point2D best_center{0, 0};
    double best_length = 0.1;
    double best_width = 0.1;

    // Search orientations from 0 to 90 degrees in 2 degree steps
    for (int deg = 0; deg < 90; deg += 2)
    {
        double angle = deg * M_PI / 180.0;
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);

        double min_u = std::numeric_limits<double>::max();
        double max_u = -std::numeric_limits<double>::max();
        double min_v = std::numeric_limits<double>::max();
        double max_v = -std::numeric_limits<double>::max();

        for (const auto& p : points)
        {
            double u = p.x * cos_a + p.y * sin_a;
            double v = -p.x * sin_a + p.y * cos_a;

            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }

        double len_u = max_u - min_u;
        double len_v = max_v - min_v;
        double area = len_u * len_v;

        if (area < best_area)
        {
            best_area = area;
            best_length = len_u;
            best_width = len_v;
            best_yaw = angle;

            double mid_u = min_u + len_u / 2.0;
            double mid_v = min_v + len_v / 2.0;

            // Transform back to global
            best_center.x = mid_u * cos_a - mid_v * sin_a;
            best_center.y = mid_u * sin_a + mid_v * cos_a;
        }
    }

    // Ensure length > width for consistency
    if (best_length < best_width)
    {
        std::swap(best_length, best_width);
        best_yaw = std::fmod(best_yaw + M_PI_2, M_PI);
    }

    center = best_center;
    length = best_length;
    width = best_width;
    yaw = best_yaw;
}

double LaserObstacleDetectorCore::distance_to_line(const Point2D& p, const Point2D& p1, const Point2D& p2)
{
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double v_norm = std::sqrt(dx*dx + dy*dy);
    if (v_norm < 1e-6)
    {
        double d_dx = p.x - p1.x;
        double d_dy = p.y - p1.y;
        return std::sqrt(d_dx*d_dx + d_dy*d_dy);
    }
    return std::abs(dx * (p1.y - p.y) - dy * (p1.x - p.x)) / v_norm;
}

std::vector<std::vector<Point2D>> LaserObstacleDetectorCore::split_and_merge(const std::vector<Point2D>& points)
{
    if (points.size() < 2)
    {
        return {};
    }

    struct Splitter
    {
        double split_thr;
        std::vector<std::vector<Point2D>> segments;

        void split_recursive(const std::vector<Point2D>& pts)
        {
            if (pts.size() < 2) return;
            Point2D p1 = pts.front();
            Point2D p2 = pts.back();

            double max_dist = -1.0;
            size_t max_idx = 0;

            for (size_t i = 1; i < pts.size() - 1; ++i)
            {
                double dist = distance_to_line(pts[i], p1, p2);
                if (dist > max_dist)
                {
                    max_dist = dist;
                    max_idx = i;
                }
            }

            if (max_dist > split_thr)
            {
                std::vector<Point2D> left(pts.begin(), pts.begin() + max_idx + 1);
                std::vector<Point2D> right(pts.begin() + max_idx, pts.end());
                split_recursive(left);
                split_recursive(right);
            }
            else
            {
                segments.push_back({p1, p2});
            }
        }
    };

    Splitter splitter{split_threshold, {}};
    splitter.split_recursive(points);
    return splitter.segments;
}

std::vector<Point2D> LaserObstacleDetectorCore::convex_hull_jarvis(const std::vector<Point2D>& points)
{
    size_t N = points.size();
    if (N < 3)
    {
        return points;
    }

    // Find leftmost
    size_t start_idx = 0;
    for (size_t i = 1; i < N; ++i)
    {
        if (points[i].x < points[start_idx].x)
        {
            start_idx = i;
        }
    }

    std::vector<size_t> hull_idx;
    size_t p = start_idx;
    do
    {
        hull_idx.push_back(p);
        size_t q = (p + 1) % N;
        for (size_t i = 0; i < N; ++i)
        {
            double cross = (points[q].x - points[p].x) * (points[i].y - points[p].y) -
                           (points[q].y - points[p].y) * (points[i].x - points[p].x);
            if (cross > 0)
            {
                q = i;
            }
            else if (cross == 0)
            {
                double dist_q = (points[q].x - points[p].x)*(points[q].x - points[p].x) +
                                (points[q].y - points[p].y)*(points[q].y - points[p].y);
                double dist_i = (points[i].x - points[p].x)*(points[i].x - points[p].x) +
                                (points[i].y - points[p].y)*(points[i].y - points[p].y);
                if (dist_i > dist_q)
                {
                    q = i;
                }
            }
        }
        p = q;
    } while (p != start_idx);

    std::vector<Point2D> hull;
    hull.reserve(hull_idx.size());
    for (size_t idx : hull_idx)
    {
        hull.push_back(points[idx]);
    }

    return hull;
}

void LaserObstacleDetectorCore::fit_shape(
    const std::vector<Point2D>& cluster,
    uint8_t& shape_type,
    Point2D& centroid,
    std::vector<double>& dims,
    std::vector<Point2D>& polygon
)
{
    size_t N = cluster.size();
    double sum_x = 0, sum_y = 0;
    for (const auto& p : cluster)
    {
        sum_x += p.x;
        sum_y += p.y;
    }
    centroid = Point2D{sum_x / N, sum_y / N};

    // 1. Circle fit
    Point2D circle_center{0, 0};
    double radius = 0.1;
    fit_circle_kasa(cluster, circle_center, radius);

    double residual_sum = 0.0;
    for (const auto& p : cluster)
    {
        double dx = p.x - circle_center.x;
        double dy = p.y - circle_center.y;
        double dist = std::sqrt(dx*dx + dy*dy);
        residual_sum += std::abs(dist - radius);
    }
    double mean_residual = residual_sum / N;

    bool is_circle = (mean_residual / radius < 0.12) && (radius < 1.0);

    if (is_circle)
    {
        shape_type = 0; // CIRCLE
        dims = {radius};
        
        // Approximate circle polygon (16 points)
        polygon.clear();
        for (int i = 0; i < 16; ++i)
        {
            double angle = i * 2.0 * M_PI / 16.0;
            polygon.push_back({
                circle_center.x + radius * std::cos(angle),
                circle_center.y + radius * std::sin(angle)
            });
        }
        return;
    }

    // 2. Split-and-Merge line fit
    auto segments = split_and_merge(cluster);

    if (segments.size() == 1)
    {
        // Single straight wall line
        shape_type = 2; // LINE
        dims.clear();
        polygon = {segments[0][0], segments[0][1]};
        return;
    }
    else if (segments.size() == 2)
    {
        Point2D p_start = segments[0][0];
        Point2D p_mid = segments[0][1];
        Point2D p_end = segments[1][1];

        double dx1 = p_mid.x - p_start.x;
        double dy1 = p_mid.y - p_start.y;
        double dx2 = p_end.x - p_mid.x;
        double dy2 = p_end.y - p_mid.y;

        double norm1 = std::sqrt(dx1*dx1 + dy1*dy1);
        double norm2 = std::sqrt(dx2*dx2 + dy2*dy2);

        if (norm1 > 0.01 && norm2 > 0.01)
        {
            double cos_theta = (dx1*dx2 + dy1*dy2) / (norm1 * norm2);
            double angle_rad = std::abs(std::acos(std::clamp(cos_theta, -1.0, 1.0)));

            // 90 deg corner (range 65 - 115 deg)
            if (angle_rad >= (65.0 * M_PI / 180.0) && angle_rad <= (115.0 * M_PI / 180.0))
            {
                shape_type = 3; // CORNER
                centroid = p_mid; // Center at corner vertex
                dims.clear();
                polygon = {p_start, p_mid, p_end};
                return;
            }
        }
    }

    // 3. Oriented Bounding Box
    double length = 0.1, width = 0.1, yaw = 0.0;
    fit_obb(cluster, centroid, length, width, yaw);
    shape_type = 1; // BOX
    dims = {length, width, yaw};

    if (use_convex_hull)
    {
        polygon = convex_hull_jarvis(cluster);
    }
    else
    {
        // Generate OBB corner points
        double cos_y = std::cos(yaw);
        double sin_y = std::sin(yaw);
        double hl = length / 2.0;
        double hw = width / 2.0;

        polygon = {
            {centroid.x - hl*cos_y + hw*sin_y, centroid.y - hl*sin_y - hw*cos_y},
            {centroid.x + hl*cos_y + hw*sin_y, centroid.y + hl*sin_y - hw*cos_y},
            {centroid.x + hl*cos_y - hw*sin_y, centroid.y + hl*sin_y + hw*cos_y},
            {centroid.x - hl*cos_y - hw*sin_y, centroid.y - hl*sin_y + hw*cos_y}
        };
    }
}

void LaserObstacleDetectorCore::predict_tracks()
{
    for (auto& track : tracks_)
    {
        // Predict state: x_new = F * x
        track.x[0] += dt * track.x[2];
        track.x[1] += dt * track.x[3];

        // Process noise parameter q
        double q_val = 0.5;

        // Predict covariance: P = F * P * F^T + Q
        std::array<double, 16> F = {
            1.0, 0.0,  dt, 0.0,
            0.0, 1.0, 0.0,  dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };

        std::array<double, 16> Q = {
            dt*dt*dt/3.0 * q_val,                 0.0, dt*dt/2.0 * q_val,                 0.0,
                             0.0, dt*dt*dt/3.0 * q_val,              0.0, dt*dt/2.0 * q_val,
            dt*dt/2.0 * q_val,                 0.0,             dt * q_val,                 0.0,
                             0.0, dt*dt/2.0 * q_val,                 0.0,              dt * q_val
        };

        auto FP = mat_mul_4x4(F, track.P);
        auto FT = mat_transpose_4x4(F);
        auto FPF = mat_mul_4x4(FP, FT);

        for (int i = 0; i < 16; ++i)
        {
            track.P[i] = FPF[i] + Q[i];
        }
    }
}

void LaserObstacleDetectorCore::update_track_kf(Track& track, const Point2D& detection_centroid)
{
    // H = [1, 0, 0, 0; 0, 1, 0, 0]
    // R = diag(0.02, 0.02)
    double r_val = 0.02;

    // Innovation y = z - H * x
    double y0 = detection_centroid.x - track.x[0];
    double y1_correct = detection_centroid.y - track.x[1];

    // S = H * P * H^T + R
    // H P H^T is the top-left 2x2 of P
    double s00 = track.P[0] + r_val;
    double s01 = track.P[1];
    double s10 = track.P[4];
    double s11 = track.P[5] + r_val;

    // Det of S
    double det = s00 * s11 - s01 * s10;
    if (std::abs(det) < 1e-9) return;

    // Sinv
    double sinv00 = s11 / det;
    double sinv01 = -s01 / det;
    double sinv10 = -s10 / det;
    double sinv11 = s00 / det;

    // Kalman Gain K = P * H^T * Sinv
    // P * H^T (4x2 matrix, first two columns of P)
    // k00 = P00*sinv00 + P01*sinv10
    double k00 = track.P[0]*sinv00 + track.P[1]*sinv10;
    double k01 = track.P[0]*sinv01 + track.P[1]*sinv11;

    double k10 = track.P[4]*sinv00 + track.P[5]*sinv10;
    double k11 = track.P[4]*sinv01 + track.P[5]*sinv11;

    double k20 = track.P[8]*sinv00 + track.P[9]*sinv10;
    double k21 = track.P[8]*sinv01 + track.P[9]*sinv11;

    double k30 = track.P[12]*sinv00 + track.P[13]*sinv10;
    double k31 = track.P[12]*sinv01 + track.P[13]*sinv11;

    // Update state state x = x + K * y
    track.x[0] += k00 * y0 + k01 * y1_correct;
    track.x[1] += k10 * y0 + k11 * y1_correct;
    track.x[2] += k20 * y0 + k21 * y1_correct;
    track.x[3] += k30 * y0 + k31 * y1_correct;

    // Update covariance P = (I - K*H)*P
    // I - K*H
    std::array<double, 16> I_KH = {
        1.0 - k00,      -k01, 0.0, 0.0,
             -k10, 1.0 - k11, 0.0, 0.0,
             -k20,      -k21, 1.0, 0.0,
             -k30,      -k31, 0.0, 1.0
    };

    track.P = mat_mul_4x4(I_KH, track.P);
}

void LaserObstacleDetectorCore::associate_and_update(
    const std::vector<std::tuple<uint8_t, Point2D, std::vector<double>, std::vector<Point2D>>>& detections
)
{
    // Predict
    predict_tracks();

    std::vector<bool> matched_detections(detections.size(), false);
    std::vector<bool> matched_tracks(tracks_.size(), false);

    // Greedy Association
    struct Association
    {
        double dist;
        size_t d_idx;
        size_t t_idx;
    };
    std::vector<Association> associations;

    for (size_t d_idx = 0; d_idx < detections.size(); ++d_idx)
    {
        const auto& det_centroid = std::get<1>(detections[d_idx]);
        for (size_t t_idx = 0; t_idx < tracks_.size(); ++t_idx)
        {
            double dx = det_centroid.x - tracks_[t_idx].x[0];
            double dy = det_centroid.y - tracks_[t_idx].x[1];
            double dist = std::sqrt(dx*dx + dy*dy);

            if (dist < max_association_dist)
            {
                associations.push_back({dist, d_idx, t_idx});
            }
        }
    }

    // Sort by distance ascending
    std::sort(associations.begin(), associations.end(), [](const Association& a, const Association& b) {
        return a.dist < b.dist;
    });

    for (const auto& assoc : associations)
    {
        if (!matched_detections[assoc.d_idx] && !matched_tracks[assoc.t_idx])
        {
            matched_detections[assoc.d_idx] = true;
            matched_tracks[assoc.t_idx] = true;

            // Update Track
            auto& track = tracks_[assoc.t_idx];
            const auto& [shape_type, centroid, dims, polygon] = detections[assoc.d_idx];
            
            update_track_kf(track, centroid);

            track.shape_type = shape_type;
            track.shape_dims = dims;
            track.polygon = polygon;
            track.age += 1;
            track.missed_frames = 0;
        }
    }

    // Manage unmatched tracks
    std::vector<Track> active_tracks;
    for (size_t t_idx = 0; t_idx < tracks_.size(); ++t_idx)
    {
        if (!matched_tracks[t_idx])
        {
            tracks_[t_idx].missed_frames += 1;
        }

        if (tracks_[t_idx].missed_frames <= max_missed_frames)
        {
            active_tracks.push_back(tracks_[t_idx]);
        }
    }
    tracks_ = active_tracks;

    // Manage unmatched detections (spawn new tracks)
    for (size_t d_idx = 0; d_idx < detections.size(); ++d_idx)
    {
        if (!matched_detections[d_idx])
        {
            const auto& [shape_type, centroid, dims, polygon] = detections[d_idx];
            
            Track new_track;
            new_track.id = next_track_id_++;
            new_track.x = {centroid.x, centroid.y, 0.0, 0.0};
            new_track.P = {
                0.1, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            };
            new_track.shape_type = shape_type;
            new_track.shape_dims = dims;
            new_track.polygon = polygon;
            new_track.age = 1;
            new_track.missed_frames = 0;
            new_track.is_confirmed = false;

            tracks_.push_back(new_track);
        }
    }
}

} // namespace autodriver_laser_object_segmentation
