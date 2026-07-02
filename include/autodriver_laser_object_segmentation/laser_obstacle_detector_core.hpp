#ifndef AUTODRIVER_LASER_OBJECT_SEGMENTATION__LASER_OBSTACLE_DETECTOR_CORE_HPP_
#define AUTODRIVER_LASER_OBJECT_SEGMENTATION__LASER_OBSTACLE_DETECTOR_CORE_HPP_

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <tuple>

namespace autodriver_laser_object_segmentation
{

struct Point2D
{
    double x;
    double y;
};

struct Track
{
    uint32_t id;
    std::array<double, 4> x; // State: [px, py, vx, vy]
    std::array<double, 16> P; // Covariance 4x4 flattened
    uint8_t shape_type; // 0: CIRCLE, 1: BOX, 2: LINE, 3: CORNER
    std::vector<double> shape_dims; // dimensions: [r] or [len, width, yaw]
    std::vector<Point2D> polygon; // Boundary profile
    uint32_t age;
    uint32_t missed_frames;
    bool is_confirmed;
    int pending_type = -1;        // candidate shape type awaiting hysteresis confirmation (-1 = none)
    uint32_t pending_count = 0;   // consecutive frames the candidate type has been seen
};

using Detection = std::tuple<uint8_t, Point2D, std::vector<double>, std::vector<Point2D>>;

class LaserObstacleDetectorCore
{
public:
    LaserObstacleDetectorCore(
        double min_range = 0.1,
        double max_range = 10.0,
        double beta_incidence_rad = 0.1745, // 10 degrees
        double sigma_r = 0.01,
        double min_jump_distance = 0.1,
        double max_jump_distance = 1.0,
        uint32_t min_cluster_points = 3,
        uint32_t max_cluster_points = 2000,
        bool use_convex_hull = true,
        double split_threshold = 0.05,
        double max_association_distance = 1.0,
        uint32_t min_track_age = 3,
        uint32_t max_missed_frames = 5,
        double dt = 0.1,
        bool use_median_filter = true,
        std::string association_method = "hungarian",
        double circle_residual_ratio = 0.12,
        double max_circle_radius = 1.0,
        double corner_angle_min_deg = 65.0,
        double corner_angle_max_deg = 115.0,
        double shape_smoothing_alpha = 0.5,
        double kf_process_noise = 0.5,
        uint32_t shape_type_hysteresis = 1
    );

    ~LaserObstacleDetectorCore() = default;

    // Parameters (can be modified directly)
    double min_range;
    double max_range;
    double beta;
    double sigma_r;
    double min_jump_distance;
    double max_jump_distance;
    uint32_t min_cluster_points;
    uint32_t max_cluster_points;
    bool use_convex_hull;
    double split_threshold;
    double max_association_dist;
    uint32_t min_track_age;
    uint32_t max_missed_frames;
    double dt;
    bool use_median_filter;
    std::string association_method;
    double circle_residual_ratio;
    double max_circle_radius;
    double corner_angle_min_deg;
    double corner_angle_max_deg;
    double shape_smoothing_alpha;
    double kf_process_noise;
    uint32_t shape_type_hysteresis;

    // Core Processing Pipeline
    std::tuple<std::vector<Track>, std::vector<Detection>, std::vector<std::vector<Point2D>>> process(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_increment,
        double dt,
        const std::vector<double>& sensor_pose = {},
        bool enable_tracking = true
    );

    // Helper functions exposed for C-linkage testing
    static std::vector<Point2D> convex_hull_jarvis(const std::vector<Point2D>& points);
    static void fit_obb(const std::vector<Point2D>& points, Point2D& center, double& length, double& width, double& yaw);
    std::vector<int> solve_hungarian(const std::vector<std::vector<double>>& cost_matrix);

private:
    std::vector<Track> tracks_;
    uint32_t next_track_id_;

    // Helper functions
    std::pair<std::vector<Point2D>, std::vector<size_t>> preprocess_scan(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_increment
    );

    std::vector<std::vector<Point2D>> cluster_points(
        const std::vector<Point2D>& points,
        const std::vector<size_t>& valid_indices,
        double angle_increment
    );

    void fit_shape(
        const std::vector<Point2D>& cluster,
        uint8_t& shape_type,
        Point2D& centroid,
        std::vector<double>& dims,
        std::vector<Point2D>& polygon
    );

    // Geometry & Shape fitting
    static void fit_circle_kasa(const std::vector<Point2D>& points, Point2D& center, double& radius);
    static double distance_to_line(const Point2D& p, const Point2D& p1, const Point2D& p2);
    std::vector<std::vector<Point2D>> split_and_merge(const std::vector<Point2D>& points);

    // Tracking helpers
    void predict_tracks(double dt);
    void associate_and_update(
        const std::vector<Detection>& detections,
        double dt
    );
    void update_track_kf(Track& track, const Point2D& detection_centroid, double dt);
    void smooth_shape(Track& track, uint8_t shape_type, const std::vector<double>& dims);
    // Applies shape-type hysteresis, then (on accept) smooths dims and adopts the
    // new type/polygon. Called from both association branches so they can't drift.
    void update_track_shape(Track& track, uint8_t shape_type,
                            const std::vector<double>& dims, const std::vector<Point2D>& polygon);
};

} // namespace autodriver_laser_object_segmentation

extern "C" {
    void test_convex_hull(const double* points_x, const double* points_y, int n, double* hull_x, double* hull_y, int* hull_n);
    void test_obb(const double* points_x, const double* points_y, int n, double* cx, double* cy, double* length, double* width, double* yaw);
    void test_hungarian(const double* cost_matrix, int rows, int cols, int* row_ind, int* col_ind, int* count);
}

#endif // AUTODRIVER_LASER_OBJECT_SEGMENTATION__LASER_OBSTACLE_DETECTOR_CORE_HPP_
