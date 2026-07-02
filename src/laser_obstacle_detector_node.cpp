#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <derived_object_msgs/msg/object.hpp>
#include <derived_object_msgs/msg/object_array.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/floating_point_range.hpp>
#include <rcl_interfaces/msg/integer_range.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <vector>
#include <cmath>
#include <cstring>
#include <memory>

#include "autodriver_laser_object_segmentation/laser_obstacle_detector_core.hpp"

namespace autodriver_laser_object_segmentation
{

class LaserObstacleDetectorNode : public rclcpp::Node
{
public:
    explicit LaserObstacleDetectorNode(const rclcpp::NodeOptions& options)
        : Node("laser_obstacle_detector", options)
    {
        // Declare Parameters. Numeric params carry a ParameterDescriptor with min/max/step
        // so rqt_reconfigure renders sliders; every param is re-read at the top of each scan
        // callback, so slider/`ros2 param set` changes apply on the next scan (no restart).
        auto fr = [](double lo, double hi, double step) {
            rcl_interfaces::msg::ParameterDescriptor d;
            rcl_interfaces::msg::FloatingPointRange r;
            r.from_value = lo; r.to_value = hi; r.step = step;
            d.floating_point_range.push_back(r);
            return d;
        };
        auto ir = [](int64_t lo, int64_t hi, uint64_t step) {
            rcl_interfaces::msg::ParameterDescriptor d;
            rcl_interfaces::msg::IntegerRange r;
            r.from_value = lo; r.to_value = hi; r.step = step;
            d.integer_range.push_back(r);
            return d;
        };

        declare_parameter("min_range", 0.1, fr(0.0, 5.0, 0.01));
        declare_parameter("max_range", 10.0, fr(0.0, 30.0, 0.01));
        declare_parameter("beta_incidence_deg", 10.0, fr(0.0, 45.0, 0.1));
        declare_parameter("sigma_r", 0.01, fr(0.0, 0.2, 0.001));
        declare_parameter("min_jump_distance", 0.1, fr(0.0, 2.0, 0.01));
        declare_parameter("max_jump_distance", 1.0, fr(0.0, 5.0, 0.01));
        declare_parameter("min_cluster_points", 3, ir(1, 50, 1));
        declare_parameter("max_cluster_points", 2000, ir(1, 5000, 1));
        declare_parameter("use_convex_hull", true);
        declare_parameter("split_threshold", 0.05, fr(0.0, 0.5, 0.001));
        declare_parameter("max_association_distance", 1.0, fr(0.0, 5.0, 0.01));
        declare_parameter("min_track_age", 3, ir(1, 20, 1));
        declare_parameter("max_missed_frames", 5, ir(0, 30, 1));
        declare_parameter("publish_debug_pointcloud", true);
        declare_parameter("publish_debug_markers", true);
        declare_parameter("use_median_filter", true);
        declare_parameter("association_method", "hungarian");
        declare_parameter("circle_residual_ratio", 0.12, fr(0.0, 1.0, 0.01));
        declare_parameter("max_circle_radius", 1.0, fr(0.0, 5.0, 0.01));
        declare_parameter("corner_angle_min_deg", 65.0, fr(0.0, 180.0, 0.5));
        declare_parameter("corner_angle_max_deg", 115.0, fr(0.0, 180.0, 0.5));
        declare_parameter("shape_smoothing_alpha", 0.5, fr(0.0, 1.0, 0.01));
        declare_parameter("kf_process_noise", 0.1, fr(0.0, 2.0, 0.01));
        declare_parameter("shape_type_hysteresis", 3, ir(1, 20, 1));
        declare_parameter("tracking_frame", "odom");
        declare_parameter("publish_unconfirmed", true);
        declare_parameter("tf_fallback_to_latest", true);
        declare_parameter("tf_latest_max_delay", 2.0, fr(0.0, 10.0, 0.05));
        declare_parameter("enable_tracking", true);
        declare_parameter("publish_object_array", true);

        // Get initial parameters
        double beta_rad = get_parameter("beta_incidence_deg").as_double() * M_PI / 180.0;
        
        core_ = std::make_unique<LaserObstacleDetectorCore>(
            get_parameter("min_range").as_double(),
            get_parameter("max_range").as_double(),
            beta_rad,
            get_parameter("sigma_r").as_double(),
            get_parameter("min_jump_distance").as_double(),
            get_parameter("max_jump_distance").as_double(),
            get_parameter("min_cluster_points").as_int(),
            get_parameter("max_cluster_points").as_int(),
            get_parameter("use_convex_hull").as_bool(),
            get_parameter("split_threshold").as_double(),
            get_parameter("max_association_distance").as_double(),
            get_parameter("min_track_age").as_int(),
            get_parameter("max_missed_frames").as_int(),
            0.1, // dt
            get_parameter("use_median_filter").as_bool(),
            get_parameter("association_method").as_string(),
            get_parameter("circle_residual_ratio").as_double(),
            get_parameter("max_circle_radius").as_double(),
            get_parameter("corner_angle_min_deg").as_double(),
            get_parameter("corner_angle_max_deg").as_double(),
            get_parameter("shape_smoothing_alpha").as_double(),
            get_parameter("kf_process_noise").as_double(),
            static_cast<uint32_t>(get_parameter("shape_type_hysteresis").as_int())
        );

        last_stamp_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());

        // Initialize TF2 buffer and listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Subscriptions
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", rclcpp::SensorDataQoS(), std::bind(&LaserObstacleDetectorNode::scan_callback, this, std::placeholders::_1)
        );

        // Publishers
        obstacles_pub_ = create_publisher<derived_object_msgs::msg::ObjectArray>("obstacles", 10);
        
        publish_pc_ = get_parameter("publish_debug_pointcloud").as_bool();
        if (publish_pc_)
        {
            pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("debug_clusters", 10);
        }

        publish_markers_ = get_parameter("publish_debug_markers").as_bool();
        if (publish_markers_)
        {
            marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("debug_markers", 10);
        }

        // Log parameter changes so live retuning (rqt_reconfigure / `ros2 param set`) gives
        // visible feedback. Registered AFTER declares so it doesn't fire for initial values.
        // The actual application happens in scan_callback, which re-reads every param each scan.
        param_cb_handle_ = add_on_set_parameters_callback(
            std::bind(&LaserObstacleDetectorNode::on_set_parameters, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "C++ Laser Obstacle Detector Node initialized.");
    }

    rcl_interfaces::msg::SetParametersResult on_set_parameters(
        const std::vector<rclcpp::Parameter>& params)
    {
        for (const auto& p : params)
        {
            RCLCPP_INFO(get_logger(), "Parameter update: %s = %s",
                        p.get_name().c_str(), p.value_to_string().c_str());
        }
        // Accept: range validation is enforced separately by each param's descriptor.
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        return result;
    }

private:
    // Look up tracking_frame<-source_frame at the exact scan stamp. On failure (laggy/gappy
    // TF -> extrapolation errors), optionally retry with the latest-available transform
    // (tf2::TimePointZero), accepting it only if no more stale than tf_latest_max_delay
    // seconds. Returns true and fills out_trans if a usable transform was found; false means
    // the caller should fall back to the sensor frame. Warnings are throttled.
    bool lookup_tracking_transform(
        const std::string& tracking_frame,
        const std::string& source_frame,
        const builtin_interfaces::msg::Time& stamp,
        const rclcpp::Time& current_stamp,
        geometry_msgs::msg::TransformStamped& out_trans)
    {
        try
        {
            out_trans = tf_buffer_->lookupTransform(
                tracking_frame, source_frame, stamp, rclcpp::Duration::from_seconds(0.3));
            return true;
        }
        catch (const tf2::TransformException& ex_exact)
        {
            if (!get_parameter("tf_fallback_to_latest").as_bool())
            {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 5000,
                    "TF lookup failed from '%s' to '%s': %s. Falling back to sensor frame.",
                    tracking_frame.c_str(), source_frame.c_str(), ex_exact.what());
                return false;
            }
        }

        // Exact-stamp lookup failed; try the latest-available transform.
        try
        {
            out_trans = tf_buffer_->lookupTransform(
                tracking_frame, source_frame, tf2::TimePointZero);
        }
        catch (const tf2::TransformException& ex_latest)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "TF lookup failed from '%s' to '%s' (exact and latest): %s. "
                "Falling back to sensor frame.",
                tracking_frame.c_str(), source_frame.c_str(), ex_latest.what());
            return false;
        }

        rclcpp::Time tf_time(out_trans.header.stamp);
        double staleness = std::abs(
            static_cast<double>(current_stamp.nanoseconds() - tf_time.nanoseconds()) / 1e9);
        double max_delay = get_parameter("tf_latest_max_delay").as_double();
        if (staleness <= max_delay)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Using latest-available TF '%s'<-'%s', stale by %.2fs "
                "(exact-stamp lookup failed).",
                tracking_frame.c_str(), source_frame.c_str(), staleness);
            return true;
        }

        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Latest TF '%s'<-'%s' is %.2fs stale (> %.2fs); falling back to sensor frame.",
            tracking_frame.c_str(), source_frame.c_str(), staleness, max_delay);
        return false;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Update dt dynamically
        double dt = core_->dt;
        rclcpp::Time current_stamp(msg->header.stamp);
        if (last_stamp_.nanoseconds() > 0)
        {
            double calc_dt = (current_stamp.nanoseconds() - last_stamp_.nanoseconds()) / 1e9;
            if (calc_dt > 0.001)
            {
                dt = calc_dt;
            }
        }
        last_stamp_ = current_stamp;

        // Clamp dt to [1e-3, 1.0]
        dt = std::max(1.0e-3, std::min(dt, 1.0));

        // Look up transform for ego-motion-aware tracking
        std::vector<double> sensor_pose;
        std::string tracking_frame = get_parameter("tracking_frame").as_string();
        std::string out_frame_id = msg->header.frame_id;

        if (!tracking_frame.empty())
        {
            geometry_msgs::msg::TransformStamped trans;
            if (lookup_tracking_transform(tracking_frame, msg->header.frame_id,
                                          msg->header.stamp, current_stamp, trans))
            {
                double tx = trans.transform.translation.x;
                double ty = trans.transform.translation.y;

                // Quaternion to yaw
                double qx = trans.transform.rotation.x;
                double qy = trans.transform.rotation.y;
                double qz = trans.transform.rotation.z;
                double qw = trans.transform.rotation.w;

                double siny_cosp = 2.0 * (qw * qz + qx * qy);
                double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
                double yaw = std::atan2(siny_cosp, cosy_cosp);

                sensor_pose = {tx, ty, yaw};
                out_frame_id = tracking_frame;
            }
        }

        // Synchronize core parameters
        core_->min_range = get_parameter("min_range").as_double();
        core_->max_range = get_parameter("max_range").as_double();
        core_->beta = get_parameter("beta_incidence_deg").as_double() * M_PI / 180.0;
        core_->sigma_r = get_parameter("sigma_r").as_double();
        core_->min_jump_distance = get_parameter("min_jump_distance").as_double();
        core_->max_jump_distance = get_parameter("max_jump_distance").as_double();
        core_->min_cluster_points = get_parameter("min_cluster_points").as_int();
        core_->max_cluster_points = get_parameter("max_cluster_points").as_int();
        core_->use_convex_hull = get_parameter("use_convex_hull").as_bool();
        core_->split_threshold = get_parameter("split_threshold").as_double();
        core_->max_association_dist = get_parameter("max_association_distance").as_double();
        core_->min_track_age = get_parameter("min_track_age").as_int();
        core_->max_missed_frames = get_parameter("max_missed_frames").as_int();
        core_->use_median_filter = get_parameter("use_median_filter").as_bool();
        core_->association_method = get_parameter("association_method").as_string();
        core_->circle_residual_ratio = get_parameter("circle_residual_ratio").as_double();
        core_->max_circle_radius = get_parameter("max_circle_radius").as_double();
        core_->corner_angle_min_deg = get_parameter("corner_angle_min_deg").as_double();
        core_->corner_angle_max_deg = get_parameter("corner_angle_max_deg").as_double();
        core_->shape_smoothing_alpha = get_parameter("shape_smoothing_alpha").as_double();
        core_->kf_process_noise = get_parameter("kf_process_noise").as_double();
        core_->shape_type_hysteresis = static_cast<uint32_t>(get_parameter("shape_type_hysteresis").as_int());

        bool enable_tracking = get_parameter("enable_tracking").as_bool();
        bool publish_object_array = get_parameter("publish_object_array").as_bool();

        // Core processing
        auto [active_tracks, detections, clusters] = core_->process(
            msg->ranges,
            msg->angle_min,
            msg->angle_increment,
            dt,
            sensor_pose,
            enable_tracking
        );

        bool publish_unconfirmed = get_parameter("publish_unconfirmed").as_bool();
        std::vector<Track> tracks_to_publish;
        for (const auto& track : active_tracks)
        {
            if (publish_unconfirmed || track.is_confirmed)
            {
                tracks_to_publish.push_back(track);
            }
        }

        std_msgs::msg::Header out_header;
        out_header.stamp = msg->header.stamp;
        out_header.frame_id = out_frame_id;

        // 1. Publish Obstacles
        if (publish_object_array && enable_tracking)
        {
            derived_object_msgs::msg::ObjectArray obj_array;
            obj_array.header = out_header;

            for (const auto& track : tracks_to_publish)
            {
                derived_object_msgs::msg::Object obj;
                obj.header = out_header;
                obj.id = track.id;
                obj.detection_level = track.is_confirmed ? 
                    static_cast<uint8_t>(derived_object_msgs::msg::Object::OBJECT_TRACKED) : 
                    static_cast<uint8_t>(derived_object_msgs::msg::Object::OBJECT_DETECTED);
                obj.object_classified = false;

                // Position
                obj.pose.position.x = track.x[0];
                obj.pose.position.y = track.x[1];
                obj.pose.position.z = 0.0;

                // Orientation
                if (track.shape_type == 1) // BOX
                {
                    double yaw = track.shape_dims[2];
                    obj.pose.orientation.x = 0.0;
                    obj.pose.orientation.y = 0.0;
                    obj.pose.orientation.z = std::sin(yaw * 0.5);
                    obj.pose.orientation.w = std::cos(yaw * 0.5);
                }
                else
                {
                    obj.pose.orientation.x = 0.0;
                    obj.pose.orientation.y = 0.0;
                    obj.pose.orientation.z = 0.0;
                    obj.pose.orientation.w = 1.0;
                }

                // Velocity
                obj.twist.linear.x = track.x[2];
                obj.twist.linear.y = track.x[3];
                obj.twist.linear.z = 0.0;
                obj.twist.angular.x = 0.0;
                obj.twist.angular.y = 0.0;
                obj.twist.angular.z = 0.0;

                // Shape representation
                if (track.shape_type == 0) // CIRCLE
                {
                    obj.shape.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
                    obj.shape.dimensions = {0.5, track.shape_dims[0]}; // [height, radius]
                }
                else if (track.shape_type == 1) // BOX
                {
                    obj.shape.type = shape_msgs::msg::SolidPrimitive::BOX;
                    obj.shape.dimensions = {track.shape_dims[0], track.shape_dims[1], 0.5}; // [x, y, z]
                }

                // Polygon boundary
                for (const auto& pt : track.polygon)
                {
                    geometry_msgs::msg::Point32 pt32;
                    pt32.x = static_cast<float>(pt.x);
                    pt32.y = static_cast<float>(pt.y);
                    pt32.z = 0.0f;
                    obj.polygon.points.push_back(pt32);
                }

                obj_array.objects.push_back(obj);
            }

            obstacles_pub_->publish(obj_array);
        }

        // 2. Publish Debug Clusters PointCloud2 (debug pointcloud stays in the sensor frame)
        if (publish_pc_ && !clusters.empty())
        {
            auto pc_msg = make_color_pc2(msg->header, clusters);
            pc_pub_->publish(pc_msg);
        }

        // 3. Publish Debug Markers. With tracking on, markers come from tracks; with tracking
        // off there are no tracks, so render the raw per-frame detections instead.
        if (publish_markers_)
        {
            if (enable_tracking)
            {
                publish_debug_markers(out_header, tracks_to_publish);
            }
            else
            {
                publish_debug_markers_detections(out_header, detections);
            }
        }
    }

    sensor_msgs::msg::PointCloud2 make_color_pc2(
        const std_msgs::msg::Header& header,
        const std::vector<std::vector<Point2D>>& clusters
    )
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.header = header;
        msg.height = 1;

        msg.fields.resize(4);
        msg.fields[0].name = "x";
        msg.fields[0].offset = 0;
        msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[0].count = 1;

        msg.fields[1].name = "y";
        msg.fields[1].offset = 4;
        msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[1].count = 1;

        msg.fields[2].name = "z";
        msg.fields[2].offset = 8;
        msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[2].count = 1;

        msg.fields[3].name = "rgb";
        msg.fields[3].offset = 12;
        msg.fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
        msg.fields[3].count = 1;

        msg.is_bigendian = false;
        msg.point_step = 16;
        msg.is_dense = true;

        size_t total_points = 0;
        for (const auto& c : clusters)
        {
            total_points += c.size();
        }
        msg.width = total_points;
        msg.row_step = msg.point_step * msg.width;
        msg.data.resize(msg.row_step);

        uint32_t colors[] = {
            0xFFFF0000, // Red
            0xFF00FF00, // Green
            0xFF0000FF, // Blue
            0xFFFF00FF, // Magenta
            0xFF00FFFF, // Cyan
            0xFFFFFF00, // Yellow
            0xFFFF8000, // Orange
            0xFF8000FF, // Purple
            0xFF00FF80, // Lime
            0xFFFF0080, // Deep Pink
            0xFF80FF00, // Yellow-Green
            0xFF0080FF  // Sky Blue
        };
        size_t num_colors = sizeof(colors) / sizeof(colors[0]);

        uint8_t* ptr = msg.data.data();
        size_t color_idx = 0;
        for (const auto& cluster : clusters)
        {
            uint32_t rgb = colors[color_idx % num_colors];
            color_idx++;

            for (const auto& p : cluster)
            {
                float x_f = static_cast<float>(p.x);
                float y_f = static_cast<float>(p.y);
                float z_f = 0.0f;

                std::memcpy(ptr, &x_f, 4);
                std::memcpy(ptr + 4, &y_f, 4);
                std::memcpy(ptr + 8, &z_f, 4);
                std::memcpy(ptr + 12, &rgb, 4);

                ptr += 16;
            }
        }

        return msg;
    }

    void publish_debug_markers(const std_msgs::msg::Header& header, const std::vector<Track>& tracks)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        // Clear all previous markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        const int SHAPE_ID = 1000;
        const int VEL_ID = 2000;
        const int TEXT_ID = 3000;
        const int BORDER_ID = 4000;

        for (const auto& track : tracks)
        {
            // Track ID specific color
            double r = static_cast<double>((13 * track.id) % 256) / 255.0;
            double g = static_cast<double>((57 * track.id) % 256) / 255.0;
            double b = static_cast<double>((127 * track.id) % 256) / 255.0;
            
            std_msgs::msg::ColorRGBA color;
            color.r = r;
            color.g = g;
            color.b = b;
            color.a = 0.7;

            // --- 1. Geometric Shape (Cylinder/Cube) ---
            visualization_msgs::msg::Marker shape_marker;
            shape_marker.header = header;
            shape_marker.ns = "shapes";
            shape_marker.id = SHAPE_ID + track.id;
            shape_marker.color = color;
            shape_marker.pose.position.x = track.x[0];
            shape_marker.pose.position.y = track.x[1];
            shape_marker.pose.position.z = 0.0;

            if (track.shape_type == 0) // CIRCLE
            {
                shape_marker.type = visualization_msgs::msg::Marker::CYLINDER;
                shape_marker.scale.x = track.shape_dims[0] * 2.0;
                shape_marker.scale.y = track.shape_dims[0] * 2.0;
                shape_marker.scale.z = 0.2;
                shape_marker.pose.orientation.w = 1.0;
                shape_marker.action = visualization_msgs::msg::Marker::ADD;
                marker_array.markers.push_back(shape_marker);
            }
            else if (track.shape_type == 1) // BOX
            {
                shape_marker.type = visualization_msgs::msg::Marker::CUBE;
                shape_marker.scale.x = track.shape_dims[0];
                shape_marker.scale.y = track.shape_dims[1];
                shape_marker.scale.z = 0.2;

                double yaw = track.shape_dims[2];
                shape_marker.pose.orientation.z = std::sin(yaw * 0.5);
                shape_marker.pose.orientation.w = std::cos(yaw * 0.5);
                shape_marker.action = visualization_msgs::msg::Marker::ADD;
                marker_array.markers.push_back(shape_marker);
            }

            // --- 2. Outline / Polygon Border ---
            visualization_msgs::msg::Marker border_marker;
            border_marker.header = header;
            border_marker.ns = "outlines";
            border_marker.id = BORDER_ID + track.id;
            border_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            border_marker.scale.x = 0.03;
            border_marker.color.r = r;
            border_marker.color.g = g;
            border_marker.color.b = b;
            border_marker.color.a = 1.0;
            border_marker.pose.orientation.w = 1.0;
            border_marker.action = visualization_msgs::msg::Marker::ADD;

            // For primitive shapes, regenerate the outline from the tracked
            // (KF-smoothed) center + smoothed dims so it coincides with the filled
            // shape marker (the raw per-frame polygon otherwise wobbles off-center).
            // For line/corner/hull, use the raw detection polygon.
            if (track.shape_type == 0) // CIRCLE: ring at smoothed radius around track center
            {
                double cx = track.x[0], cy = track.x[1], rad = track.shape_dims[0];
                for (int k = 0; k <= 16; ++k)
                {
                    double a = 2.0 * M_PI * (k % 16) / 16.0;
                    geometry_msgs::msg::Point gp;
                    gp.x = cx + rad * std::cos(a);
                    gp.y = cy + rad * std::sin(a);
                    gp.z = 0.0;
                    border_marker.points.push_back(gp);
                }
            }
            else if (track.shape_type == 1) // BOX: rectangle from smoothed dims/yaw
            {
                double cx = track.x[0], cy = track.x[1];
                double hl = track.shape_dims[0] * 0.5, hw = track.shape_dims[1] * 0.5;
                double yaw = track.shape_dims[2];
                double c = std::cos(yaw), s = std::sin(yaw);
                const double lx[5] = {-hl,  hl, hl, -hl, -hl};
                const double ly[5] = {-hw, -hw, hw,  hw, -hw};
                for (int k = 0; k < 5; ++k)
                {
                    geometry_msgs::msg::Point gp;
                    gp.x = cx + lx[k] * c - ly[k] * s;
                    gp.y = cy + lx[k] * s + ly[k] * c;
                    gp.z = 0.0;
                    border_marker.points.push_back(gp);
                }
            }
            else // LINE / CORNER: raw detection polygon
            {
                for (const auto& pt : track.polygon)
                {
                    geometry_msgs::msg::Point gp;
                    gp.x = pt.x;
                    gp.y = pt.y;
                    gp.z = 0.0;
                    border_marker.points.push_back(gp);
                }
            }

            if (border_marker.points.size() > 1)
            {
                marker_array.markers.push_back(border_marker);
            }

            // --- 3. Velocity Arrow ---
            double vel_norm = std::sqrt(track.x[2] * track.x[2] + track.x[3] * track.x[3]);
            if (vel_norm > 0.1)
            {
                visualization_msgs::msg::Marker vel_marker;
                vel_marker.header = header;
                vel_marker.ns = "velocity";
                vel_marker.id = VEL_ID + track.id;
                vel_marker.type = visualization_msgs::msg::Marker::ARROW;
                vel_marker.scale.x = 0.04; // shaft
                vel_marker.scale.y = 0.08; // head
                vel_marker.scale.z = 0.1;  // height
                vel_marker.color.r = 1.0;
                vel_marker.color.g = 0.0;
                vel_marker.color.b = 0.0;
                vel_marker.color.a = 0.9;
                vel_marker.action = visualization_msgs::msg::Marker::ADD;

                geometry_msgs::msg::Point p_start, p_end;
                p_start.x = track.x[0];
                p_start.y = track.x[1];
                p_start.z = 0.0;

                p_end.x = track.x[0] + track.x[2];
                p_end.y = track.x[1] + track.x[3];
                p_end.z = 0.0;

                vel_marker.points = {p_start, p_end};
                marker_array.markers.push_back(vel_marker);
            }

            // --- 4. Floating Labels ---
            visualization_msgs::msg::Marker text_marker;
            text_marker.header = header;
            text_marker.ns = "labels";
            text_marker.id = TEXT_ID + track.id;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.scale.z = 0.2;
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            text_marker.pose.position.x = track.x[0];
            text_marker.pose.position.y = track.x[1];
            text_marker.pose.position.z = 0.3;
            text_marker.pose.orientation.w = 1.0;
            text_marker.action = visualization_msgs::msg::Marker::ADD;

            char buf[64];
            if (vel_norm > 0.1)
                std::snprintf(buf, sizeof(buf), "ID:%d %.1f", track.id, vel_norm);
            else
                std::snprintf(buf, sizeof(buf), "ID:%d", track.id);
            text_marker.text = buf;

            marker_array.markers.push_back(text_marker);
        }

        marker_pub_->publish(marker_array);
    }

    // Render per-frame detections (used when enable_tracking=false, so there are no tracks).
    // Detections carry no id/velocity: color/id are synthesized from the vector index and no
    // velocity arrows are drawn. Circle/box outlines are regenerated from centroid+dims
    // (matching the tracked-marker path); line/corner use the raw polygon. Keep in sync with
    // publish_debug_markers() and the Python node's publish_debug_markers_detections().
    void publish_debug_markers_detections(
        const std_msgs::msg::Header& header,
        const std::vector<Detection>& detections)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        // Clear all previous markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        const int SHAPE_ID = 1000;
        const int BORDER_ID = 4000;

        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto& det = detections[i];
            uint8_t shape_type = std::get<0>(det);
            const Point2D& centroid = std::get<1>(det);
            const std::vector<double>& dims = std::get<2>(det);
            const std::vector<Point2D>& polygon = std::get<3>(det);
            double cx = centroid.x, cy = centroid.y;
            int idx = static_cast<int>(i);

            // Color based on detection index (no track id available)
            double r = static_cast<double>((13 * idx) % 256) / 255.0;
            double g = static_cast<double>((57 * idx) % 256) / 255.0;
            double b = static_cast<double>((127 * idx) % 256) / 255.0;

            std_msgs::msg::ColorRGBA color;
            color.r = r;
            color.g = g;
            color.b = b;
            color.a = 0.7;

            // --- 1. Geometric Shape (Cylinder/Cube) ---
            visualization_msgs::msg::Marker shape_marker;
            shape_marker.header = header;
            shape_marker.ns = "shapes";
            shape_marker.id = SHAPE_ID + idx;
            shape_marker.color = color;
            shape_marker.pose.position.x = cx;
            shape_marker.pose.position.y = cy;
            shape_marker.pose.position.z = 0.0;

            if (shape_type == 0) // CIRCLE
            {
                shape_marker.type = visualization_msgs::msg::Marker::CYLINDER;
                shape_marker.scale.x = dims[0] * 2.0;
                shape_marker.scale.y = dims[0] * 2.0;
                shape_marker.scale.z = 0.2;
                shape_marker.pose.orientation.w = 1.0;
                shape_marker.action = visualization_msgs::msg::Marker::ADD;
                marker_array.markers.push_back(shape_marker);
            }
            else if (shape_type == 1) // BOX
            {
                shape_marker.type = visualization_msgs::msg::Marker::CUBE;
                shape_marker.scale.x = dims[0];
                shape_marker.scale.y = dims[1];
                shape_marker.scale.z = 0.2;
                double yaw = dims[2];
                shape_marker.pose.orientation.z = std::sin(yaw * 0.5);
                shape_marker.pose.orientation.w = std::cos(yaw * 0.5);
                shape_marker.action = visualization_msgs::msg::Marker::ADD;
                marker_array.markers.push_back(shape_marker);
            }

            // --- 2. Outline / Polygon Border ---
            visualization_msgs::msg::Marker border_marker;
            border_marker.header = header;
            border_marker.ns = "outlines";
            border_marker.id = BORDER_ID + idx;
            border_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            border_marker.scale.x = 0.03;
            border_marker.color.r = r;
            border_marker.color.g = g;
            border_marker.color.b = b;
            border_marker.color.a = 1.0;
            border_marker.pose.orientation.w = 1.0;
            border_marker.action = visualization_msgs::msg::Marker::ADD;

            if (shape_type == 0) // CIRCLE: ring at radius around centroid
            {
                double rad = dims[0];
                for (int k = 0; k <= 16; ++k)
                {
                    double a = 2.0 * M_PI * (k % 16) / 16.0;
                    geometry_msgs::msg::Point gp;
                    gp.x = cx + rad * std::cos(a);
                    gp.y = cy + rad * std::sin(a);
                    gp.z = 0.0;
                    border_marker.points.push_back(gp);
                }
            }
            else if (shape_type == 1) // BOX: rectangle from dims/yaw
            {
                double hl = dims[0] * 0.5, hw = dims[1] * 0.5;
                double yaw = dims[2];
                double c = std::cos(yaw), s = std::sin(yaw);
                const double lx[5] = {-hl,  hl, hl, -hl, -hl};
                const double ly[5] = {-hw, -hw, hw,  hw, -hw};
                for (int k = 0; k < 5; ++k)
                {
                    geometry_msgs::msg::Point gp;
                    gp.x = cx + lx[k] * c - ly[k] * s;
                    gp.y = cy + lx[k] * s + ly[k] * c;
                    gp.z = 0.0;
                    border_marker.points.push_back(gp);
                }
            }
            else // LINE / CORNER: raw detection polygon
            {
                for (const auto& pt : polygon)
                {
                    geometry_msgs::msg::Point gp;
                    gp.x = pt.x;
                    gp.y = pt.y;
                    gp.z = 0.0;
                    border_marker.points.push_back(gp);
                }
            }

            if (border_marker.points.size() > 1)
            {
                marker_array.markers.push_back(border_marker);
            }
        }

        marker_pub_->publish(marker_array);
    }

    std::unique_ptr<LaserObstacleDetectorCore> core_;
    rclcpp::Time last_stamp_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // Publishers
    rclcpp::Publisher<derived_object_msgs::msg::ObjectArray>::SharedPtr obstacles_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    bool publish_pc_;
    bool publish_markers_;

    // Keeps the on-set-parameters callback alive for the node's lifetime.
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

} // namespace autodriver_laser_object_segmentation

// Register node component
RCLCPP_COMPONENTS_REGISTER_NODE(autodriver_laser_object_segmentation::LaserObstacleDetectorNode)
