#include "stereo_proc.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

namespace dvrk_stereo {

StereoImageProcessor::StereoImageProcessor() : StereoImageProcessor(cv::INTER_AREA) { }

StereoImageProcessor::StereoImageProcessor(int cv_interpolation_method)
    : desired_image_width(0),
      desired_image_height(0),
      input_camera(""),
      cv_interpolation_method(cv_interpolation_method),
      private_nh("~"),
      transport(public_nh)
{ }

void StereoImageProcessor::init() {
    // Get parameters
    private_nh.param("width", desired_image_width, 0);
    private_nh.param("height", desired_image_height, 0);
    bool camera_specified = private_nh.getParam("camera", input_camera);
    if (!camera_specified) {
        ROS_ERROR("Required parameter '_camera' not specified");
    }

    //auto left_connect_info = std::bind(&StereoImageProcessor::connectInfoCallback, this, left_info_publisher, left_info_subscriber, "left");
    auto left_connect_info = [&](const ros::SingleSubscriberPublisher& _) { connectInfoCallback(left_info_publisher, left_info_subscriber, "left"); };
    auto right_connect_info = [&](const ros::SingleSubscriberPublisher& _) { connectInfoCallback(right_info_publisher, right_info_subscriber, "right"); };
    auto left_connect_image = [&](const image_transport::SingleSubscriberPublisher& _) { connectImageCallback(left_image_publisher, left_image_subscriber, "left"); };
    auto right_connect_image = [&](const image_transport::SingleSubscriberPublisher& _) { connectImageCallback(right_image_publisher, right_image_subscriber, "right"); };

    // Ensure connectCallback isn't entered before camera publishers are set up completely
    std::lock_guard<std::mutex> lock(transport_setup_mutex);
    left_image_publisher = transport.advertise("left/image", 1, left_connect_image, left_connect_image);
    left_info_publisher = public_nh.advertise<sensor_msgs::CameraInfo>("left/camera_info", 1, left_connect_info, left_connect_info);
    right_image_publisher = transport.advertise("right/image", 1, right_connect_image, right_connect_image);
    right_info_publisher = public_nh.advertise<sensor_msgs::CameraInfo>("right/camera_info", 1, right_connect_info, right_connect_info);
}

void StereoImageProcessor::connectInfoCallback(ros::Publisher& publisher, ros::Subscriber& subscriber, std::string side) {
    std::lock_guard<std::mutex> lock(transport_setup_mutex);

    if (publisher.getNumSubscribers() == 0) {
        subscriber.shutdown();
    } else if (!subscriber) {
        // Bind callback to specific left/right publisher
        auto info_callback = [&](const sensor_msgs::CameraInfoConstPtr& msg) { infoCallback(publisher, msg); };
        subscriber = public_nh.subscribe<sensor_msgs::CameraInfo>(input_camera + "/" + side + "/camera_info", 1, info_callback);
    }
}

void StereoImageProcessor::connectImageCallback(image_transport::Publisher& publisher, image_transport::Subscriber& subscriber, std::string side) {
    std::lock_guard<std::mutex> lock(transport_setup_mutex);

    if (publisher.getNumSubscribers() == 0) {
        subscriber.shutdown();
    } else if (!subscriber) {
        // Bind callback to specific left/right publisher
        auto image_callback = [&](const sensor_msgs::ImageConstPtr& msg) { imageCallback(publisher, msg); };
        subscriber = transport.subscribe(input_camera + "/" + side + "/image_rect_color", 1, image_callback);
    }
}

void StereoImageProcessor::infoCallback(ros::Publisher& publisher, const sensor_msgs::CameraInfoConstPtr& info_msg) {
    bool no_desired_size = (desired_image_width == 0) || (desired_image_height == 0);
    bool correct_size = (info_msg->width == desired_image_width) && (info_msg->height == desired_image_height);

    if (no_desired_size || correct_size) {
        publisher.publish(info_msg);
        return;
    }

    double desired_aspect_ratio = static_cast<double>(desired_image_width)/static_cast<double>(desired_image_height);
    double current_aspect_ratio = static_cast<double>(info_msg->width)/static_cast<double>(info_msg->height);
   
    int crop_x = 0;
    int crop_y = 0;
    int width = info_msg->width;
    int height = info_msg->height;
 
    // Want image to be narrower, crop width
    if (desired_aspect_ratio < current_aspect_ratio) {
        width = desired_aspect_ratio*info_msg->height;
        crop_x = (info_msg->width - width)/2;
    // Want image to be shorter, crop height
    } else {
        height = info_msg->width/desired_aspect_ratio;
        crop_y = (info_msg->height - height)/2;
    }

    sensor_msgs::CameraInfoPtr resized_info(new sensor_msgs::CameraInfo(*info_msg));
    double scale_x = static_cast<double>(desired_image_width)/static_cast<double>(width);
    int offset_x = static_cast<int>(-crop_x * scale_x);
    double scale_y = static_cast<double>(desired_image_height)/static_cast<double>(height);
    int offset_y = static_cast<int>(-crop_y * scale_y);
    
    resized_info->width = desired_image_width;
    resized_info->height = desired_image_height;

    // Entries other than 0, 2, 4, 5 are 0.0
    resized_info->K[0] = info_msg->K[0] * scale_x;            // fx
    resized_info->K[2] = info_msg->K[2] * scale_x + offset_x; // cx
    resized_info->K[4] = info_msg->K[4] * scale_y;            // fy
    resized_info->K[5] = info_msg->K[5] * scale_y + offset_y; // cy

    // Entries other than 0, 2, 3, 5, 6 are 0.0
    resized_info->P[0] = info_msg->P[0] * scale_x;            // fx
    resized_info->P[2] = info_msg->P[2] * scale_x + offset_x; // cx
    resized_info->P[3] = info_msg->P[3] * scale_x + offset_x; // Tx
    resized_info->P[5] = info_msg->P[5] * scale_y;            // fy
    resized_info->P[6] = info_msg->P[6] * scale_y + offset_y; // cy

    // Since we've cropped, we can't necessarily respect ROI, so set to full image
    resized_info->roi.x_offset = 0;
    resized_info->roi.y_offset = 0;
    resized_info->roi.width = 0;
    resized_info->roi.height = 0;

    publisher.publish(resized_info);
}

void StereoImageProcessor::imageCallback(image_transport::Publisher& publisher, const sensor_msgs::ImageConstPtr& image_msg) {
    bool no_desired_size = (desired_image_width == 0) || (desired_image_height == 0);
    bool correct_size = (image_msg->width == desired_image_width) && (image_msg->height == desired_image_height);

    if (no_desired_size || correct_size) {
        publisher.publish(image_msg);
        return;
    }

    cv_bridge::CvImageConstPtr in_image;
    cv_bridge::CvImage resized_image;
    resized_image.header = image_msg->header;
    resized_image.encoding = "bgr8";
    
    try {
        in_image = cv_bridge::toCvShare(image_msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'", image_msg->encoding.c_str());
    }

    double desired_aspect_ratio = static_cast<double>(desired_image_width)/static_cast<double>(desired_image_height);
    double current_aspect_ratio = static_cast<double>(image_msg->width)/static_cast<double>(image_msg->height);
   
    cv::Rect ROI;
    int crop_x = 0;
    int crop_y = 0;
    int width = image_msg->width;
    int height = image_msg->height;
 
    // Want image to be narrower, crop width
    if (desired_aspect_ratio < current_aspect_ratio) {
        width = desired_aspect_ratio*image_msg->height;
        crop_x = (image_msg->width - width)/2;
        ROI = cv::Rect(crop_x, 0, width, image_msg->height);
    // Want image to be shorter, crop height
    } else {
        height = image_msg->width/desired_aspect_ratio;
        crop_y = (image_msg->height - height)/2;
        ROI = cv::Rect(0, crop_y, image_msg->width, height);
    }

    // Crop
    cv::Mat cropped = in_image->image(ROI);

    // Scale to desired size while maintaining aspect ratio
    cv::Size size(desired_image_width, desired_image_height);
    cv::resize(cropped, resized_image.image, size, 0.0, 0.0, cv::INTER_CUBIC);

    auto out_msg = resized_image.toImageMsg();
    publisher.publish(out_msg);
}

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "stereo_proc");
    
    dvrk_stereo::StereoImageProcessor stereo_processor;
    stereo_processor.init();

    ros::spin();

    return 0;
}


