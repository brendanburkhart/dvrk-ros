#ifndef DVRK_STEREO_PROC
#define DVRK_STEREO_PROC

#include <mutex>
#include <string>

#include <ros/ros.h>
#include <image_transport/image_transport.h>

namespace dvrk_stereo {

class StereoImageProcessor {
public:
    StereoImageProcessor();
    StereoImageProcessor(int cv_interpolation_method);

    void init();
    
    void connectInfoCallback(ros::Publisher& publisher, ros::Subscriber& subscriber, std::string side);
    void connectImageCallback(image_transport::Publisher& publisher, image_transport::Subscriber& subscriber, std::string side);

    void infoCallback(ros::Publisher& publisher, const sensor_msgs::CameraInfoConstPtr& info_msg);
    void imageCallback(image_transport::Publisher& publisher, const sensor_msgs::ImageConstPtr& image_msg);

private:
    int desired_image_width;
    int desired_image_height;
    std::string input_camera;

    int cv_interpolation_method;

    ros::NodeHandle public_nh;
    ros::NodeHandle private_nh;

    image_transport::ImageTransport transport;
    image_transport::Publisher left_image_publisher, right_image_publisher;
    ros::Publisher left_info_publisher, right_info_publisher;
    image_transport::Subscriber left_image_subscriber, right_image_subscriber;
    ros::Subscriber left_info_subscriber, right_info_subscriber;

    std::mutex transport_setup_mutex;
};

}

#endif

