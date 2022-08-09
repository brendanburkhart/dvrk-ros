# dVRK ROS 1 Stereo Camera Calibration

## Install
- Install gscam ROS package using instructions from [dvrk_robot/video.md](https://github.com/jhu-dvrk/dvrk-ros/blob/devel/dvrk_robot/video.md#ros-ubuntu-packages-vs-build-from-source), "ROS Ubuntu packages vs build from source".
  - Make sure gscam 1.0 dependencies are installed, see [hap1961/gscam/README.md](https://github.com/hap1961/gscam/tree/noetic-devel#10x-experimental)

## Stereo calibration

- Launch stereo pipeline: `roslaunch dvrk_robot rviz_stereo_pipeline.launch rig_name:=jhu_daVinci`
- Run a stereo camera calibration, see [ROS camera_calibration](https://wiki.ros.org/camera_calibration).
  - Make sure to add flag `--approximate=0.05` when running `cameracalibrator.py`. Since the two gscam nodes aren't synchronized, this is needed to tell the camera calibration to consider frames to be simultaneous even if the timestamps aren't quite identical.
  - Depending on board used, your command should be something like `rosrun camera_calibration cameracalibrator.py --size 12x10 --square 0.45 --approximate=0.05 right:=/jhu_daVinci/stereo_raw/right/image_raw left:=/jhu_daVinci/stereo_raw/left/image_raw left_camera:=/jhu_daVinci/stereo_raw/left right_camera:=/jhu_daVinci/stereo_raw/right`

## Running stereo pipeline

- Launch stereo pipeline: `roslaunch dvrk_robot rviz_stereo_pipeline.launch rig_name:=jhu_daVinci`

If calibration was performed, this will, in addition to the default `<rig_name>/stereo_raw/<left or right>/image_raw>`, provide `<rig_name>/stereo_processed/<left or right>/image` which are colorized and rectified/undistorted images.

RViz should open and display the stereo cameras as separate views, with the test marker visible. If you want to view the stereo cameras in an existing RViz set up, add two cameras to the display, set their input source to `jhu_daVinci/stereo_processed/<side>/image`, set `Overlay Alpha` to 0.0, and disable visibility of anything you don't won't the cameras to render, e.g. TF transforms, or the grid.

The `rviz_stereo_pipeline.launch` launch file publishes a base transform named `<rig_name>_stereo_base` that the stereo cameras will positions themselves relative to. If you want to change the position of the cameras, simply modify the `static_transform_publisher` args in the launch file, or delete that node and publish the desired transforms yourself.

Once camera calibration has been done, camera intrinsics, extrinics, and distortion models are published on `<rig_name>/stereo_processed/<side>/camera_info` - the `stereo_image_proc` and `dvrk_stereo` nodes will automatically use them to undistort and rectifiy the images, and publish these to `<rig_name>/stereo_processed/<side>/image`. The camera frame ID (`<rig_name>_stereo_base`) is also included in the camera info, and RViz will automatically position the cameras relative to this base - the *left* camera will be replaced at this frame, the *right* camera will be offset appropriately based on their relative position measured during calibration.
