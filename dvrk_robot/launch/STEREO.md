# dVRK ROS 1 Stereo Camera Calibration

## Install
- Install gscam ROS package using instructions from [dvrk_robot/video.md](https://github.com/jhu-dvrk/dvrk-ros/blob/devel/dvrk_robot/video.md#ros-ubuntu-packages-vs-build-from-source), "ROS Ubuntu packages vs build from source".
  - Make sure gscam 1.0 dependencies are installed, see [hap1961/gscam/README.md](https://github.com/hap1961/gscam/tree/noetic-devel#10x-experimental)

## Stereo calibration

- Launch stereo pipeline: `roslaunch dvrk_robot rviz_stereo_pipeline.launch rig_name:=jhu_daVinci`
- Run a stereo camera calibration, see [ROS camera_calibration](https://wiki.ros.org/camera_calibration).
  - Make sure to add flag `--approximate=0.05` when running `cameracalibrator.py`.
  - Depending on board used, your command should be something like `rosrun camera_calibration cameracalibrator.py --size 12x10 --square 0.45 --aproximate=0.05 right:=/jhu_daVinci/right/image_raw left:=/jhu_daVinci/left/image_raw left_camera:=/jhu_daVinci/left right_camera:=/jhu_daVinci/right`

## Running stereo pipeline

- Launch stereo pipeline: `roslaunch dvrk_robot rviz_stereo_pipeline.launch rig_name:=jhu_daVinci`

RViz should open and display the stereo cameras as separate views, with the test marker visible. If you want to view the stereo cameras in an existing RViz set up, add two cameras to the display, set their input source to `jhu_daVinci/<side>/image_rect_color`, set `Overlay Alpha` to 0.0, and disable visibility of anything you don't won't the cameras to render, e.g. TF transforms, or the grid.
