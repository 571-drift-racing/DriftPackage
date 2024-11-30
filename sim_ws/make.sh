#!/bin/bash

source /opt/ros/foxy/setup.bash
rm -rf build install log
rosdep install -i --from-path src --rosdistro foxy -y
colcon build
source install/setup.bash
