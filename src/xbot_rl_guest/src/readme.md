sudo -s
cd /home/s2/s3_ws
source ./devel/setup.bash
roslaunch robotera_s3_hw robotera_s3_grip_hw.launch

sudo -s
cd /home/s2/s3_ws
source ./devel/setup.bash
roslaunch isrhumanoid2_controllers isrhumanoid2_6dof_arm_controller.launch

sudo -s
cd /home/s2/test_ws
source ./devel/setup.bash
rosrun pytouch_test pytouch_test_node
