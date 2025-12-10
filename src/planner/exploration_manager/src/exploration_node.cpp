#include <ros/ros.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_fsm_traj.h>

#include <exploration_manager/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace apexnav_planner;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "apexnav_node");
  ros::NodeHandle nh("~");

  // Check if real-world mode
  bool is_real_world = false;
  nh.param("is_real_world", is_real_world, false);

  if (is_real_world) {
    ROS_INFO("========================================");
    ROS_INFO("  Starting in REAL WORLD mode");
    ROS_INFO("========================================");
    ExplorationFSMReal expl_fsm;
    expl_fsm.init(nh);
    ros::Duration(1.0).sleep();
    ros::spin();
  }
  else {
    ROS_INFO("========================================");
    ROS_INFO("  Starting in SIMULATION mode");
    ROS_INFO("========================================");
    ExplorationFSM expl_fsm;
    expl_fsm.init(nh);
    ros::Duration(1.0).sleep();
    ros::spin();
  }

  return 0;
}
