#include <ros/ros.h>
#include <exploration_manager/exploration_fsm.h>

#include <exploration_manager/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace apexnav_planner;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "apexnav_node");
  ros::NodeHandle nh("~");

  ExplorationFSM expl_fsm;
  expl_fsm.init(nh);

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
