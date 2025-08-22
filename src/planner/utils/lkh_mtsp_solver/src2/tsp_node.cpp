#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <string>

#include <lkh_mtsp_solver/lkh3_interface.h>
#include <lkh_mtsp_solver/SolveMTSP.h>

using std::string;

std::string mtsp_dir1_;

bool mtspCallback(
    lkh_mtsp_solver::SolveMTSP::Request& req, lkh_mtsp_solver::SolveMTSP::Response& res)
{
  if (req.prob == 1)
    solveMTSPWithLKH3(mtsp_dir1_.c_str());
  return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tsp_node");
  ros::NodeHandle nh("~");

  // Read mtsp file dir
  std::string tsp_dir;
  nh.param("exploration/tsp_dir", tsp_dir, std::string("null"));

  mtsp_dir1_ = tsp_dir + "/atsp_tour.par";

  string service_name = "/solve_tsp";
  ros::ServiceServer mtsp_server = nh.advertiseService(service_name, mtspCallback);

  ROS_WARN("TSP server is ready.");
  ros::spin();

  return 1;
}
