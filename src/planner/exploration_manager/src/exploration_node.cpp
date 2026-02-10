#include <rclcpp/rclcpp.hpp>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_fsm_traj.h>

#include <exploration_manager/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace apexnav_planner;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  // Create node
  auto node = std::make_shared<rclcpp::Node>("apexnav_node");

  // Check if real-world mode
  node->declare_parameter("is_real_world", false);
  bool is_real_world = node->get_parameter("is_real_world").as_bool();

  // Use MultiThreadedExecutor to allow service calls from within callbacks
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  if (is_real_world) {
    RCLCPP_INFO(node->get_logger(), "========================================");
    RCLCPP_INFO(node->get_logger(), "  Starting in REAL WORLD mode");
    RCLCPP_INFO(node->get_logger(), "========================================");
    ExplorationFSMReal expl_fsm;
    expl_fsm.init(node);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    executor.spin();
  }
  else {
    RCLCPP_INFO(node->get_logger(), "========================================");
    RCLCPP_INFO(node->get_logger(), "  Starting in SIMULATION mode");
    RCLCPP_INFO(node->get_logger(), "========================================");
    ExplorationFSM expl_fsm;
    expl_fsm.init(node);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    executor.spin();
  }

  rclcpp::shutdown();
  return 0;
}
