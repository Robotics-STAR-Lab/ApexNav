#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <string>

#include <lkh_mtsp_solver/lkh3_interface.h>
#include <lkh_mtsp_solver/srv/solve_mtsp.hpp>

using std::string;

class TspNode : public rclcpp::Node
{
public:
  TspNode() : Node("tsp_node")
  {
    // Declare and get parameter
    if (!this->has_parameter("exploration/tsp_dir")) {
      this->declare_parameter("exploration/tsp_dir", "null");
    }
    std::string tsp_dir = this->get_parameter("exploration/tsp_dir").as_string();

    mtsp_dir1_ = tsp_dir + "/atsp_tour.par";

    // Create service server
    service_ = this->create_service<lkh_mtsp_solver::srv::SolveMTSP>(
        "/solve_tsp",
        std::bind(&TspNode::mtspCallback, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_WARN(this->get_logger(), "TSP server is ready.");
  }

private:
  void mtspCallback(
      const std::shared_ptr<lkh_mtsp_solver::srv::SolveMTSP::Request> req,
      std::shared_ptr<lkh_mtsp_solver::srv::SolveMTSP::Response> res)
  {
    if (req->prob == 1)
      solveMTSPWithLKH3(mtsp_dir1_.c_str());
    res->empty = 0;
  }

  std::string mtsp_dir1_;
  rclcpp::Service<lkh_mtsp_solver::srv::SolveMTSP>::SharedPtr service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TspNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
