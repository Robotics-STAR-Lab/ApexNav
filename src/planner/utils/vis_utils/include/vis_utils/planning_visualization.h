#ifndef _PLANNING_VISUALIZATION_H_
#define _PLANNING_VISUALIZATION_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>

using std::string;
using std::vector;
namespace apexnav_planner {
class PlanningVisualization {
private:
  enum TRAJECTORY_PLANNING_ID {
    GOAL = 1,
    PATH = 200,
    BSPLINE = 300,
    BSPLINE_CTRL_PT = 400,
    POLY_TRAJ = 500
  };

  enum TOPOLOGICAL_PATH_PLANNING_ID {
    GRAPH_NODE = 1,
    GRAPH_EDGE = 100,
    RAW_PATH = 200,
    FILTERED_PATH = 300,
    SELECT_PATH = 400
  };

  /* data */
  /* visib_pub is seperated from previous ones for different info */
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr traj_pub_;       // 0
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr topo_pub_;       // 1
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr predict_pub_;    // 2
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr visib_pub_;      // 3, visibility constraints
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr frontier_pub_;   // 4, frontier searching
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr yaw_pub_;        // 5, yaw trajectory
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viewpoint_pub_;  // 6, viewpoint planning
  vector<rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr> pubs_;   //

  int last_topo_path1_num_;
  int last_topo_path2_num_;
  int last_bspline_phase1_num_;
  int last_bspline_phase2_num_;
  int last_frontier_num_;

public:
  PlanningVisualization(/* args */)
  {
  }
  ~PlanningVisualization()
  {
  }
  PlanningVisualization(rclcpp::Node::SharedPtr node);

  // new interface
  void fillBasicInfo(visualization_msgs::msg::Marker& mk, const Eigen::Vector3d& scale,
      const Eigen::Vector4d& color, const string& ns, const int& id, const int& shape);
  void fillGeometryInfo(visualization_msgs::msg::Marker& mk, const vector<Eigen::Vector3d>& list);
  void fillGeometryInfo(visualization_msgs::msg::Marker& mk, const vector<Eigen::Vector3d>& list1,
      const vector<Eigen::Vector3d>& list2);

  void drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale,
      const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);
  void drawCubes(const vector<Eigen::Vector3d>& list, const double& scale,
      const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
      const double& scale, const Eigen::Vector4d& color, const string& ns, const int& id,
      const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list, const double& scale,
      const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);
  void drawBox(const Eigen::Vector3d& center, const Eigen::Vector3d& scale,
      const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);

  // Deprecated
  // draw basic shapes
  void displaySphereList(const vector<Eigen::Vector3d>& list, double resolution,
      const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayCubeList(const vector<Eigen::Vector3d>& list, double resolution,
      const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
      double line_width, const Eigen::Vector4d& color, int id, int pub_id = 0);
  // draw a piece-wise straight line path
  void drawGeometricPath(const vector<Eigen::Vector3d>& path, double resolution,
      const Eigen::Vector4d& color, int id = 0);
  // draw a polynomial trajectory

  void drawGoal(Eigen::Vector3d goal, double resolution, const Eigen::Vector4d& color, int id = 0);

  Eigen::Vector4d getColor(const double& h, double alpha = 1.0);

  typedef std::shared_ptr<PlanningVisualization> Ptr;

  // SECTION developing
  void drawVisibConstraint(
      const Eigen::MatrixXd& ctrl_pts, const vector<Eigen::Vector3d>& block_pts);
  void drawFrontier(const vector<vector<Eigen::Vector3d>>& frontiers);
};
}  // namespace apexnav_planner
#endif