#ifndef _TRAJ_REPRESENTATION_H
#define _TRAJ_REPRESENTATION_H

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#define IN_CLOSE_SET 'a'
#define IN_OPEN_SET 'b'
#define NOT_EXPAND 'c'

class PathNode {
public:
  /* -------------------- */
  Eigen::Vector2i index;
  int yaw_idx;
  /* --- the state is x y theta(orientation) */
  Eigen::Vector4d state;
  double g_score, f_score;
  double penalty_score;
  /* control input should be steer and arc */
  Eigen::Vector3d input;
  PathNode* parent;
  // Not expanded, in close list, in open list - three states
  char node_state;
  // Reverse, forward, and speed less than 0.01 - three levels
  int singul = 0;
  /* -------------------- */
  PathNode()
  {
    parent = NULL;
    node_state = NOT_EXPAND;
  }
  ~PathNode(){};
};
typedef PathNode* PathNodePtr;

struct FlatTrajData {
  int singul;
  std::vector<Eigen::Vector4d> traj_pts;  // All coordinate points after uniform sampling in a single direction: x,y,t
                                          // Has endpoint but no start point
  std::vector<double> thetas;   // Store yaw after uniform sampling
  Eigen::MatrixXd start_state;  // pva
  Eigen::MatrixXd final_state;  // end flat state (2, 3)
};

#endif
