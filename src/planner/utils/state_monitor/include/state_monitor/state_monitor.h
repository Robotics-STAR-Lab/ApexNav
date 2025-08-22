#ifndef _STATE_MONITOR_H
#define _STATE_MONITOR_H

#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <std_msgs/String.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>

#include <ftxui/screen/string.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include "ftxui/dom/node.hpp"      // for Render
#include "ftxui/screen/color.hpp"  // for ftxui

enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };
enum ROS_STATE { INIT, WAIT_TRIGGER, PLAN_ACTION, WAIT_ACTION_FINISH, PUB_ACTION, FINISH };
enum ACTION { STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, TURN_DOWN, TURN_UP };
enum HABITAT_STATE { READY, ACTION_EXEC, ACTION_FINISH, EPISODE_FINISH };
enum EXPL_RESULT {
  EXPLORATION,
  SEARCH_BEST_OBJECT,
  SEARCH_OVER_DEPTH_OBJECT,
  SEARCH_SUSPICIOUS_OBJECT,
  NO_PASSABLE_FRONTIER,
  NO_COVERABLE_FRONTIER
};

class StateMonitor {
public:
  StateMonitor() {};
  ~StateMonitor() {};
  void init(ros::NodeHandle& nh);
  void updateDisplay();

private:
  void drawTUI();
  void clearTerminal();
  void progressCallback(const std_msgs::Int32MultiArray::ConstPtr& msg);
  void actionCallback(const std_msgs::Int32::ConstPtr& msg);
  void finalResultCallback(const std_msgs::Int32::ConstPtr& msg);
  void rosStateCallback(const std_msgs::Int32::ConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32::ConstPtr& msg);
  void explResultCallback(const std_msgs::Int32::ConstPtr& msg);
  void recordCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);
  void extractResults(
      const std_msgs::Int32& finished_task_, const std::string& previous_record_path_);
  void readRecordAvg();
  std::wstring toWStringWithPrecision(float value, int precision = 2);
  ros::Subscriber progress_sub_, action_sub_, final_result_sub_, ros_state_sub_, habitat_state_sub_,
      expl_result_sub_, record_sub_;

  int finished_task_, total_task_, habitat_action_, final_result_, ros_state_, habitat_state_,
      expl_result_, num_total_, num_success_, num_infeasible_, num_no_frontier_,
      num_false_positive_, num_stepout_true_negative_, num_stepout_feasible_, num_stucking_,
      num_no_frontier_false_negative_, num_stucking_false_negative_,
      num_stepout_feasible_false_negative_;

  float average_success_, average_spl_, average_soft_spl_, average_distance_to_goal_,
      previous_average_success_, previous_average_spl_, previous_average_soft_spl_,
      previous_average_distance_to_goal_;

  std::string previous_record_path_;
  std::vector<std::string> result_types;
};

#endif
