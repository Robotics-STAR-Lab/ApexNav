#include <state_monitor/state_monitor.h>
#include <chrono>
#include <string>
#include <thread>
#include <regex>
#include <sstream>
#include <iomanip>

using namespace ftxui;
using namespace std::chrono_literals;

void StateMonitor::init(ros::NodeHandle& nh)
{
  progress_sub_ = nh.subscribe("/habitat/progress", 10, &StateMonitor::progressCallback, this);
  action_sub_ = nh.subscribe("/habitat/plan_action", 10, &StateMonitor::actionCallback, this);
  final_result_sub_ = nh.subscribe("/ros/expl_state", 10, &StateMonitor::finalResultCallback, this);
  ros_state_sub_ = nh.subscribe("/ros/state", 10, &StateMonitor::rosStateCallback, this);
  habitat_state_sub_ =
      nh.subscribe("/habitat/state", 10, &StateMonitor::habitatStateCallback, this);
  expl_result_sub_ = nh.subscribe("/ros/expl_result", 10, &StateMonitor::explResultCallback, this);
  record_sub_ = nh.subscribe("/habitat/record", 10, &StateMonitor::recordCallback, this);
  finished_task_ = 0;
  total_task_ = 0;
  habitat_action_ = 0;
  final_result_ = 0;
  ros_state_ = 0;
  habitat_state_ = 0;
  expl_result_ = 0;

  average_success_ = 0.0;
  previous_average_success_ = 0.0;
  average_spl_ = 0.0;
  previous_average_spl_ = 0.0;
  average_soft_spl_ = 0.0;
  previous_average_soft_spl_ = 0.0;
  average_distance_to_goal_ = 0.0;
  previous_average_distance_to_goal_ = 0.0;

  num_success_ = 0;
  num_infeasible_ = 0;
  num_no_frontier_ = 0;
  num_false_positive_ = 0;
  num_stepout_true_negative_ = 0;
  num_stepout_feasible_ = 0;
  num_stucking_ = 0;
  num_no_frontier_false_negative_ = 0;
  num_stucking_false_negative_ = 0;
  num_stepout_feasible_false_negative_ = 0;
  num_total_ = 0;

  nh.param("/state_monitor_node/previous_record_path", previous_record_path_,
      std::string("default.txt"));

  result_types = { "success", "infeasible", "no frontier", "false positive",
    "stepout true negative", "false negative", "stepout feasible", "stucking",
    "no frontier false negative", "stucking false negative", "stepout false negative" };

  clearTerminal();
}

// Render a labeled box; when selected_state is non-empty, highlight the background.
auto create_mode_box = [](const std::wstring& label, const std::wstring& selected_state) {
  if (selected_state.empty()) {
    return ftxui::text(label);
  }
  return ftxui::text(label) | ftxui::bgcolor(ftxui::Color::Green);
};

void StateMonitor::clearTerminal()
{
  constexpr const char* CLEAR_SCREEN = "\033[2J\033[1;1H";
  printf("%s", CLEAR_SCREEN);
}

void StateMonitor::drawTUI()
{
  // removed unused: reset_position
  std::string data_downloaded = std::to_string(finished_task_) + "/" + std::to_string(total_task_);
  double percentage = total_task_ > 0 ?
                          static_cast<double>(finished_task_) / static_cast<double>(total_task_) :
                          0.0;

  readRecordAvg();

  auto document =
      ftxui::vbox({
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Here is the state of the agent") | ftxui::bold |
                  color(ftxui::Color::Cyan) | ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              text("Evaluate:"),
              gauge(percentage) | flex,
              text(" " + data_downloaded),
          }),
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Result Record") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"  Average Success: ") | ftxui::bold | color(ftxui::Color::YellowLight),
              ftxui::text(toWStringWithPrecision(average_success_, 2)),
              ftxui::text(L"  Average SPL: ") | ftxui::bold | color(ftxui::Color::YellowLight),
              ftxui::text(toWStringWithPrecision(average_spl_, 2)),
              ftxui::text(L"  Average Soft SPL: ") | ftxui::bold | color(ftxui::Color::YellowLight),
              ftxui::text(toWStringWithPrecision(average_soft_spl_, 2)),
              ftxui::text(L"  Average Distance to Goal: ") | ftxui::bold |
                  color(ftxui::Color::YellowLight),
              ftxui::text(toWStringWithPrecision(average_distance_to_goal_, 2)),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Previous Result Record") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"  Average Success: ") | ftxui::bold | color(ftxui::Color::RedLight),
              ftxui::text(toWStringWithPrecision(previous_average_success_, 2)),
              ftxui::text(L"  Average SPL: ") | ftxui::bold | color(ftxui::Color::RedLight),
              ftxui::text(toWStringWithPrecision(previous_average_spl_, 2)),
              ftxui::text(L"  Average Soft SPL: ") | ftxui::bold | color(ftxui::Color::RedLight),
              ftxui::text(toWStringWithPrecision(previous_average_soft_spl_, 2)),
              ftxui::text(L"  Average Distance to Goal: ") | ftxui::bold |
                  color(ftxui::Color::RedLight),
              ftxui::text(toWStringWithPrecision(previous_average_distance_to_goal_, 2)),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Record Details") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"  Total: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_total_)),
              ftxui::text(L"  Success: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_success_)),
              ftxui::text(L"  Infeasible: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_infeasible_)),
              ftxui::text(L"  No Frontier: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_no_frontier_)),
              ftxui::text(L"  False Positive: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_false_positive_)),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"  Stepout True Positive: ") | ftxui::bold |
                  color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_stepout_true_negative_)),
              ftxui::text(L"  Stepout Feasible: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_stepout_feasible_)),
              ftxui::text(L"  Stucking: ") | ftxui::bold | color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_stucking_)),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"  No Frontier False Negative: ") | ftxui::bold |
                  color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_no_frontier_false_negative_)),
              ftxui::text(L"  Stucking False Negative: ") | ftxui::bold |
                  color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_stucking_false_negative_)),
              ftxui::text(L"  Stepout False Negative: ") | ftxui::bold |
                  color(ftxui::Color::BlueLight),
              ftxui::text(std::to_wstring(num_stepout_feasible_false_negative_)),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Final Result: ") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              create_mode_box(
                  L"EXPLORE", final_result_ == FINAL_RESULT::EXPLORE ? L"EXPLORE" : L""),
              ftxui::separator(),
              create_mode_box(L"SEARCH_OBJECT",
                  final_result_ == FINAL_RESULT::SEARCH_OBJECT ? L"SEARCH_OBJECT" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"STUCKING", final_result_ == FINAL_RESULT::STUCKING ? L"STUCKING" : L""),
              ftxui::separator(),
              create_mode_box(L"NO_FRONTIER",
                  final_result_ == FINAL_RESULT::NO_FRONTIER ? L"NO_FRONTIER" : L""),
              ftxui::separator(),
              create_mode_box(L"REACH_OBJECT",
                  final_result_ == FINAL_RESULT::REACH_OBJECT ? L"REACH_OBJECT" : L""),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Exploration Result: ") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              create_mode_box(
                  L"EXPLORATION", expl_result_ == EXPL_RESULT::EXPLORATION ? L"EXPLORATION" : L""),
              ftxui::separator(),
              create_mode_box(L"SEARCH_BEST_OBJECT",
                  expl_result_ == EXPL_RESULT::SEARCH_BEST_OBJECT ? L"SEARCH_BEST_OBJECT" : L""),
              ftxui::separator(),
              create_mode_box(L"SEARCH_OVER_DEPTH_OBJECT",
                  expl_result_ == EXPL_RESULT::SEARCH_OVER_DEPTH_OBJECT ? L"SEARCH_OVER_DEPTH_"
                                                                          L"OBJECT" :
                                                                          L""),
              ftxui::separator(),
              create_mode_box(L"SEARCH_SUSPICIOUS_OBJECT",
                  expl_result_ == EXPL_RESULT::SEARCH_SUSPICIOUS_OBJECT ? L"SEARCH_SUSPICIOUS_"
                                                                          L"OBJECT" :
                                                                          L""),
              ftxui::separator(),
              create_mode_box(L"NO_PASSABLE_FRONTIER",
                  expl_result_ == EXPL_RESULT::NO_PASSABLE_FRONTIER ? L"NO_PASSABLE_FRONTIER" :
                                                                      L""),
              ftxui::separator(),
              create_mode_box(L"NO_COVERABLE_FRONTIER",
                  expl_result_ == EXPL_RESULT::NO_COVERABLE_FRONTIER ? L"NO_COVERABLE_FRONTIER" :
                                                                       L""),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"Habitat FSM State: ") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              create_mode_box(L"READY", habitat_state_ == HABITAT_STATE::READY ? L"READY" : L""),
              ftxui::separator(),
              create_mode_box(L"ACTION_EXEC",
                  habitat_state_ == HABITAT_STATE::ACTION_EXEC ? L"ACTION_EXEC" : L""),
              ftxui::separator(),
              create_mode_box(L"ACTION_FINISH",
                  habitat_state_ == HABITAT_STATE::ACTION_FINISH ? L"ACTION_FINISH" : L""),
              ftxui::separator(),
              create_mode_box(L"EPISODE_FINISH",
                  habitat_state_ == HABITAT_STATE::EPISODE_FINISH ? L"EPISODE_FINISH" : L""),
              ftxui::separator(),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"ROS FSM State: ") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              create_mode_box(L"INIT", ros_state_ == ROS_STATE::INIT ? L"INIT" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"WAIT_TRIGGER", ros_state_ == ROS_STATE::WAIT_TRIGGER ? L"WAIT_TRIGGER" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"PLAN_ACTION", ros_state_ == ROS_STATE::PLAN_ACTION ? L"PLAN_ACTION" : L""),
              ftxui::separator(),
              create_mode_box(L"WAIT_ACTION_FINISH",
                  ros_state_ == ROS_STATE::WAIT_ACTION_FINISH ? L"WAIT_ACTION_FINISH" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"PUB_ACTION", ros_state_ == ROS_STATE::PUB_ACTION ? L"PUB_ACTION" : L""),
              ftxui::separator(),
              create_mode_box(L"FINISH", ros_state_ == ROS_STATE::FINISH ? L"FINISH" : L""),
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              ftxui::text(L"ACTION: ") | ftxui::bold | color(ftxui::Color::Cyan) |
                  ftxui::align_right,
          }) | ftxui::center,
          ftxui::separator(),
          ftxui::hbox({
              create_mode_box(L"STOP", habitat_action_ == ACTION::STOP ? L"STOP" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"MOVE_FORWARD", habitat_action_ == ACTION::MOVE_FORWARD ? L"MOVE_FORWARD" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"TURN_LEFT", habitat_action_ == ACTION::TURN_LEFT ? L"TURN_LEFT" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"TURN_RIGHT", habitat_action_ == ACTION::TURN_RIGHT ? L"TURN_RIGHT" : L""),
              ftxui::separator(),
              create_mode_box(
                  L"TURN_DOWN", habitat_action_ == ACTION::TURN_DOWN ? L"TURN_DOWN" : L""),
              ftxui::separator(),
              create_mode_box(L"TURN_UP", habitat_action_ == ACTION::TURN_UP ? L"TURN_UP" : L""),
          }) | ftxui::center,
          ftxui::separator(),
      }) |
      ftxui::center;

  auto screen = ftxui::Screen::Create(ftxui::Dimension::Full(), ftxui::Dimension::Fit(document));
  Render(screen, document);
  clearTerminal();
  screen.Print();
}

void StateMonitor::updateDisplay()
{
  drawTUI();
}

void StateMonitor::progressCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
{
  if (msg->data.size() != 2) {
    ROS_WARN("Received invalid progress data.");
    return;
  }
  finished_task_ = msg->data[0];
  total_task_ = msg->data[1];
}

void StateMonitor::actionCallback(const std_msgs::Int32::ConstPtr& msg)
{
  habitat_action_ = msg->data;
}

void StateMonitor::finalResultCallback(const std_msgs::Int32::ConstPtr& msg)
{
  final_result_ = msg->data;
}

void StateMonitor::rosStateCallback(const std_msgs::Int32::ConstPtr& msg)
{
  ros_state_ = msg->data;
}

void StateMonitor::habitatStateCallback(const std_msgs::Int32::ConstPtr& msg)
{
  habitat_state_ = msg->data;
}

void StateMonitor::explResultCallback(const std_msgs::Int32::ConstPtr& msg)
{
  expl_result_ = msg->data;
}

void StateMonitor::recordCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
  if (msg->data.size() < 14) {
    ROS_WARN("Received record array with insufficient length: %zu", msg->data.size());
    return;
  }
  average_success_ = msg->data[0];
  average_spl_ = msg->data[1];
  average_soft_spl_ = msg->data[2];
  average_distance_to_goal_ = msg->data[3];
  num_success_ = msg->data[4];
  num_infeasible_ = msg->data[5];
  num_no_frontier_ = msg->data[6];
  num_false_positive_ = msg->data[7];
  num_stepout_true_negative_ = msg->data[8];
  num_stepout_feasible_ = msg->data[9];
  num_stucking_ = msg->data[10];
  num_no_frontier_false_negative_ = msg->data[11];
  num_stucking_false_negative_ = msg->data[12];
  num_stepout_feasible_false_negative_ = msg->data[13];
  num_total_ = num_success_ + num_infeasible_ + num_no_frontier_ + num_false_positive_ +
               num_stepout_true_negative_ + num_stepout_feasible_ + num_stucking_ +
               num_no_frontier_false_negative_ + num_stucking_false_negative_ +
               num_stepout_feasible_false_negative_;
}

std::wstring StateMonitor::toWStringWithPrecision(float value, int precision)
{
  std::wostringstream woss;
  woss << std::fixed << std::setprecision(precision) << value;
  return woss.str();
}

void StateMonitor::readRecordAvg()
{
  // Open the previous record file.
  std::ifstream record_file(previous_record_path_);
  if (!record_file.is_open()) {
    std::cerr << "Failed to open record file: " << previous_record_path_ << std::endl;
    return;
  }

  // Read the entire file into memory.
  std::string content(
      (std::istreambuf_iterator<char>(record_file)), std::istreambuf_iterator<char>());

  // Split into lines and iterate in reverse order.
  std::istringstream content_stream(content);
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(content_stream, line)) {
    lines.push_back(line);
  }

  // Traverse all lines in reverse to find the last finished task.
  bool is_record_found = false;
  int task_number = 0;
  std::regex re(R"(No\.(\d+) task is finished)");

  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    line = *it;

    // Extract the finished task number if present.
    std::smatch match;
    if (std::regex_search(line, match, re)) {
      // Parse the task number and check if it matches the current finished_task_.
      task_number = std::stoi(match[1].str());
      if (task_number == finished_task_) {
        is_record_found = true;
      }
    }

    if (is_record_found) {
      // Parse metrics for the matching task.
      if (line.find("Average Success") != std::string::npos) {
        // Extract percentage value.
        std::regex re(R"(\s*\|\s*(\d+\.\d+)%\s*\|)");
        std::smatch match;
        if (std::regex_search(line, match, re)) {
          previous_average_success_ = std::stod(match[1].str());
        }
        else {
          std::cerr << "Error: Could not extract Average Success value from line." << std::endl;
        }
        is_record_found = false;
      }
      else if (line.find("Average SPL") != std::string::npos) {
        // Extract percentage value.
        std::regex re(R"(\s*\|\s*(\d+\.\d+)%\s*\|)");
        std::smatch match;
        if (std::regex_search(line, match, re)) {
          previous_average_spl_ = std::stod(match[1].str());
        }
        else {
          std::cerr << "Error: Could not extract Average SPL value from line." << std::endl;
        }
      }
      else if (line.find("Average Soft SPL") != std::string::npos) {
        // Extract percentage value.
        std::regex re(R"(\s*\|\s*(\d+\.\d+)%\s*\|)");
        std::smatch match;
        if (std::regex_search(line, match, re)) {
          previous_average_soft_spl_ = std::stod(match[1].str());
        }
        else {
          std::cerr << "Error: Could not extract Average Soft SPL value from line." << std::endl;
        }
      }
      else if (line.find("Average Distance to Goal") != std::string::npos) {
        // Extract a numeric value (no percent sign).
        std::regex re(R"(\s*\|\s*(\d+\.\d+)\s*\|)");
        std::smatch match;
        if (std::regex_search(line, match, re)) {
          previous_average_distance_to_goal_ = std::stod(match[1].str());
        }
        else {
          std::cerr << "Error: Could not extract Average Distance to Goal value from line."
                    << std::endl;
        }
      }
    }
  }
  record_file.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "state_monitor_node");
  ros::NodeHandle nh;
  StateMonitor state_monitor_screen;
  state_monitor_screen.init(nh);

  ros::Rate loop_rate(25);
  while (ros::ok()) {
    state_monitor_screen.updateDisplay();
    loop_rate.sleep();
    ros::spinOnce();
  }

  return EXIT_SUCCESS;
}