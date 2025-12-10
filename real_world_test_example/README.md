# ApexNav Real-world Test Example

This release contains trajectory-generation and MPC control code. Using the supplied scripts, you can perform a real-world test within the Habitat simulator, and the implementation is readily portable to a physical mobile robot.

## Dependencies

Please follow [the main README](../README.md) to set up the ApexNav
environment and the Habitat simulator before running this example.

## Quick Start

> All following commands should be run in the `apexnav` conda environment

### ROS Compilation
```bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

### Run VLM servers
Run each of the following commands in a separate terminal:
```bash
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm --port 12182
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolov7 --port 12184
```

### Launch visualization and main algorithm
```bash
source ./devel/setup.bash && roslaunch exploration_manager rviz_traj.launch   # RViz visualization
source ./devel/setup.bash && roslaunch exploration_manager exploration_traj.launch   # ApexNav main algorithm
```

### Run Habitat simulator (velocity-control version)
```bash
source ./devel/setup.bash && python habitat_vel_control.py
```

### Run the real-world example node
```bash
source ./devel/setup.bash && python ./real_world_test_example/real_world_test_habitat.py
```


### Visualization with RViz

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="../assets/rviz_show.jpg" alt="piperviz_showline" width="80%">
  </a>
</p>


You can use RViz's *<strong>"2D Pose Estimate"</strong>* tool to set a target pose so the robot can navigate to a specified location.
<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="../assets/point2point.gif" alt="apexnav_demo1" width="80%">
  </a>
  <br>
</p>

You can use RViz's *<strong>"2D Nav Goal"</strong>* tool to start autonomous exploration.


<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="../assets/auto_search.gif" alt="apexnav_demo2" width="80%">
  </a>
  <br>
</p>