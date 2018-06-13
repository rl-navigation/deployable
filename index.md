Training behavior policies with model-free reinforcement learning algorithms currently requires a very large amount of agent interaction in order to solve challenging tasks, often far more interaction than would be practical on a real robot in real time. In addition, photorealistic simulations of specific environments can be hard to come by. For these reasons, we propose learning as much as possible directly from real recorded data.

This project page presents the code and data required to reproduce the results from "Learning Deployable Navigation Policies at Kilometer Scale from a Single Traversal", and apply the approach to other datasets and robots.


![CampusMap](plots/campus_map_graph_cropped_robot_buildings_inset.png)

### Entire Dataset ###
![EntireDatasetGif](gifs/dataset.gif)
This animation shows one frame for each of the 2099 discrete locations in the dataset.


### Experiment Videos ###
The following animations show one in every 60 frames of the deployment trajectories. This equates to approximately one frame per meter at maximum driving speed.

#### Trajectory 1 (successful) ####
![Gif1](gifs/bag1.gif)
![Trj1](plots/trajectory_0.gif)

#### Trajectory 2 (successful) ####
![Gif2](gifs/bag2.gif)
![Trj2](plots/trajectory_1.gif)

#### Trajectory 3 (successful) ####
![Gif3](gifs/bag3.gif)
![Trj3](plots/trajectory_2.gif)

#### Trajectory 4 (successful) ####
![Gif4](gifs/bag4.gif)
![Trj4](plots/trajectory_3.gif)

#### Trajectory 5 (successful) ####
![Gif5](gifs/bag5.gif)
![Trj5](plots/trajectory_4.gif)

#### Trajectory 6 (failed) ####
![Gif6](gifs/bag6.gif)
![Trj6](plots/trajectory_5.gif)

#### Trajectory 7 (successful) ####
![Gif8](gifs/bag8.gif)
![Trj7](plots/trajectory_6.gif)

#### Trajectory 8 (successful) ####
![Gif9](gifs/bag9.gif)
![Trj8](plots/trajectory_7.gif)

#### Trajectory 9 (successful) ####
![GifA](gifs/bagA.gif)
![Trj9](plots/trajectory_8.gif)

#### Trajectory 10 (successful) ####
![GifD](gifs/bagD.gif)
![Trj10](plots/trajectory_9.gif)

#### Trajectory 11 (successful) ####
![GifF](gifs/bagF.gif)
![Trj11](plots/trajectory_10.gif)


