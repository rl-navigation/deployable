Training model-free reinforcement learning algorithms currently requires a very large amount of agent interaction to solve challenging tasks, often far more interaction than would be practical on a robot in real time. In addition, photorealistic simulations of specific environments can be hard to come by. For these reasons, we propose learning as much as possible directly from real recorded data.

This project page presents the code and data required to reproduce the results from "Learning Deployable Navigation Policies at Kilometer Scale from a Single Traversal", and apply the approach to other datasets and robots.




<center>




<img src="plots/campus_map_graph_cropped_robot_buildings_inset.png" alt="CampusMap"/>




<h3>Entire Dataset</h3>

This animation shows one frame for each of the 2099 discrete locations in the dataset.
<br/>
<br/>

<img src="gifs/dataset.gif" alt="EntireDatasetGif" width="600px"/>





<h3>Experiment Videos</h3>

</center>
The following animations show one in every 60 frames of the deployment trajectories. This equates to approximately one frame per meter at maximum driving speed.
Below each animation is a plot of the trajectory taken at deployment time compared against the optimal trajectory and the trajectory the agent executes on the training data.
<center>

<h4>Trajectory 1 (successful)</h4>
<img src="gifs/bag1.gif" alt="Trj1" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_0.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_0.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_0.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 2 (successful)</h4>
<img src="gifs/bag2.gif" alt="Trj2" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_1.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_1.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_1.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 3 (successful)</h4>
<img src="gifs/bag3.gif" alt="Trj3" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_2.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_2.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_2.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 4 (successful)</h4>
<img src="gifs/bag4.gif" alt="Trj4" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_3.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_3.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_3.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 5 (successful)</h4>
<img src="gifs/bag5.gif" alt="Trj5" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_4.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_4.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_4.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 6 (failed)</h4>
<img src="gifs/bag6.gif" alt="Trj6" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_5.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_5.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_5.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 7 (successful)</h4>
<img src="gifs/bag8.gif" alt="Trj7" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_6.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_6.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_6.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 8 (successful)</h4>
<img src="gifs/bag9.gif" alt="Trj8" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_7.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_7.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_7.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 9 (successful)</h4>
<img src="gifs/bagA.gif" alt="Trj9" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_8.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_8.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_8.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 10 (successful)</h4>
<img src="gifs/bagD.gif" alt="Trj10" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_9.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_9.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_9.png"  alt="Trj1" width="200px"/>



<br/>
<br/>
<h4>Trajectory 11 (successful)</h4>
<img src="gifs/bagF.gif" alt="Trj11" width="600px"/>
<br/>

<img src="plots/trajectory_optimal_10.png"    alt="Trj1" width="200px"/>
<img src="plots/trajectory_simulation_10.png" alt="Trj1" width="200px"/>
<img src="plots/trajectory_empirical_10.png"  alt="Trj1" width="200px"/>







<h3>Video of Trained Agent</h3>

The following video shows the trained agent navigating in recorded data with a gradually increasing curriculum.

<br/>
<br/>
<div class="embed-container">
  <iframe
      src="https://www.youtube.com/embed/lI7oN7lyIb4"
      width="600"
      height="338"
      frameborder="0"
      allowfullscreen="">
  </iframe>
</div>




<center>

