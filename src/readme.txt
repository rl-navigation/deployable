The scripts in this folder can be run with:

python2 <script> --help

to print a description of the command-line arguments available.



To train an agent to navigate a graph, simply execute:

python2 train --datafile <graph.pytorch>

To train an agent on the dataset presented in the paper, this would be:

python2 train --datafile ../data/entire-campus.pytorch



The training script will load the data, initially preprocessing all the images in the dataset into dense feature representations using ResNet18. Note that pytorch with CUDA is required to run this software.
Preprocessing may take several hours, depending on computational power available, but the result will be cached on disk in .cache/ so this only needs to be run once.

Once precomputation is finished, the agent will begin training to navigate the graph. Tensorboard logs will be generated during the course of training, in ./runs/running/<run_tag>.
Training may take up to 24 hours to reach perfect performance, depending on computational power available and options provided to the script (for example, larger amounts of noise will slow down training).



The data (graph dataset, pretrained weights for visual encoder and navigation agent) can be downloaded by running data/download_data.sh. Most programs in the src/ directory provide an option for loading a pretrained
navigation agent with the --load_ckpt flag.



Scripts for generating navigation graphs from ROS bag files, and generating the plots, data, and other results presented in the paper, are available in utils/.

