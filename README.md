# Homework-3

## Setup and Installation

### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Install mujoco-py: `pip install -U 'mujoco-py<2.2,>=2.1'`


### Setup environment

To set up the project environment, Use the `environment.yml` file. It contains the necessary dependencies and installation instructions.

    conda env create -f environment.yml
    
### Export paths variables

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    
## Running the assignment

### Policy Gradient
    python3 main.py --task policy_gradient
    
### Behavior Cloning
    python3 main.py --task behavior_cloning
    
### DAgger
    python3 main.py --task dagger

    
    

