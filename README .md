# isaacgym_env_symmetry_humanoid
## Install dependencies

* simulator: Isaac Gym https://developer.nvidia.com/isaac-gym
* Ubuntu 18.04 or 20.04

Download this repository and run:
```bash
pip install -e .
```
## Training Agent

Training the humanoid walking and running task by:
```bash
python train.py task=HumanoidSymmetryWalking headless=True ++train.params.config.symmetry_index=50
python train.py task=HumanoidSymmetryrunning headless=True ++train.params.config.symmetry_index=50
```
The weight of symmetry loss can be adjusted by changing the value of ++train.params.config.symmetry_index.

## Training data is recorded in Tensorboard
```bash
tensorboard --logdir ./runs/
```

## Reference repository
IsaacGymEnvs:
https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

SynergyTool:
https://github.com/JiazhengChai/synergyTool




