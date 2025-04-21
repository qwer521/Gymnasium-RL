# Gymnasium-RL

This repository contains AI‑Capstone Project 2 code for:
- **PPO** on continuous‑control Mujoco environments (Humanoid, HalfCheetah, Walker2d, Ant)
- **Double DQN** on the Atari game BeamRider‑v5
## Result
<img src="docs/humanoid.gif" alt="Demo" width="350"/> <img src="docs/ant.gif" alt="Demo" width="350"/> <img src="docs/beam.gif" alt="Demo" width="300"/>  

## Repository Structure
```
Gymnasium-RL/
├── vedios/           # Store the playing vedio
├── logs/             # Store training logs for tensorboard
├── checkpoints/      # Store checkpoint models
├── train_PPO.py      # Main script for PPO
├── train_DDQN.py     # Main script for DDQN
└── requirements.txt  # Dependencies
```
## Setup

Install dependencies:
```
pip install -r requirements.txt
pip install gymnasium[atari] ale-py
pip install gymnasium[mujoco]
```
## Usage

### Train PPO Agent
You can customize the training by modifying the command-line arguments:

- `--envs`: Gymnasium Mujoco environments (default: HalfCheetah-v5).
- `--n-envs`: Number of environments to run in parallel (default: 32).
- `--n-epochs`: Epochs to train the model (default: 3000).
- `--render-epoch`: Render every n epochs (default: 25).

For example:

```bash
python train_PPO.py --env Ant-v5
```
Checkpoints, videos and logs will be saved under checkpoints/ , videos/ and logs/.
### Train DDQN Agent (BeamRider‑v5)
Train DDQN Agent. For BeamRider‑v5 in Atari environment
```
python train_DDQN.py
```
Checkpoints, videos and logs will be saved under checkpoints/ , videos/ and logs/.
## Results
### PPO
- **Half Cheetah reward**:  
![Reward](/docs/Cheetah-reward.png)  
- **Walker2D reward**:  
![Reward](/docs/Walker-reward.png)  
- **Ant reward**:  
![Reward](/docs/Ant-reward.png)  
### DDQN
- **Beamrider reward**:  
![Reward](/docs/beam-reward.png)  
