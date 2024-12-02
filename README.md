# DRL-Drone-Tracker  

This repository contains the implementation of a drone active tracking system based on Deep Reinforcement Learning (DRL). The system enables a drone to autonomously track a moving target using visual input.

## Article citation
The provided code is based on the method described in the following article :

```latex
@misc{souissi2024leveragingeventstreamsdeep,
      title={Leveraging Event Streams with Deep Reinforcement Learning for End-to-End UAV Tracking}, 
      author={Ala Souissi and Hajer Fradi and Panagiotis Papadakis},
      year={2024},
      eprint={2410.14685},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.14685}, 
}
  ```

--- 

## Set up the Environment          

The code has been tested with **Python 3.8** on a **Windows** host machine. To set up the environment, follow these steps:

### **Install the Requirements**
Run the following commands in your terminal or command prompt:  
```markdown

# Install PyTorch with CUDA 11.3 support
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Upgrade setuptools and install wheel 
pip install -U setuptools==65.5.0
python -m pip install --upgrade pip wheel==0.38.4 setuptools==65.5.1

# Install additional dependencies
pip install gym==0.21  
pip install stable-baselines3==1.5.0
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install wandb
pip install scipy
```

### **Note**
- The version of PyTorch must be chosen based on your system configuration:
  - **CPU-only:** Install the appropriate CPU version of PyTorch.
  - **GPU-enabled:** Use a CUDA-compatible version (as shown above).

---

## State Space and Observation Space

### **State Space**
The **state space** defines the relative state of the target with respect to the tracker. This includes:
- **Relative Position**: The difference in position between the target and the tracker.
- **Relative Velocity**: The difference in velocity between the target and the tracker.
- **Relative Acceleration**: The difference in acceleration between the target and the tracker.

---

### **Observation Space**
The **observation space** consists of inputs provided to the reinforcement learning policy for decision-making:
1. **Visual Inputs**:
   - The target's relative state with respect to the tracker (used when `obs_type` is `state_vector`)
   - RGB images (if `obs_type` is set to `RGB`).
   - Event-based images (if `obs_type` is set to `event_img`).
   - Event-based representation vector (`Event_rep_vector`).
3. **System State**:
   - The target's relative state with respect to the tracker.

The size and format of the observations depend on the configuration:
- **Vector Observations**:
  - Size: 3 (`vect_size` parameter) for `state_vector`.
- **Image Observations**:
  - Height: 250 pixels (`screen_height`).
  - Width: 250 pixels (`screen_width`).
  - Stack Length: 3 consecutive frames (`stack_length`).

---

## Configuration Settings

### **Training Settings**
- **num_cpu**: Number of CPU cores used for parallel training which refers to the number of RL agents (default: 7).
- **policy**: The reinforcement learning policy architecture:
  - `Resnet_DenseMlpPolicy`: Combines ResNet (for visual input) with dense layers.
  - `DenseMlpPolicy`: Uses only dense layers (for vector input).
- **n_timesteps**: Total training timesteps (default: 50,000).
- **eval_episodes**: Number of episodes used for evaluation (default: 6).
- **eval_mode**: Boolean flag to enable evaluation during training (default: `true`).

---

### **Weights and Biases (WandB) Settings**
- **WandB**: Enable or disable WandB integration (`false` by default).
- **WandB_project**: Project name for logging experiments (`DRL-Tracker`).
- **WandB_entity**: Entity name (user or organization) for WandB.
- **WandB_API_key**: API key for authentication (replace with your own key).
- **render**: Enable rendering to plot the trajectories (`true`).
- **debug**: Enable debug mode to visualize event camera encoder-decoder output (`false` by default).

---

### **Reinforcement Learning Environment**
- **dt**: Time step for the simulation (0.005 seconds for events sensor and 0.05 for RGB sensor).
- **episode_time**: Total time per episode in seconds (default: 40 seconds).
- **Optimal Distance**:
  - **optimal_x**: Optimal relative distance in meters (default: 0.2 m, equivalent to 2 in AirSim units).
  - **max_x_dist**: Maximum allowable relative distance (default: 0.35 m).
  - **min_x_dist**: Minimum allowable relative distance (default: 0.1 m).

---

### **Action Space**
- **action_dim**: The action is a 4D vector representing:
  1. Pitch, Rolle, Yaw rotations.
  2. Thrust force.
- **action_limit**: Each action element is clipped within [-4, 4] and the thrust is adjusted later to simulate the real force value of the drone.

---

### **Reward Weights**
The reward function is configured to penalize or reward specific behaviors:
- **alpha (0.4)**: Penalizes high velocity.
- **beta (0)**: Penalizes when the target is out of the field of view (FOV).
- **gamma (0)**: Penalizes high acceleration.
- **sigma (0)**: Additional penalty for being out of FOV.

Other settings:
- **random_pose_init**: Boolean flag for initializing the target at a random pose (`false` by default).
---

## Simulation Setup

Before running the code, make sure to **install AirSim** and set up the environment following the instructions from the official AirSim website:

- **AirSim Installation Guide**: [https://github.com/microsoft/AirSim](https://github.com/microsoft/AirSim)

Once AirSim is installed, follow these additional steps:

### Setting the `settings.json` File

1. Create a `settings.json` file in the appropriate directory on your local machine (usually under `Documents/AirSim/`).
2. Add the following configuration to the file, adjusting parameters as needed:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "PhysicsEngineName": "ExternalPhysicsEngine",
        }


```

- Restart AirSim to apply the changes.

---

## Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/AlaSouissi5/DRL-Drone-Tracking.git
   cd DRL-Drone-Tracking
   ```

2. Install dependencies as outlined in the **Configuration** section.

3. Ensure AirSim is installed and properly configured with the `settings.json` file.

4. Train the model with the parametre `eval_mode` set to `False`:
   ```bash
   python train.py
   ```

5. Evaluate the model with the parametre `eval_mode` set to `True`:
   ```bash
   python train.py
   ```

---
