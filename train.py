import json
import string
import time
import gym
import numpy as np
from stable_baselines3 import SAC
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import wandb
import sys
from pathlib import Path
import random
from stable_baselines3.common.vec_env import DummyVecEnv , SubprocVecEnv
import models.denseMlpPolicy
import models.asyncCnnPolicy

############# for the Resnet arch
import models.Resnet_denseMlpPolicy.denseMlpPolicy
from models.Resnet_denseMlpPolicy.customCNN import *
######### for DMR architecture



from models.myCnnPolicy import CustomCNN
from models.myCnnPolicy import CustomCombinedExtractor
from models.myCnnPolicy import Feat_extract_Async
from stable_baselines3.common.evaluation import evaluate_policy
from trackingEnv import TrackingEnv
import yaml

"""
We are implementing three different policies for our reinforcement learning model:

1. DenseMlpPolicy:
    -Inputs:
        Critic Network: Current full states.
        Actor Network: History of errors between real states and desired states.
    -Architecture:
        MLP for full state feature extraction (Critic).
        MLP for error history feature extraction (Actor).
        Description: This policy employs an asymmetric approach where the critic network receives 
        the current full states and the actor network receives a history of errors between real states and desired states
        and they processes them through separate MLP networks.

2. CnnPolicy: This method didnt work properly  
    - Input: Stacked images.
    - Architecture: Uses a Convolutional Neural Network (CNN) based feature extractor.
    - Description: This policy takes stacked images as a single input. The images are processed 
      through a CNN to extract features (the actor and the critic use same observations but they have
      separate feature extractors: one for the actor and one for the critic, since the best
      performance is obtained with this configuration.)

3. MultiInputPolicy: This method didnt work properly  
    - Inputs: Stacked images and a state vector containing additional features.
    - Architecture: 
        * CNN for image feature extraction.
        * Multi-Layer Perceptron (MLP) for state vector feature extraction.
    - Description: This policy takes two inputs - stacked images and a state vector. The images 
      are processed using a CNN to extract features, while the state vector is processed using a 
      simple MLP. The extracted features from both inputs are then combined.(the actor and the critic
      also use same observations and they have separate feature extractors.)

4. AsyncPolicy:
    - Inputs:
        * Actor Network: Stacked images.
        * Critic Network: State vector.
    - Architecture: 
        * CNN for image feature extraction (Actor).
        * Simple flatten operation for the state vector (Critic).
    - Description: This policy uses an asymmetric approach where the actor network receives stacked 
      images as input and processes them through a CNN to extract features. The critic network 
      receives a state vector as input and applies a simple flatten operation to extract features. 
      This approach allows for specialized processing for the actor and critic networks, potentially 
      improving learning efficiency and performance.
"""


def make_env(rank : int,
             policy : str = "policy",
             optimal_x: float = 4,
             max_x_dist : float = 0.5,
             min_x_dist : float = 0.1 ,
             dt: float = 0.2,
             episode_time : int = 800 ,
             stack_length: int = 3,
             action_limit : int = 4,
             action_dim : int = 4,
             screen_height : int =60,
             screen_width : int =90,
             random_pose_init : bool= False,
             alpha : float = 0.4,
             beta : float = 0.4,
             gamma : float= 0.2,
             segma : float= 0.2,
             WandB: bool = False,
             debug: bool = False,
             obs_type : str= "image",
             vect_size : int = 3):





    def _init() -> gym.Env:
        env = TrackingEnv(
            rank,
            policy=policy,
            optimal_x=optimal_x,
            max_x_dist = max_x_dist,
            min_x_dist = min_x_dist,
            dt=dt,
            episode_time= episode_time,
            stack_length =stack_length,
            action_limit= action_limit,
            action_dim= action_dim,
            screen_height=screen_height,
            screen_width=screen_width,
            random_pose_init=random_pose_init,
            alpha=alpha,
            beta =beta,
            gamma = gamma,
            segma = segma,
            WandB=WandB,
            debug = debug,
            obs_type = obs_type,
            vect_size = vect_size
        )
        env.seed(random.randint(1, 10000) + rank)
        return env

    return _init



if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        settings = config['settings']

    # initialise airsim settings
    rank = settings['num_cpu']
    optimal_x = settings['optimal_x']
    current_user = os.environ['USERNAME']
    setting_json_path = fr"C:\Users\{current_user}\Documents\AirSim\settings.json"
    with open(setting_json_path, 'r') as file:
        data = json.load(file)
    data['Vehicles'] = {}
    for i in range(1,rank+1):
        data['Vehicles'][f'Tracker_{i}'] = {
            "VehicleType": "SimpleFlight",
            "DefaultVehicleState": "Disarmed",
            "EnableTrace": False,
            "X": 0,
            "Y": (i-1)* 65,  # 65 training box , 90 cave_1
            "Z": 0,
            "Pitch": 0, "Roll": 0, "Yaw": 0
        }
        data['Vehicles'][f'Target_{i}'] = {
            "VehicleType": "SimpleFlight",
            "DefaultVehicleState": "Disarmed",
            "PawnPath": "TargetPawn",
            "EnableTrace": False,
            "X": optimal_x*10,
            "Y": (i-1) * 65,
            "Z": 0,
            "Pitch": 0, "Roll": 0, "Yaw": 0
        }
    with open(setting_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print("The settings.json file has been modified successfully now start the unreal editor.")



    if settings['eval_mode']:
        eval_env = DummyVecEnv([make_env(rank=1,
                                         policy= settings['policy'],
                                         optimal_x=settings['optimal_x'],
                                         max_x_dist=settings['max_x_dist'],
                                         min_x_dist=settings['min_x_dist'],
                                         dt=settings['dt'],
                                         episode_time = settings['episode_time'],
                                         stack_length=settings['stack_length'],
                                         action_limit = settings['action_limit'],
                                         action_dim = settings['action_dim'],
                                         screen_height=settings['screen_height'],
                                         screen_width=settings['screen_width'],
                                         obs_type=settings['obs_type'],
                                         vect_size = settings['vect_size'],
                                         random_pose_init=settings['random_pose_init'],
                                         alpha = settings['alpha'],
                                         beta = settings['beta'],
                                         gamma = settings['gamma'],
                                         segma=settings['segma'],
                                         WandB = settings['WandB'],
                                         debug = settings['debug']
                                         )])

        # MODEL TO TEST
        model_ID = 1721055678
        model_NUMBER = 4
        model = SAC.load(os.path.join("experiments/SAC_{}".format(model_ID), "SAC_{}".format(model_NUMBER)), env=eval_env)
        # for deg=bug
        # policy_kwargs = dict(
        #     net_arch=[512]
        # )
        # model = SAC('Resnet_DenseMlpPolicy',
        #             eval_env,
        #             verbose=1,
        #             policy_kwargs=policy_kwargs,
        #             buffer_size=10000,
        #             batch_size=64,
        #             train_freq=8)

        t = int(time.time())

        if settings['WandB']:
            wandb.login(key=settings['WandB_API_key'])
            wandb.init(project=settings['WandB_project'], entity=settings['WandB_entity'],
                       name=f"SAC_{model_ID}", config=settings)
        # Eval

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=settings['eval_episodes'],
                                                  render=settings['render'])
        print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

        if settings['WandB']:
            wandb.log({'Eval_mean_reward': mean_reward})

    else:
        t = int(time.time())

        # Path for Models
        pathname = os.path.dirname(sys.argv[0])
        abs_path = os.path.abspath(pathname)
        current_path = Path(os.path.join(abs_path, "experiments", "SAC_{}".format(t)))
        current_path.mkdir(parents=True, exist_ok=True)

        #if settings['WandB']:
        if True :
            wandb.login(key=settings['WandB_API_key'])
            wandb.init(project=settings['WandB_project'], entity=settings['WandB_entity'],
                       name="SAC_CNN_{}".format(t), config=settings)

        # Multiprocess training
        vec_env = SubprocVecEnv([make_env(rank=i + 1,
                                          policy=settings['policy'],
                                          optimal_x=settings['optimal_x'],
                                          max_x_dist=settings['max_x_dist'],
                                          min_x_dist=settings['min_x_dist'],
                                          dt=settings['dt'],
                                          episode_time=settings['episode_time'],
                                          stack_length=settings['stack_length'],
                                          action_limit=settings['action_limit'],
                                          action_dim=settings['action_dim'],
                                          screen_height=settings['screen_height'],
                                          screen_width=settings['screen_width'],
                                          obs_type=settings['obs_type'],
                                          vect_size=settings['vect_size'],
                                          random_pose_init = settings['random_pose_init'],
                                          alpha=settings['alpha'],
                                          beta=settings['beta'],
                                          gamma=settings['gamma'],
                                          segma=settings['segma'],
                                          WandB=settings['WandB'],
                                          debug = settings['debug'])
                                 for i in range(settings['num_cpu'])])



        # Create Model for Training
        model = None
        model_ID =   1720537270
        model_NUMBER = 8
        model = SAC.load(os.path.join("experiments/SAC_{}".format(model_ID), "SAC_{}".format(model_NUMBER)), env=vec_env)
        #model = SAC("DenseMlpPolicy", vec_env, verbose=1,buffer_size=int(1e4))

        if settings['policy'] == "Resnet_DenseMlpPolicy" and model is None :
            policy_kwargs = dict(
                net_arch=[512]
            )
            model = SAC('Resnet_DenseMlpPolicy',
                         vec_env,
                         verbose=1,
                         policy_kwargs=policy_kwargs,
                         buffer_size=10000,
                         batch_size=64,
                         train_freq=8)
            #model = SAC.load(os.path.join("experiments/SAC_{}".format(model_ID), "SAC_{}".format(model_NUMBER)), env=vec_env)


        elif settings['policy'] == "DenseMlpPolicy" and model is None :
            model = SAC('DenseMlpPolicy',
                         vec_env,
                         verbose=1,
                         buffer_size=50000,
                         batch_size=32)
            #model = SAC.load(os.path.join("experiments/SAC_{}".format(model_ID), "SAC_{}".format(model_NUMBER)), env=vec_env)

        # print(model.policy)

        # We create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(rank=1,
                                         policy= settings['policy'],
                                         optimal_x=settings['optimal_x'],
                                         max_x_dist=settings['max_x_dist'],
                                         min_x_dist=settings['min_x_dist'],
                                         dt=settings['dt'],
                                         episode_time = settings['episode_time'],
                                         stack_length=settings['stack_length'],
                                         action_limit=settings['action_limit'],
                                         action_dim = settings['action_dim'],
                                         screen_height=settings['screen_height'],
                                         screen_width=settings['screen_width'],
                                         obs_type=settings['obs_type'],
                                         vect_size=settings['vect_size'],
                                         alpha=settings['alpha'],
                                         beta=settings['beta'],
                                         gamma=settings['gamma'],
                                         segma=settings['segma'],
                                         WandB = settings['WandB'],
                                         debug = settings['debug'])])

        # Save Best Models
        best_episodes = np.full((10,), -100.0)
        max_number_of_steps = settings['episode_time'] /  settings['dt']
        # RL Training
        while True:
            model.learn(settings['n_timesteps'])

            # Eval
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=settings['eval_episodes'], render=settings['render'])
            print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

            #if settings['WandB']:
            if True :
                wandb.log({'Mean reward': mean_reward })
                wandb.log({'Std reward': std_reward })

                wandb.log({'Mean reward (norm 0-->100)': mean_reward*(100/max_number_of_steps)}) # 0--> 100
                wandb.log({'Std reward  (norm 0-->100)': std_reward*(100/max_number_of_steps)})
            # After each n_timesteps (the training budget) we get a new policy,
            # and we calculate the mean reward over the sum of collected rewards per episode (eval_episodes)
            worst_model = np.argmin(best_episodes)
            if mean_reward > best_episodes[worst_model]:
                best_episodes[worst_model] = mean_reward
                model.save(os.path.join(current_path, "SAC_{}".format(worst_model)))
                np.savetxt(os.path.join(current_path, "models_score.csv"), best_episodes, delimiter=",")

