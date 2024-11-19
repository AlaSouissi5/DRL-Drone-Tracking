import os

import cv2
import numpy as np
import torch as th
import torch.nn as nn
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import matplotlib.pyplot as plt

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            self.n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))



class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten())

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = extractors[key](
                        th.as_tensor(observation_space.spaces[key].sample()[None]).float()
                    ).shape[1]

                total_concat_size += n_flatten
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16





        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


class Feat_extract_Async(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        # to visualize the features
        self.episode_steps = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "actor":            # We will just downsample one channel of the image by 4x4 and flatten.
                n_input_channels = subspace.shape[0]
                extractors["actor"] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                    nn.ReLU())

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = extractors[key](
                        th.as_tensor(observation_space.spaces[key].sample()[None]).float()
                    ).shape[1]

            elif key == "critic"  :
                # Run through a Flatter
                extractors["critic"] = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)
    def plot_features(self,output_channels,stack_obs,folder_name):
        fig, axes = plt.subplots(4, 8, figsize=(15, 8))
        for i, ax in enumerate(axes.flat):
            if i < 32:
                ax.imshow(output_channels[i], cmap='gray')
                ax.set_title(f'Channel {i + 1}')
            ax.axis('off')

        plt.tight_layout()
        # Save the figure as an image
        parent_dir = os.path.abspath('..')
        images_dir = os.path.join(parent_dir, folder_name)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        plt.savefig(os.path.join(images_dir, f'output_channels_{self.episode_steps}.png'))

        plt.show()

        # save observation ( assuming they are gray )
        stack_obs = stack_obs[0, :, :, :]
        # Ensure that stack_obs is a tensor of shape (3, H, W)
        if stack_obs.shape[0] == 3 :

            # gray scale images
            image1 = stack_obs[0, :, :]
            image2 = stack_obs[1, :, :]
            image3 = stack_obs[2, :, :]
            # Convert tensors to NumPy arrays

            image1 = (image1.detach().cpu().numpy() * 255).astype('uint8')  # Scale and convert to uint8
            image2 = (image2.detach().cpu().numpy() * 255).astype('uint8')
            image3 = (image3.detach().cpu().numpy() * 255).astype('uint8')

        elif stack_obs.shape[0] == 9 :
            # RGB images

            image1 = stack_obs[0:3, :, :].detach().cpu().numpy() * 255 # Extracting channels 0 to 2
            image2 = stack_obs[3:6, :, :].detach().cpu().numpy() * 255  # Extracting channels 4 to 6
            image3 = stack_obs[6:9, :, :].detach().cpu().numpy() * 255  # Extracting channels 8 to 10


            image1 = image1.transpose(1, 2, 0).astype(np.uint8)  # Extracting channels 0 to 2
            image2 = image2.transpose(1, 2, 0).astype(np.uint8)  # Extracting channels 4 to 6
            image3 =image3.transpose(1, 2, 0).astype(np.uint8)  # Extracting channels 8 to 10

        # Resize the NumPy arrays using cv2.resize
        resized_image1_np = cv2.resize(image1, (1000, 600))  # Resize to 1000x600, adjust as needed
        resized_image2_np = cv2.resize(image2, (1000, 600))
        resized_image3_np = cv2.resize(image3, (1000, 600))

        cv2.imwrite(os.path.join(images_dir, f'stacked_obs_1_{self.episode_steps}.jpg'), resized_image1_np)
        cv2.imwrite(os.path.join(images_dir, f'stacked_obs_2_{self.episode_steps}.jpg'), resized_image2_np)
        cv2.imwrite(os.path.join(images_dir, f'stacked_obs_3_{self.episode_steps}.jpg'), resized_image3_np)

    def forward(self, obs) -> th.Tensor:

        if len(obs.shape) > 2:
            output = self.extractors['actor'](obs)

            # Plot features and observations
            # print("actor obs", obs.shape)
            # output_channels = output.detach().cpu().numpy()[0]
            # self.plot_features(output_channels,obs,"stacked_obs_AsymCNN_mountains_RGB")
            self.episode_steps += 1

            features = nn.Flatten()(output)
        else :
            #print("critic obs", obs.shape)
            features = self.extractors['critic'](obs)

        return features