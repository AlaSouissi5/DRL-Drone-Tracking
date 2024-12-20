import time

import gym
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Resnet_rgb(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        file originale messo nei modelli di Sistema unreal
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Resnet_rgb, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        ''' Resnet18 da pytorch '''
        resnet18 = models.resnet18(pretrained=True)
        self.resnet = th.nn.Sequential(*(list(resnet18.children()))[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Compute shape by doing one forward pass
        with th.no_grad():
            print(observation_space.shape)
            dims = self.resnet(
                th.as_tensor(np.zeros(observation_space.shape)).float()
            ).shape

            self.n_flatten = int(np.prod(dims))
            print(self.n_flatten)

        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Preprocess
        # t = time.time()
        input_batch = self.preprocess(observations)

        b, s, c, w, h = input_batch.shape
        x0 = input_batch.view(-1, c, w, h)

        x1 = self.resnet(x0)
        x1 = x1.view(b, s, -1)
        x2 = torch.flatten(x1, start_dim=1)
        x3 = self.linear(x2)
        # print("Time: ", time.time() - t, observations.shape)
        return x3


class Resnet_one_channel(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        file originale messo nei modelli di Sistema unreal
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Resnet_one_channel, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        ''' Resnet18 da pytorch '''
        resnet18 = models.resnet18(pretrained=True)
        #Assuming we have 1 channel input
        resnet18.conv1 = th.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = th.nn.Sequential(*(list(resnet18.children()))[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = True

        #for 1 channel
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        # Compute shape by doing one forward pass
        with th.no_grad():
            print(observation_space.shape)
            dims = self.resnet(
                th.as_tensor(np.zeros(observation_space.shape)).float()
            ).shape

            self.n_flatten = int(np.prod(dims))
            print(self.n_flatten)

        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Preprocess
        # t = time.time()
        input_batch = self.preprocess(observations)

        b, s, c, w, h = input_batch.shape
        x0 = input_batch.view(-1, c, w, h)

        x1 = self.resnet(x0)
        x1 = x1.view(b, s, -1)
        x2 = torch.flatten(x1, start_dim=1)
        x3 = self.linear(x2)
        # print("Time: ", time.time() - t, observations.shape)
        return x3


class Resnet_two_channel(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        file originale messo nei modelli di Sistema unreal
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Resnet_two_channel, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        ''' Resnet18 da pytorch '''
        resnet18 = models.resnet18(pretrained=True)
        #Assuming we have 2 channel input
        resnet18.conv1 = th.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Initialize the new conv1 weights by copying from the original 3-channel conv1 weights
        original_conv1_weights = resnet18.state_dict()['conv1.weight']
        new_conv1_weights = th.zeros((64, 2, 7, 7))
        new_conv1_weights[:, :2, :, :] = original_conv1_weights[:, :2, :, :]
        resnet18.state_dict()['conv1.weight'] = new_conv1_weights

        self.resnet = th.nn.Sequential(*(list(resnet18.children()))[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Define the preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224]),
        ])

        # Compute shape by doing one forward pass
        with th.no_grad():
            print(observation_space.shape)
            dims = self.resnet(
                th.as_tensor(np.zeros(observation_space.shape)).float()
            ).shape

            self.n_flatten = int(np.prod(dims))
            print(self.n_flatten)

        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Preprocess
        # t = time.time()
        input_batch = self.preprocess(observations)

        b, s, c, w, h = input_batch.shape
        x0 = input_batch.view(-1, c, w, h)

        x1 = self.resnet(x0)
        x1 = x1.view(b, s, -1)
        x2 = torch.flatten(x1, start_dim=1)
        x3 = self.linear(x2)
        # print("Time: ", time.time() - t, observations.shape)
        return x3