# @brief: a GYM env for PySTK game
# @author: Xiteng Yao, Boston University, 
# Chengze Zheng, Boston University
# License: GNU GPL V3.0
# Version V1.4


import utils
import pystk
from IPython.display import display
from moviepy.editor import ImageSequenceClip

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from PIL import Image
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import EnvBase

from torchrl.data import Bounded, Composite, Unbounded, Categorical, Binary
from torchrl.envs.transforms import (
    CatTensors,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp





class KartEnvTorchRL(EnvBase):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 15,
    }
    def __init__(self, td_params=None, seed=None, device="cpu"):

        super(KartEnvTorchRL, self).__init__() # call the constructor of the base class

        # Constants
        self.RESCUE_TIMEOUT = utils.RESCUE_TIMEOUT
        self.MAX_FRAMES = 2000
        self.N_CHANNELS = 3
        self.HEIGHT = 96
        self.WIDTH = 128
        self.num_steer_bins = 31
        self.num_accel_bins = 21
        
        self.device = device
        self.dtype = torch.float32

        # Initialize PyTux
        self.pytux = utils.PyTux()
        self.trackName = 'lighthouse'
    
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.observation_spec = Composite(
            speed = Bounded(
                low=-1.0, 
                high=1.0, 
                shape=(), 
                dtype=self.dtype
            ),
        
            image = Bounded(
                low=0.0, 
                high=255.0, 
                shape=(3, 96, 128), 
                dtype= torch.uint8
            ),
            shape=(),
        )

        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        # self.action_spec = Bounded(
        #     low= [-1, 0, 0, 0, 0],
        #     high= [1, 1, 1, 1, 1],
        #     shape=(5,),
        #     dtype=self.dtype,
        # )

        # first 2 are bounded continuous actions, the rest are discrete actions
        self.action_spec = Composite(
            steer = Bounded(
                low=-1.0, 
                high=1.0, 
                shape=(), 
                dtype=self.dtype
            ),
            acceleration = Bounded(
                low=0.0, 
                high=1.0, 
                shape=(), 
                dtype=self.dtype
            ),
            others = Binary(
                n = 3, 
                dtype=self.dtype
            ),
        )

        # self.reward_spec = Unbounded(shape=torch.Size([1]), dtype=self.dtype)
        # self.truncated_spec = Categorical(n=2, shape= torch.Size([1]))



    def _reset(self, tensordict=None):

        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params()
        
        high_speed = torch.tensor(1.0, dtype=self.dtype)
        low_speed = torch.tensor(-1.0, dtype=self.dtype)

        high_image = torch.ones((self.N_CHANNELS, self.HEIGHT, self.WIDTH), dtype=self.dtype)
        # but actually, the image is not normalized, so it can go up to 255
        high_image = high_image * 255

        low_image = torch.zeros((self.N_CHANNELS, self.HEIGHT, self.WIDTH), dtype=self.dtype)

        # Initialize or reset PyTux
        if self.pytux.k is not None and self.pytux.k.config.track == self.trackName:
            self.pytux.k.restart()
            self.pytux.k.step()
        else:
            if self.pytux.k is not None:
                self.pytux.k.stop()
                del self.pytux.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=self.trackName)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.pytux.k = pystk.Race(config)
            self.pytux.k.start()
            self.pytux.k.step()
        
        # Update state
        self.state = pystk.WorldState()
        self.track = pystk.Track()
        self.state.update()
        self.track.update()
        
        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)
        
        self.currentFrame = 0
        self.lap_completed = False
        self.lastPosition = 0
        self.record_distance = 0
        self.last_rescue = 0
        self.images = []
        
        # Normalize velocity
        current_vel = np.clip(current_vel, -50, 50)
        normalized_vel = current_vel / 50.0
        
        # Create initial observation as TensorDict
        out = TensorDict(
            {
                    "speed": torch.tensor(normalized_vel, dtype=self.dtype, device=self.device),
                    "image": torch.from_numpy(
                        np.moveaxis(np.array(self.pytux.k.render_data[0].image), -1, 0)).to(self.device),
                "done": False,
            },
            batch_size=tensordict.shape,
        )

        return out

    def gen_params(self, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {

                    },
                    [],
                )
            },
            [],
        )
        return td
    

    def _step(self, tensordict):

        # Extract discrete actions from TensorDict
        steer_idx = tensordict["steer"]
        accel_idx = tensordict["acceleration"]
        other = tensordict["others"]

        drift = other[0]
        brake = other[1]
        nitro = other[2]


        # Map discrete actions to original ranges
        steer = (steer_idx / (self.num_steer_bins - 1)) * 2 - 1  # [-1, 1]
        acceleration = accel_idx / (self.num_accel_bins - 1)    # [0, 1]
        
        # Create Pystk Action
        kartAction = pystk.Action()
        kartAction.steer = steer
        kartAction.acceleration = acceleration
        kartAction.drift = drift
        kartAction.brake = brake
        kartAction.nitro = nitro
        
        # Rescue logic
        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)
        self.currentFrame += 1
        
        if current_vel < 1.0 and self.currentFrame - self.last_rescue > self.RESCUE_TIMEOUT:
            self.last_rescue = self.currentFrame
            kartAction.rescue = True
        
        # Apply action
        self.pytux.k.step(kartAction)
        
        # Update state
        self.state.update()
        self.track.update()
        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)
        
        # Initialize reward
        step_reward = 0.0
        
        # Survival cost
        step_reward -= 0.25
        
        # Incremental distance reward
        incremental_distance = kart.distance_down_track - self.lastPosition
        step_reward += incremental_distance * 0.005  # Adjust as needed
        self.lastPosition = kart.distance_down_track
        
        # Termination conditions
        terminated = False
        
        if kart.finished_laps >= 1:
            step_reward += 100.0
            terminated = True
        elif self.currentFrame >= self.MAX_FRAMES:
            terminated = True
        
        # Normalize velocity
        current_vel = np.clip(current_vel, -50, 50)
        normalized_vel = current_vel / 50.0
        
        # Create reward as Tensor
        step_reward = np.array(step_reward)

        out = TensorDict({
                    "speed": torch.tensor(normalized_vel, dtype=self.dtype, device=self.device),
                    "image": torch.from_numpy(
                        np.moveaxis(np.array(self.pytux.k.render_data[0].image), -1, 0)).to(self.device),
                "reward": torch.tensor(step_reward.astype(np.float32), dtype=self.dtype, device=self.device),
                "done": terminated,
            },
            tensordict.shape,
        )
    
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng



    # @brief: render the environment and add the current frame to the video
    # unlike the render function in the standard gym environment, this function does not give a visual output, need to use playVideo to see the video
    def render(self, mode='human', **kwargs):
        # add the current frame image to the video    
        image = Image.fromarray(self.pytux.k.render_data[0].image)

        self.images.append(np.array(image))
        # print("Frame: ", self.currentFrame, "finished", self.terminate, end="\r")
    
    
    # @brief: used to play the video of the current episode
    # not a part of the standard gym environment, but used to play the video after the training
    def playVideo(self):
        print("finished at frame: ", self.currentFrame)
        
        display(ImageSequenceClip(self.images, fps=15).ipython_display(width=512, autoplay=True, loop=True, maxduration=400))
    
    
    # @brief: used to change the track, and reset the environment
    def changeTrack(self, track):
        self.trackName = track
        self.reset()
    


    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        if self.pytux.k is not None:
            self.pytux.k.stop()
            del self.pytux.k
        super().close()