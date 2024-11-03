# @brief: a GYM env for PySTK game
# @author: Xiteng Yao, Boston University, xtyao@bu.edu
# Chengze Zheng, Carnegie Mellon University
# License: GNU GPL V3.0
# Version V4.8


from gymnasium import Env
from gymnasium.spaces import box, multi_discrete, dict

from moviepy.editor import ImageSequenceClip
from IPython.display import display
from PIL import Image

import numpy as np
import pystk

# if there is a problem with the import, check the README.md, section "How to use"
import utils

# The kart can be rescued every 100 frames
RESCUE_TIMEOUT = utils.RESCUE_TIMEOUT

# used to get the aimpoint, not really used here
TRACK_OFFSET = 15

# the maximum number of frames in an episode. After this number of frames, the env sets terminated to True
# however, the episode is not terminated, the real termination is done by the agent, set in example.ipynb
MAX_FRAMES = 2000

# dimensions of the image, default is 3x96x128
N_CHANNELS = 3
HEIGHT = 96
WIDTH = 128

# Define number of bins for steering and acceleration
num_steer_bins = 51  # For steering between -1 and 1
num_accel_bins = 51  # For acceleration between 0 and 1


# @brief: the environment for the kart
# @details: the environment is a wrapper for the PyTux class, which is a wrapper for the pystk library
class kartEnv(Env):

    _singleton = None

    def __init__(self):
        super().__init__()
        # dummy variables, to satisfy the gym interface
        self.num_envs = 1
        
        # defines the observation space, 
        # the agent can get an image of the track and the speed of the kart
        self.observation_space = dict.Dict(
            {
                # speed is normalized to [-1,1]
                "speed" : box.Box(low=-1.0, high=1.0,shape=(1,),dtype=np.float32),
                
                # image is a 3x96x128 array of uint8
                "image" : box.Box(low=0, high=1, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
                
         }   
        )
        
        # # defines the action space, this is a continuous space
        # self.action_space = box.Box(
        #     low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        #     high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, ], dtype=np.float32),
        #     dtype=np.float32,
        #     shape=(5,)
        # )

        # defines the action space, this is a discrete space
        self.action_space = multi_discrete.MultiDiscrete([
            num_steer_bins,   # Steer
            num_accel_bins,   # Acceleration
            2,                # Drift (0 or 1)
            2,                # Brake (0 or 1)
            2                 # Nitro (0 or 1)
        ])
        
        # initialize the pytux
        self.pytux = utils.PyTux()

        # default track is lighthouse, can be changed in the changeTrack function
        self.trackName = 'lighthouse'

        # initailize the rest of the variables in reset
        self.reset()
        
    # @brief: step the agent in the environment
    # @details: the agent takes an action, also calulates the reward
    # @param action: the action taken by the agent
    # @return: observation, reward, terminated, info
    def step(self, action):
        # do not change below this line
        kart = self.state.players[0].kart
                
        # step reward is used for evaluation
        step_reward = 0

        current_vel = np.linalg.norm(kart.velocity)
        
        self.currentFrame += 1
        
        # Map discrete actions back to original ranges
        steer = (action[0] / (num_steer_bins - 1)) * 2 - 1  # From [0, num_steer_bins-1] to [-1, 1]
        acceleration = action[1] / (num_accel_bins - 1)     # From [0, num_accel_bins-1] to [0, 1]
        drift = bool(action[2])
        brake = bool(action[3])
        nitro = bool(action[4])

        # print("steer, acceleration, drift, nitro: ", steer, acceleration, drift, nitro)

        kartAction = pystk.Action()
        kartAction.steer = steer
        kartAction.acceleration = acceleration
        kartAction.drift = drift
        kartAction.brake = brake
        kartAction.brake = False
        kartAction.nitro = nitro

        
        # print("steer, acceleration, drift, brake, nitro: ", \
        # kartAction.steer, kartAction.acceleration, kartAction.drift, kartAction.brake, kartAction.nitro)
        
        current_time = self.currentFrame
        if current_vel < 1.0 and current_time - self.last_rescue > RESCUE_TIMEOUT:
            self.last_rescue = current_time
            kartAction.rescue = True
            # print("Rescue")
            # step_reward -= 2

        # the kart take the action and the environment is stepped        
        self.pytux.k.step(kartAction)
        
        # get the observation
        self.state.update()
        self.track.update()

        kart = self.state.players[0].kart

        # gets the new speed of the kart after the action
        current_vel = np.linalg.norm(kart.velocity)
        
        # terminated when the kart reaches the end of the track
        terminated = False
        truncated = False
        
        if action is not None:
            # cost of survival
            step_reward -= 0.25

            # Reward for going in the right direction
            if kart.distance_down_track >= self.lastPosition:
                step_reward += 0.1
            else:
                step_reward -= 0.2

            
            self.lastPosition = kart.distance_down_track
            # print("incremental_distance: ", incremental_distance)

            # # Reward for speed
            # step_reward += current_vel * 0.002

            # # Reward for reaching further distance
            # if kart.distance_down_track >= self.record_distance:
            #     step_reward += 0.3
            #     self.record_distance = kart.distance_down_track
            
            # if the kart finished the lap, give it a reward and terminate the episode
            if kart.finished_laps >= 1:
                step_reward += 100

                terminated = True
                truncated = False

                self.terminate = True
            
            # if the kart used up the time, signal a termination to stop generating the video
            elif self.currentFrame  >= MAX_FRAMES:
                terminated = False
                truncated = True
                
                self.terminate = True
                # however, unlike the code above, terminated is not set to True
                # the episode is not terminated by this code, but by the code in the example.ipynb
        
        # clip and normalize the speed
        normalized_vel = current_vel / 200
        clipped_vel = np.clip(normalized_vel, -1, 1)

        image = np.moveaxis(np.array(self.pytux.k.render_data[0].image), -1, 0).astype(np.float32)
        image = image/255.0
        # print("current_vel and normalized_vel: ", current_vel, normalized_vel)
        observation = {
            "speed": np.array([clipped_vel], dtype=np.float32),
            "image": image
        }
        
        # print("Frame: ", self.currentFrame, "finished", terminated, "reward: ", step_reward, "distance: ", kart.distance_down_track, end="\r")

        info = {}

        return observation, step_reward, terminated, truncated, info
    
    
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
    
    
    # @brief: used to reset the environment, and return the first observation
    # called by the agent at the beginning of each episode
    # @return: the observation of the first step
    # @note: the observation is a dictionary, with keys "speed" and "image"
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Existing reset logic
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

        self.state = pystk.WorldState()
        self.track = pystk.Track()

        self.last_rescue = 0
        self.state.update()
        self.track.update()

        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)
    
        self.currentFrame = 0
        self.lap_completed = False
        self.lastPosition = 0
        self.record_distance = 0
        self.images = []


        # clip and normalize the speed
        normalized_vel = current_vel / 200
        clipped_vel = np.clip(normalized_vel, -1, 1)

        image = np.moveaxis(np.array(self.pytux.k.render_data[0].image), -1, 0).astype(np.float32)
        image = image/255.0

        observation = {
            "speed": np.array([clipped_vel], dtype=np.float32),
            "image": image
        }

        info = {}

        return observation, info
            
    # @brief: close the environment
    def close(self):
        self.pytux.close()
        super().close()
