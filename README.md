# PySTK GYM Environment
## Table of Contents
- [PySTK GYM Environment](#pystk-gym-environment)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. File Organization](#2-file-organization)
  - [3. Prerequisites](#3-prerequisites)
  - [4. Tuning the Environment](#4-tuning-the-environment)
  - [4. Select a Proper Algorithm](#4-select-a-proper-algorithm)
  - [5. Other tunings for SB3](#5-other-tunings-for-sb3)
  - [6. Monitor the training](#6-monitor-the-training)
  - [7. FAQ](#7-faq)
  - [8. Authors and License](#8-authors-and-license)
  - [9. Reference](#9-reference)

## 1. Introduction

This is a GYM environment for the PySuperTuxKart game. It is designed for Reinforcement Learning (RL) applications, particularly for educational purposes. The repository also includes examples of using this environment with Stable Baselines 3 (SB3)'s PPO algorithm.

I created this project as the final project for EC418, taught by Prof. Alex Olshevsky at BU during the fall 2022 semester. I believe that this can still be used as the final project in later semesters. It can also be used to complete the ML concentration in the ECE department. This repo should not be used directly for the project as you should learn tuning parameters yourself.

I believe that this is also relevant to an ML/RL course taught by Prof. Philipp Krähenbühl at UT Austin (CS 394D). I really appreciate his work on the PyTuxKart package that makes all further developments possible.


## 2. File Organization
The files I added are as below
   1. kartEnv.py
      1. The GYM environment that I created
   2. env_example.ipynb
      1. This file has a comprehensive example of using the environment with Stable Baselines 3
   3. kartEnv_TorchRL.py
      1. A WIP environment for using the environment with TorchRL. But I really don't have much time for that. Please contact me if you can help, really appreciate that.


## 3. Prerequisites

   1. Already installed Python 3 and necessary softwares like CUDA.
   2. The pytuxkart package may not work on the latest version of Python, the tested version is Python 3.9.13. I think it is a good idea to use this with a venv.
   3. The repo is a standalone example, no additional files need to be added.
   4. A good GPU/NPU. At least 4 GB of vram, 16-24GB or cloud GPU recommended.

## 4. Tuning the Environment

The environment is ready to run out of the box. However, there are still many things to change to improve the performance of it in reinforcement learning. 

1. Choice of action space

   The environment has two different action space set up. One is discrete (MultiDiscrete), the other is continuous(Box). These two sets of action space both defines how the agent can act in the environment. However, this may lead to very different results when training. They are different in these aspects

   1. Discrete action space limits the number of possible actions at each state to a finite number, however, continuous action space is much larger as every action can be any float32 from -1.0 to 1.0 or 0 to 1.0. 
   2. Continuous action space is more intuitive for steer and acceleration. Discrete action space is more intuitive for brake, nitro, and drift. You can try to use Dict or Tuple in `gym.spaces` if the RL implementation supports. However, I belive that they are not supported by stable baselines 3.
   3. By experiment, both continuous and discrete action space can achieve good results. However, continuous action spaces usually requires more training time. And discrete action space performs significantly worse than continuous action space on two maps. 

2. Reward engineering

   The environment already has some reward set up. However, it may not be optimal. You should adjust the reward to motivate the agent to get the optimal results. You can change the reward code or add reward using the following syntax. You can check the file for where to add reward.


   Below are some tips for reward engineering.

   1. Read papers, especially those related to reward engineering.
   2. General rules
      1. set a high reward when agent finishes the game
      2. set a penalty for the time used
      3. set some motivation for the AI to move forward (frequent rewards)
   3. Be careful when setting the reward, sometimes the agent may find a way of keep getting reward without finishing the game. Read this [34] if interested.

## 4. Select a Proper Algorithm
SB3 has implemented multiple algorithms. The example files shows how to use PPO. But there are many other algorithms that may be used for this task. Below is a incomplete list of that.

   1. **DQN (Deep Q-Network)**: Suitable for discrete action spaces, it uses Q-learning with deep neural networks.
   2. **A2C (Advantage Actor Critic)**: A synchronous version of A3C, combining value and policy learning.
   3. **DDPG (Deep Deterministic Policy Gradient)**: An off-policy algorithm for continuous action spaces.
   4. **SAC (Soft Actor-Critic)**: An off-policy algorithm that maximizes entropy for efficient exploration.
   5. **TD3 (Twin Delayed DDPG)**: An improvement over DDPG, addressing overestimation of Q-values.
   6. **ACKTR (Actor Critic using Kronecker-Factored Trust Region)**: An actor-critic algorithm with efficient trust region optimization.


## 5. Other tunings for SB3

1. feature extractor

   1. the example.ipynb provides a feature extractor. See

      ```python
      class CustomCombinedExtractor(BaseFeaturesExtractor)
      ```

      

      However, the CNN part only has one layer, you need to add more to it to get better performance.

   2. for our application, 3 -5 layers should be enough. Although deeper networks may be helpful, they also requires more training resources

   3.  for details, check https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

2. network architecture

   1. the network used for policy and value function can be customized. One example is given in 

      ```python
      policy_kwargs1 = {
          'activation_fn':torch.nn.ReLU,
      	'net_arch':[dict(pi=[64], vf=[64])],
      }
      ```

      This defines a single layer connected by ReLU. You can try add more layers and see what's the difference

   2. Same as above, detailed instruction at https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

## 6. Monitor the training
The example ipynb file shows how to use tensorboard when train with SB3.
   1. It shows total time in frames, reward, and other useful info
   2. It can be collected and saved locally
   3. VSCode has a nice integration with tensorbord


## 7. FAQ

Actually, nobody asked any questions, so I just listed some questions I had and the answers I found

1. Where to get help with hyperparameter tuning?

   1. rl-zoo3. You can try to use rl zoo directly. I don't think it will work. But the hyperparameter examples at https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml could be helpful.
   2. Hugging Face examples https://huggingface.co/sb3

2. Why my agent cannot learn?

   1. Try **increase training budget**, i.e. increase the timesteps
   2. Encourage exploration
   3. Hyperparameter tuning
   4. Reward engineering and choice of action space
   5. Improve feature extractor
   6. Try normalization

3. How long should I train the agent?

   1. **1 million** timestep at least. 
   2. Depends on your rewards and hyperparameter.
   3. Some tracks can give good results even within 300K timesteps, however, some may require 3 million or more.

4. How fast can the agent get in the end?

   1. Human reference is played by myself. I am not good at this game. 

   2. RL time is obtained by coollecting the best runtime of 4 runs after using PPO to train 5 Million timesteps on each track.
   3. Device used was RTX 4080 or 4090. It took a long  long  long time to train them.

      Unit is seconds, times 15 for frames

      |       Track        | RL Best time |        Human Reference       |
      | :----------------: | :-------: | :----------------------------------: |
      |     lighthouse     |   37.1    |                 38.033                  |
      |     zengarden      |   34.9    |                 36.825                 |
      |      hacienda      |   56.2    |                 55.908                 |
      |    snowtuxpeak     |   50.9    |                 55.626                 |
      | cornfield_crossing |   60.4    |                 52.333                 |
      |      scotland      |   53.0    |                 56.767                  |
      |    cocoa_temple    |   64.8    |                 63.733                 |

      The result is actually not quite satisfactory due to limited training. I am pretty sure you can get much better results.

5. Why I can't create more than one environment?
   1. Multiple environment not supported.
   2. Due to the singleton nature of the PySTK. Cannot create more than one kart object.





## 8. Authors and License

1. Authors

   1. Xiteng Yao, Boston University, e-mail: xtyao@bu.edu.

   2. Chengze Zheng, Carnegie Mellon University.

2. License: GNU GPL V3

   1. Please feel free to use this as course assignment, looking forward to get bug report, suggestions, and more.

## 9. Reference 


[1] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” Jan. 30, 2017, arXiv: arXiv:1412.6980. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1412.6980

[2] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, “Aggregated Residual Transformations for Deep Neural Networks,” Apr. 11, 2017, arXiv: arXiv:1611.05431. doi: 10.48550/arXiv.1611.05431.

[3] V. Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning,” Jun. 16, 2016, arXiv: arXiv:1602.01783. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1602.01783

[4] L.-C. Chen, Y. Yang, J. Wang, W. Xu, and A. L. Yuille, “Attention to Scale: Scale-aware Semantic Image Segmentation,” Jun. 02, 2016, arXiv: arXiv:1511.03339. Accessed: Oct. 31, 2024. 
[Online]. Available: http://arxiv.org/abs/1511.03339

[5] Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel, “Benchmarking Deep Reinforcement Learning for Continuous Control,” May 27, 2016, arXiv: arXiv:1604.06778. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1604.06778

[6] Y. Jia et al., “Caffe: Convolutional Architecture for Fast Feature Embedding,” Jun. 20, 2014, arXiv: arXiv:1408.5093. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1408.5093

[7] L. A. Hendricks, S. Venugopalan, M. Rohrbach, R. Mooney, K. Saenko, and T. Darrell, “Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data,” Apr. 27, 2016, arXiv: arXiv:1511.05284. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1511.05284

[8] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Weinberger, “Deep Networks with Stochastic Depth,” Jul. 28, 2016, arXiv: arXiv:1603.09382. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1603.09382

[9] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” Dec. 10, 2015, arXiv: arXiv:1512.03385. doi: 10.48550/arXiv.1512.03385.

[10] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu, “Deeply-Supervised Nets,” Sep. 25, 2014, arXiv: arXiv:1409.5185. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1409.5185

[11] X. B. Peng, P. Abbeel, S. Levine, and M. Van De Panne, “DeepMimic: example-guided deep reinforcement learning of physics-based character skills,” ACM Trans. Graph., vol. 37, no. 4, pp. 1–14, Aug. 2018, doi: 10.1145/3197517.3201311.

[12] B. Zhao, X. Wu, J. Feng, Q. Peng, and S. Yan, “Diversified Visual Attention Networks for Fine-Grained Object Classification,” IEEE Trans. Multimedia, vol. 19, no. 6, pp. 1245–1256, Jun. 2017, doi: 10.1109/TMM.2017.2648498.

[13] “DLR-RM/rl-baselines3-zoo: A training framework for Stable Baselines3 reinforcement learning agents, with hyperparameter optimization and pre-trained agents included.” Accessed: Nov. 03, 2024. [Online]. Available: https://github.com/DLR-RM/rl-baselines3-zoo

[14] N. Heess et al., “Emergence of Locomotion Behaviours in Rich Environments,” Jul. 10, 2017, arXiv: arXiv:1707.02286. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1707.02286

[15] A. M. Saxe, J. L. McClelland, and S. Ganguli, “Exact solutions to the nonlinear dynamics of learning in deep linear neural networks,” Feb. 19, 2014, arXiv: arXiv:1312.6120. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1312.6120

[16] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” Oct. 20, 2018, arXiv: arXiv:1506.02438. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1506.02438

[17] R. K. Srivastava, K. Greff, and J. Schmidhuber, “Highway Networks,” Nov. 03, 2015, arXiv: arXiv:1505.00387. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1505.00387

[18] “Homework 5.” Accessed: Nov. 03, 2024. [Online]. Available: https://www.philkr.net/cs342/homework/05/

[19] K. He, X. Zhang, S. Ren, and J. Sun, “Identity Mappings in Deep Residual Networks,” Jul. 25, 2016, arXiv: arXiv:1603.05027. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1603.05027

[20] O. Russakovsky et al., “ImageNet Large Scale Visual Recognition Challenge,” Jan. 30, 2015, arXiv: arXiv:1409.0575. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1409.0575

[21] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” Jul. 03, 2012, arXiv: arXiv:1207.0580. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1207.0580

[22] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,” Aug. 23, 2016, arXiv: arXiv:1602.07261. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1602.07261

[23] jsn5, jsn5/pytuxkart. (Jan. 28, 2023). Python. Accessed: Oct. 31, 2024. [Online]. Available: https://github.com/jsn5/pytuxkart

[24] I. J. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio, “Maxout Networks,” Sep. 20, 2013, arXiv: arXiv:1302.4389. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1302.4389

[25] M. Lin, Q. Chen, and S. Yan, “Network In Network,” Mar. 04, 2014, arXiv: arXiv:1312.4400. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1312.4400

[26] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun, “Object Detection Networks on Convolutional Feature Maps,” Aug. 17, 2016, arXiv: arXiv:1504.06066. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1504.06066

[27] G. Brockman et al., “OpenAI Gym,” Jun. 05, 2016, arXiv: arXiv:1606.01540. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1606.01540

[28] “PPO/best-practices-ppo.md at master · EmbersArc/PPO,” GitHub. Accessed: Nov. 03, 2024. [Online]. Available: https://github.com/EmbersArc/PPO/blob/master/best-practices-ppo.md

[29] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” Aug. 28, 2017, arXiv: arXiv:1707.06347. doi: 10.48550/arXiv.1707.06347.

[30] P. Krähenbühl, “pystk Documentation”.

[31] “PySuperTuxKart · PyPI.” Accessed: Oct. 31, 2024. [Online]. Available: https://pypi.org/project/PySuperTuxKart/

[32] “PyTorch.” Accessed: Nov. 01, 2024. [Online]. Available: https://pytorch.org/

[33] “Redirecting.” Accessed: Nov. 03, 2024. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S0896627302009637

[34] “Reinforcement Learning with Corrupted Reward Channel.” Accessed: Nov. 03, 2024. [Online]. Available: https://www.tomeveritt.se/paper/2017/05/29/reinforcement-learning-with-corrupted-reward-channel.html

[35] F. Wang et al., “Residual Attention Network for Image Classification,” Apr. 23, 2017, arXiv: arXiv:1704.06904. doi: 10.48550/arXiv.1704.06904.

[36] P. Dayan and B. W. Balleine, “Reward, Motivation, and Reinforcement Learning,” Neuron, vol. 36, no. 2, pp. 285–298, Oct. 2002, doi: 10.1016/S0896-6273(02)00963-7.

[37] Z. Wang et al., “Sample Efficient Actor-Critic with Experience Replay,” Jul. 10, 2017, arXiv: arXiv:1611.01224. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1611.01224

[38] “sb3 (Stable-Baselines3).” Accessed: Nov. 03, 2024. [Online]. Available: https://huggingface.co/sb3

[39] V. Badrinarayanan, A. Handa, and R. Cipolla, “SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling,” May 27, 2015, arXiv: arXiv:1505.07293. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1505.07293

[40] A. Raffin, J. Kober, and F. Stulp, “Smooth Exploration for Robotic Reinforcement Learning,” Jun. 20, 2021, arXiv: arXiv:2005.05719. doi: 10.48550/arXiv.2005.05719.

[41] “Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations — Stable Baselines3 2.4.0a10 documentation.” Accessed: Oct. 31, 2024. [Online]. Available: https://stable-baselines3.readthedocs.io/en/master/

[42] A. Newell, K. Yang, and J. Deng, “Stacked Hourglass Networks for Human Pose Estimation,” Jul. 26, 2016, arXiv: arXiv:1603.06937. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1603.06937

[43] “SuperTuxKart.” Accessed: Oct. 31, 2024. [Online]. Available: https://supertuxkart.net/Main_Page

[44] “SuperTuxKart,” Wikipedia. Oct. 31, 2024. Accessed: Oct. 31, 2024. [Online]. Available: https://en.wikipedia.org/w/index.php?title=SuperTuxKart&oldid=1254609920

[45] “supertuxkart/stk-code: The code base of supertuxkart.” Accessed: Oct. 31, 2024. [Online]. Available: https://github.com/supertuxkart/stk-code

[46] A. D. Laud, “Theory and application of reward shaping in reinforcement learning,” Ph.D., University of Illinois at Urbana-Champaign, United States -- Illinois, 2004. Accessed: Nov. 03, 2024. [Online]. Available: https://www.proquest.com/docview/305194948/abstract/56561C59AFC44EBPQ/1

[47] S. Sukhbaatar, J. Bruna, M. Paluri, L. Bourdev, and R. Fergus, “Training Convolutional Networks with Noisy Labels,” Apr. 10, 2015, arXiv: arXiv:1406.2080. Accessed: Oct. 31, 2024. [Online]. 
Available: http://arxiv.org/abs/1406.2080

[48] R. K. Srivastava, K. Greff, and J. Schmidhuber, “Training Very Deep Networks,” Nov. 23, 2015, arXiv: arXiv:1507.06228. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1507.06228

[49] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel, “Trust Region Policy Optimization,” Apr. 20, 2017, arXiv: arXiv:1502.05477. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1502.05477

[50] “Welcome to Stable Baselines3 Contrib docs! — Stable Baselines3 - Contrib 2.4.0a10 documentation.” Accessed: Nov. 01, 2024. [Online]. Available: https://sb3-contrib.readthedocs.io/en/master/index.html

[51] S. Zagoruyko and N. Komodakis, “Wide Residual Networks,” Jun. 14, 2017, arXiv: arXiv:1605.07146. Accessed: Oct. 31, 2024. [Online]. Available: http://arxiv.org/abs/1605.07146
