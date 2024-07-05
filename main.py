import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

import argparse
# from CustomPPO import CustomPPO
from datetime import datetime
import Env
import requests


def send_line(message:str):
    token = '7ZPjzeQrRcI70yDFnhBd4A6xpU8MddE7MntCSdbLBgC'
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    data = {
        'message':message
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("LINE message send sucessfuly")
    else:
        print("LINE message send error：", response.status_code)

if __name__ == "__main__":
    
    n_cpu = 1
    batch_size = 64
    env_name = 'CartPoleSwingUpFixInitState-v0'
    #trained_env = GrayScale_env
    # trained_env = make_vec_env('CartPoleSwingUp-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1)
    # trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1)
    trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs = {"init_x": 2, "init_angle": np.pi/2})
    tensorboard_log = "./"

    #trained_env = make_vec_env(GrayScale_env, n_envs=n_cpu,)
    #env = gym.make("highway-fast-v0", render_mode="human")
    model = PPO("MlpPolicy",
                trained_env,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=1,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=1,
                target_kl=0.1,
                ent_coef=0.6,
                vf_coef=0.8,
                tensorboard_log=tensorboard_log,)
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    # Train the agent
    try:
        model.learn(total_timesteps=int(1e6), tb_log_name=time_str)
    except:
        print("Training stop")
    finally:
        send_line("CartPoleSwingUp Test Done!!!!!")
        print("log name: ", tensorboard_log + time_str)
        model.save(tensorboard_log + "model")

        model = PPO.load(tensorboard_log + "model")
        env = gym.make(env_name, render_mode="human", init_x = 2, init_angle = np.pi/2)
        while True:
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()