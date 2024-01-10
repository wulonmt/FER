import gymnasium as gym
import torch as th
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import highway_env

import argparse
from utils.Ptime import Ptime
import MyEnv
from utils.CustomPPO import CustomPPO

import flwr as fl
from collections import OrderedDict
import os

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="modified log name", type=str, default ="auto")
parser.add_argument("-s", "--save_log", help="whether save log or not", type=str, default = "True") #parser can't pass bool
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required = True)
parser.add_argument("-t", "--train", help="training or not", type=str, default = "True")
parser.add_argument("-r", "--render_mode", help="h for human & r for rgb_array", type=str, default = "r")
parser.add_argument("-i", "--index", help="client index", type=int, default = 0, required = True)
args = parser.parse_args()
ENV_LIST=["merge", "highway", "racetrack", "roundabout", "intersection", "crowded_highway", "crowded_merge",]


class HighwayClient(fl.client.NumPyClient):
    def __init__(self, client_index):
        
        assert args.environment in ENV_LIST, "Wrong my-ENV"
        assert args.render_mode in "hr", "Wrong render mode"
        rm = "rgb_array" if args.render_mode == "r" else "human"

        n_cpu = 2
        batch_size = 64
        #self.env = gym.make(f"my-{args.environment}-v0", render_mode=rm)
        self.env = make_vec_env(f"my-{args.environment}-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        self.tensorboard_log=f"{args.environment}_ppo/" if args.save_log == "True" else None
        time_str = Ptime()
        time_str.set_time_now()
        if args.save_log == "True":
            self.tensorboard_log = "multiagent/" + time_str.get_time_to_hour() + "/" + self.tensorboard_log
        trained_env = self.env
        
        self.model = CustomPPO("CnnPolicy",
                    trained_env,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    n_steps=batch_size * 4 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    target_kl=0.5,
                    ent_coef=0.,
                    kl_coef=0.3,
                    vf_coef=0.8,
                    tensorboard_log=self.tensorboard_log,
                    use_advantage = True,
                    tau = 0.05,
                    )

        self.n_round = int(0)
        
        if args.save_log == "True":
            description = args.log_name if args.log_name != "auto" else \
                        f"multiagent_targetkl{self.model.target_kl:.1e}_entcoef{self.model.ent_coef:.1e}_klcoef{self.model.kl_coef:.1e}_vfcoef{self.model.vf_coef:.1e}_tau{self.model.tau:.1e}"
            self.log_name = f"{client_index}_{description}"
        else:
            self.log_name = None
        
        
    def get_parameters(self, config):
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        return policy_state

    def set_parameters(self, parameters):
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)
        self.model.regul_policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.n_round += 1
        self.set_parameters(parameters)
        if("learning_rate" in config.keys()):
            self.model.learning_rate = config["learning_rate"]
        print(f"Training learning rate: {self.model.learning_rate}")
        # Train the agent
        self.model.learn(total_timesteps=int(2.5e4),
                         tb_log_name=(self.log_name + f"/round_{self.n_round}" if self.n_round>9 else self.log_name + f"/round_0{self.n_round}") if self.log_name is not None else None ,
                         reset_num_timesteps=False,
                         )
        # Save the agent
        if args.save_log == "True":
            print("log name: ", self.tensorboard_log + self.log_name)
            self.model.save(self.tensorboard_log + self.log_name + "/model")
            
        return self.get_parameters(config={}), self.model.num_timesteps, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        reward_mean, reward_std = evaluate_policy(self.model, self.env)
        return -reward_mean, self.model.num_timesteps, {"reward mean": reward_mean, "reward std": reward_std} 

def main():        
    # Start Flower client
    #port = 8080 + args.index
    fl.client.start_numpy_client(
        server_address=f"127.0.0.1:8080",
        client=HighwayClient(args.index),
    )
if __name__ == "__main__":
    main()

