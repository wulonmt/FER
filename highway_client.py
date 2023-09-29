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

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="modified log name", type=str, default ="auto")
parser.add_argument("-s", "--save_log", help="whether save log or not", type=str, default = "True") #parser can't pass bool
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required = True)
parser.add_argument("-t", "--train", help="training or not", type=str, default = "True")
parser.add_argument("-r", "--render_mode", help="h for human & r for rgb_array", type=str, default = "r")
args = parser.parse_args()
ENV_LIST=["merge", "highway", "racetrack", "roundabout", "intersection",]


class HighwayClient(fl.client.NumPyClient):
    def __init__(self):
        
        assert args.environment in ENV_LIST, "Wrong my-ENV"
        assert args.render_mode in "hr", "Wrong render mode"
        rm = "rgb_array" if args.render_mode == "r" else "human"
        
        self.env = gym.make(f"my-{args.environment}-v0", render_mode=rm)

        n_cpu = 8
        batch_size = 64
        self.tensorboard_log=f"{args.environment}_ppo/" if args.save_log == "True" else None
        trained_env = self.env
        #trained_env = make_vec_env(self.env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        #trained_env = make_vec_env(self.env, n_envs=n_cpu,)
        #env = gym.make("highway-fast-v0", render_mode="human")
        self.model = CustomPPO("CnnPolicy",
                    trained_env,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    n_steps=batch_size * 16 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    target_kl=0.5,
                    ent_coef=0.1,
                    kl_coef=0.5,
                    vf_coef=0.8,
                    tensorboard_log=self.tensorboard_log,
                    use_advantage = True,
                    tau = 0.005,
                    )

        self.n_round = int(0)
        
        if args.save_log == "True":
            time_str = Ptime()
            time_str.set_time_now()
            description = args.log_name if args.log_name != "auto" else \
                        f"targetkl{self.model.target_kl:.1e}_entcoef{self.model.ent_coef:.1e}_klcoef{self.model.kl_coef:.1e}_vfcoef{self.model.vf_coef:.1e}_tau{self.model.tau:.1e}"
            self.log_name = time_str.get_time() + f"_{description}"
        
        
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
                         tb_log_name=self.log_name + f"/round_{self.n_round}" if self.n_round>9 else self.log_name + f"/round_0{self.n_round}",
                         reset_num_timesteps=False,
                         )
        print("log name: ", self.tensorboard_log + self.log_name)
        # Save the agent
        if args.save_log:
            self.model.save(self.tensorboard_log + self.log_name + "/model")
            
        return self.get_parameters(config={}), self.model.num_timesteps, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        reward_mean, reward_std = evaluate_policy(self.model, self.env)
        return -reward_mean, self.model.num_timesteps, {"reward mean": reward_mean, "reward std": reward_std} 

def main():        
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="192.168.1.121:8080",
        client=HighwayClient(),
    )
if __name__ == "__main__":
    main()

