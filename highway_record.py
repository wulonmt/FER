import gymnasium as gym
from matplotlib import pyplot as plt
import argparse
import MyEnv
from utils.CustomPPO import CustomPPO
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_model", help="modle to be logged", type=str)
args = parser.parse_args()
ENV_LIST=["merge", "highway", "racetrack", "roundabout", "intersection",]
RECORD =False

if __name__ == "__main__":
    log_env = args.log_model.split('/')[0].split('_')[0]
    assert log_env in ENV_LIST, "Wrong my-ENV"
    env = gym.make(f"my-{log_env}-v0", render_mode="rgb_array")
    env.reset()

    model = PPO.load(args.log_model+"model")
    record_env = RecordVideo(env, video_folder= args.log_model + "/videos", episode_trigger=lambda e: True)
    record_env.unwrapped.set_record_video_wrapper(record_env)
    
    if RECORD:
        for video in range(10):
            done = truncated = False
            obs, info = record_env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action)
                # Render
                env.render()
        record_env.close()
    
    for _ in range(10):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.reset()
    for _ in range(3):
        obs, reward, done, truncated, info = env.step(env.action_type.actions_indexes["IDLE"])

        fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
    plt.show()
