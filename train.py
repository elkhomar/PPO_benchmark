
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from custom_callbacks import SaveOnBestTrainingRewardCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

def make_env(env_id: str, rank: int, seed: int = 1):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        Monitor(env, log_dir)  # record stats such as returns
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50) # Records episode-reward
        return wrapped_env
    set_random_seed(seed)
    return _init

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

@hydra.main(config_path=".", config_name="train_config")
def main(cfg: DictConfig) -> None:
    # Setting up the configs
    env_name = cfg.env.env_name
    timesteps = cfg.env.total_timesteps
    num_cpu = cfg.env.num_cpu
    algorithm = cfg.algorithm
    group = cfg.logging.group
    seed = cfg.seed
    wandb_config = {"env_name"  : env_name,
                    "total_timesteps" : timesteps,
                    "num_cpu" : num_cpu,
                    "algorithm" : algorithm,
                    "group" : group,
                    "seed" : seed}

    # Initialising wandb
    run = wandb.init(
        project="sb3",
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        group=group,
    )

    # Creation of the environment
    env = SubprocVecEnv([make_env(env_name, i + 1, i + seed) for i in range(num_cpu)])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0,
        video_length=1200,
    )

    # Instantiation of the model
    model = instantiate(config=algorithm, env=env, verbose=1, tensorboard_log=f"runs/{run.id}")   

    # Training the model
    model.learn(total_timesteps=int(timesteps), callback=[WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,), callback])

if __name__ == "__main__":
    main()

