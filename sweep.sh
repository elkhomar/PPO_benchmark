python train.py -m logging.group="Halfcheetah algo comparison" algorithm._target_="stable_baselines3.PPO","stable_baselines3.TD3","stable_baselines3.SAC","sb3_contrib.TRPO","sb3_contrib.TQC"
python train.py -m logging.group="Halfcheetah entropy comparison" algorithm.ent_coef=1e-4,1e-3,5e-3,1e-2,5e-2,1e-1
