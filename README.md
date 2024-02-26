# PPO_benchmark

The goal of this project is to provide a benchmark of various RL algorithms, PPO in particular, using gymnasium and stable_baselines_3

- The parameters of the experiments such as the environnement and the algorithm can be specified in the .yaml config file and are independent from the code
- The logging is done through wandb and shows video snippets of rollouts at different moments in the training

## Files :
- train.py : where all the training code happens, use "python train.py" to train an agent
- train_config.yaml : configuration file containing all the choices we might make when training an agent (env, algo, hyperparameters etc...)
- script.sh : contains bash code to train multiple agents in succession
- custom_callbacks.py : contains a callback that saves the best performing agent in a training
- /custom ppo : a folder that contains an attempt to implement the ppo algorithm from scratch, it is not finished. Although it is not recommended since the PPO in stable baselines 3 has a lot of implementation details that make it efficient, it is meant as an exercise
