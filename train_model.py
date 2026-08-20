import json
import numpy as np
import os
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
import csv

from torch.utils.tensorboard import SummaryWriter

model_name = "demo_model"


from stable_baselines3.common.vec_env import VecMonitor
import numpy as np
import os


# Override the Monitor Wrapper so I can log/store more specific data
class CustomVecMonitor(VecMonitor):
    def __init__(self, venv, log_dir=None):
        # Ensure filename for logging is defined if log_dir is provided
        filename = os.path.join(log_dir, "monitor.csv") if log_dir is not None else None
        super(CustomVecMonitor, self).__init__(venv, filename=filename)
        self.log_dir = log_dir
        # You can initialize additional variables if needed
        self.episode_counter = 0

    #Override the step_wait method to add custom logging before the episode resets.
    def step_wait(self):
        observations, rewards, dones, infos = super(CustomVecMonitor, self).step_wait()
        for i, done in enumerate(dones):
            if done:
                self.episode_counter += 1  # Increment episode counter
                action_history = infos[0]["action_history"]
                position_history = infos[0]["position_history"]
                number_putts = infos[0]["putts"]
                done_history = infos[0]["done"]

                action_history_converted = [action.tolist() for action in action_history]


                action_history_str = json.dumps([list(action) for action in action_history_converted])
                position_history_str = json.dumps([list(position) for position in position_history])
                done_history_str = json.dumps(done_history)

                i_reward = infos[i]['episode']['r'] # Total episode reward
                i_length = infos[i]['episode']['l']  # Total episode reward

           
                reward = json.dumps(float(i_reward))
                length = json.dumps(float(i_length) + float(number_putts))

                # Specify the path for the CSV file
                csv_path = os.path.join(self.log_dir, f'{model_name}/additional_info.csv')

                # Ensure the directory exists
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)

                # Write the data to a CSV file, appending to it for each call to this method
                # SAVE ADDITIONAL INFO: Uncomment this if you want to create the files with the information
                # Note: It consumes to much space
                """
                
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Optionally write headers if the file is new/empty
                    if os.stat(csv_path).st_size == 0:
                        writer.writerow(['Action History', 'Position History', 'Done', 'Reward', 'length'])
                    
                    # Write the entire histories as single rows
                    writer.writerow([action_history_str, position_history_str, done_history_str, reward, length])

                print(f"Episode ended for environment {i}")
                """
                
        return observations, rewards, dones, infos



# Hybrid Action Environment Wrapper
class HybridActionEnv(gym.Wrapper):
    def __init__(self, env, epsilon=0.1):
        super(HybridActionEnv, self).__init__(env)
        self.epsilon = epsilon

    def step(self, action):
        if np.random.random() < self.epsilon:
            action = self.action_space.sample()
        return super(HybridActionEnv, self).step(action)

# Callback for saving the best model
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.best_seed = None  # Track the best seed

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")


                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True
    


def make_env(env_id, epsilon, seed=0):
    def _init():
        env = gym.make(env_id)
        env = HybridActionEnv(env, epsilon=epsilon)
        env.seed(seed)
        return env
    return _init

# Environment setup
env_id = 'golf-environment-v1'
register(
    id=env_id,
    entry_point='environment:Environment',
)

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

epsilon = 0.2  # Probability to select a random action
env = CustomVecMonitor(DummyVecEnv([make_env(env_id, epsilon=epsilon, seed=12345)]), log_dir)
# Ensure make_env passes the seed to the environment

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./board/", learning_rate=0.0003)

# Callback for saving best model and additional information based on training reward
# Adjust check_freq to 10000 to match requirement
callback = SaveOnBestTrainingRewardCallback(check_freq=100000, log_dir=log_dir)

# Start training
print("----------------- Starting Learning ---------------")
model.learn(total_timesteps=int(1e5), callback=callback, tb_log_name=model_name)
model.save(env_id)

print("----------------- Done Learning ---------------")

#1.7e6