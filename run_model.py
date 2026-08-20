import csv
import os
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv

from golf_hole import GolfHole
from golf_hole_plot import drawHoleStrategy



def load_environment():
    # Ensure the custom environment is registered correctly
    env_id = 'golf-sample-v0'
    register(
        id=env_id,
        entry_point='environment:Environment',  # Update with your module path
    )
    # Load the saved model
    model_path = "tmp/best_model.zip"  # Update with the correct path to your model
    model = PPO.load(model_path)

    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)

    # Modification: Wrap the environment with DummyVecEnv for consistency with training
    def make_env():
        def _init():
            env = gym.make(env_id)
            return env
        return _init

    new_env = DummyVecEnv([make_env()])

    obs = new_env.reset()
    # Data storage for CSV output
    data_storage = []

    sample_hole = os.path.join('assets', 'sample_par4.kml')
    golf_hole = GolfHole(sample_hole)
    return new_env, model, obs, data_storage, golf_hole

# Run simulation
def run_simulation(new_env, model, obs, data_storage, golf_hole, no_holes = 100000, save = False, stop = 0):
    i = 0
    while i < no_holes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = new_env.step(action)
        #new_env.envs[0].render()

        if dones[0]:
            
            strat = info[0]["translated_strat"]
            no_shots = info[0]["no_shots"]
            first_club_used = strat[0]["club"] if strat else None
            data_storage.append({
                "no_shots": no_shots,
                "first_club_used": first_club_used,
                "strategy": strat
            })

            if stop != 0 and no_shots == stop:
            
                pos_history = info[0]["position_history"]
                print(strat)
                drawHoleStrategy(golf_hole, pos_history)
                

            obs = new_env.reset()

            if(save):
                new_env.close()

                # Determine the maximum number of shots in any hole
                max_shots = max(len(entry['strategy']) for entry in data_storage)

                # Write the collected data to a CSV file
                csv_file_path = os.path.join("runs", "golf_data_flat3.csv")
                os.makedirs("runs", exist_ok=True)

                with open(csv_file_path, mode='w', newline='') as file:
                    fieldnames = ['no_shots', 'first_club_used'] + [f'shot_{i}_{attr}' for i in range(max_shots) for attr in ('club', 'aim', 'power')]
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for data in data_storage:
                        row = {'no_shots': data['no_shots'], 'first_club_used': data['first_club_used']}
                        for i in range(max_shots):
                            shot = data['strategy'].get(i, {})
                            row[f'shot_{i}_club'] = shot.get('club', '')
                            row[f'shot_{i}_aim'] = shot.get('aim', '')
                            row[f'shot_{i}_power'] = shot.get('power', '')
                        writer.writerow(row)
            
                print(f"Flattened data has been saved to {csv_file_path}")
        i += 1


#### Run the trained model ####
new_env, model, obs,  data_storage, golf_hole = load_environment()
run_simulation(new_env, model, obs, data_storage, golf_hole, stop = 4)
