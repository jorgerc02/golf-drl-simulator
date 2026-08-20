import os

from environment import Environment
from golf_hole import GolfHole

#Demo to simulate and view the outcome of a specific shot
env = Environment()


env.step([1,0.53,0.9])

info = env.reward_information
for key,value in info.items():
    print(f'{key}: {value}')

env.render("matplot")