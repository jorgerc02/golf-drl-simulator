import os

from environment import Environment
from golf_hole import GolfHole

# Demo to view the translated coordinates from the KML file
env = Environment()

flag = env.golf_hole.flag_coordinates
lake = env.golf_hole.lakes_coordinates

print(f'Flag coordinates: \n{flag}\n')
print(f'Lakes coordinates: \n{lake}\n')


#env.step([1,0.53,0.9])
#env.print_state_information()
#env.render("matplot")