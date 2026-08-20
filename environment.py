import io
import random
import os 

import gymnasium as gym
from gymnasium import spaces
from math import sin, pi
from matplotlib import pyplot as plt
import numpy as np
import pygame
from golf_player import GolfPlayer

from PIL import Image
from golf_hole import GolfHole
from golf_hole_plot import drawHole
from assets.putting_probabilities import probabilities
from shapely.geometry import Point, LineString, Polygon, MultiPoint

from py_game_golf import drawHolePyGame

# Check to make sure the environment is loading as expected
#print("Environment module is being imported.")

# Class inizialization
class Environment(gym.Env):
    def __init__(self, render_mode=None):  #environment = Environment(player.get_club_data)
        # Set up GolfHole and GolfPlayer internally
        sample_hole = os.path.join('assets', 'sample_par4.kml')
        self.golf_hole = GolfHole(sample_hole)
        

        # Flag to check if Pygame has been initialized
        self.pygame_initialized = False

        # Initialize the arguments needed
        player = GolfPlayer()
        self.all_club_data = player.get_all_club_data()
        self.all_clubs = list(self.all_club_data.keys())
        self.par = int(self.golf_hole.par)
        self.flag_position = self.golf_hole.get_flag_coordinates()
        self.hole_distance = self.calculate_distance_two_coords((0,0), self.flag_position)
        self.reward_information = {}
        self.action_history = []  # Initialize action history
        self.position_history = [(0,0)]
        self.done = False
        self.action = np.array([])


        # Standards for Gym Library
        # Define the action space
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),  # Lower bounds for club, aim, and power
            high=np.array([1.0, 1.0, 1.0]),  # Upper bounds for club, aim, and power
            dtype=np.float32
        )

        # Mapping for categorical variables (lie and club_used)
        self.lie_mapping = {"Teebox": 0, "Lake": 1, "Green": 2, "Rough": 3, "Bunker": 4, "Fairway": 5, "Tree": 6}
        self.club_mapping = self.map_clubs()  
        
        # Set the max values for the gym observation space
        max_course_length = 1000  # Default max course length
        self.max_course_length = max_course_length
        min_course_length = -1000  # Its not needed but is a good practice to set the min course length
        num_lie_categories = len(self.lie_mapping)  # Number of categories in lie_mapping
        num_club_categories = len(self.club_mapping)  # Number of categories in club_mapping
        
        # Update 'low' and 'high' arrays to include negative values for position
        low = np.array([
            min_course_length, min_course_length,  # Position x, y can go negative
            0,  # Lie starts from 0
            0, 0,  # Distances are non-negative
            0,  # Club used starts from 0
            0, 0  # Line of sight binary values
        ], dtype=np.float32)

        high = np.array([
            max_course_length, max_course_length,  # Position x, y max values
            num_lie_categories - 1,  # Max lie category
            max_course_length, max_course_length,  # Max values for distances
            num_club_categories - 1,  # Max club used category
            1, 1  # Line of sight binary values can be 1
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.reset()

    # Initialize the pygame window
    def init_pygame(self):
        # Initialize Pygame only when needed
        if not self.pygame_initialized:
            pygame.init()
            height = self.hole_distance + 80
            width = height - 20
            self.window_size = (width, height)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Golf Hole Visualization')
            self.fps = 30
            self.pygame_initialized = True
        
    def seed(self, seed=None):
        self.current_seed = seed  # Store the current seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)  # For Python's random module
        np.random.seed(seed)  # For Numpy's random
        return [seed]

    #Wrap a scalar value into a 2D array with shape (1, 1), needed for the initial observation 
    def wrap_observation(self,value):
        return np.array([[value]], dtype=np.float32)
    
    def flatten_observation(self, observation):
        # Convert each observation component into a numpy array if not already
        # and flatten the arrays to ensure they're 1-dimensional
        components = [np.array(value).flatten() for value in observation.values()]
        # Concatenate all components into a single numpy array
        return np.concatenate(components)

    #Map the club names to the psotion in the array of clubs (numeric encoding needed for the algorithm to work)
    def map_clubs(self):
        clubs = self.all_clubs
        d = {}
        for i in range(0,len(clubs)):
            d[clubs[i]] = i
        return d
    

    # Function to reset the enviroment to the initial state (called after every final episode)    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  

        # Reset arguments to the initial values
        self.ball_position = (0,0)  
        self.lie = "Teebox"
        self.distance_from_pin = self.hole_distance
        self.last_distance_to_pin = self.hole_distance
        self.last_shot_distance = 0
        self.club_used = 1
        self.shots_number = 0
        self.penalties = 0
        self.number_of_putts = 0
        self.reward_information = {}
        self.action_history = []  # Reset action history at the start of each episode
        self.position_history = [(0,0)]
        self.line_of_sight_lake = 0  # False
        self.line_of_sight_tree = 0  # False
        self.done = False
        self.action = np.array([])
        self.translated_strategy = {}

        self.update_state()

        # Construct the initial observation based on the reset state
        initial_observation = {
            'position': np.array(self.ball_position, dtype=np.float32),
            'lie': self.wrap_observation(self.lie_mapping[self.lie]),
            'last_distance_to_pin': np.array([self.last_distance_to_pin], dtype=np.float32),
            'distance_from_pin': np.array([self.distance_from_pin], dtype=np.float32),
            'club_used': self.wrap_observation(self.club_mapping.get(self.club_used, 1)),  # 1 or appropriate value for 'None'
            "line_of_sight_lake": self.wrap_observation(self.line_of_sight_lake),
            "line_of_sight_tree": self.wrap_observation(self.line_of_sight_tree),
        }

        return (self.flatten_observation(initial_observation), {}) 
        
    # Define the step function (fundamental in deep reinforcement learning)
    def step(self, action):
        self.action = action
        
        # Process action (e.g., simulate shot)
        new_state = self.simulate_shot(action)
        self.shots_number += new_state["penalties"] + 1

        # Calculate reward of new state 
        prev_done = self.prev_check_done(new_state)
        reward = self.calculate_reward(action, new_state, prev_done)
        self.reward = reward

        self.last_shot_distance = new_state["last_shot_distance"]
        self.ball_position = new_state["position"]
        self.lie = new_state["lie"]
        self.distance_from_pin = new_state["distance_from_pin"]
        self.club_used = new_state["club_used"]
        self.penalties += new_state["penalties"]
        self.line_of_sight_lake = new_state["line_of_sight_lake"]
        self.line_of_sight_tree = new_state["line_of_sight_tree"]
        self.update_state()

        # Update state based on new ball position
        self.last_distance_to_pin = new_state["distance_from_pin"]

        # Store the shots history 
        self.action_history.append(action)
        self.position_history.append(self.ball_position)

        # Check if the eipsode is done (the ball lies to the green)
        done = self.check_if_done(self.state)

        # Construct the info dictionary to include the action history
        info = {'action_history': self.action_history, 'position_history': self.position_history, 'done':self.done, 'putts':self.number_of_putts, 'translated_strat': self.translated_strategy, 'no_shots': self.shots_number }

        # Set the truncated value to False (its not used, but needs to be passed as an argument for the next Gym method)
        truncated = False

        # Construct the new_state in the same format as initial_observation
        new_observation = {
            'position': np.array(self.ball_position, dtype=np.float32),
            'lie': self.wrap_observation(self.lie_mapping[self.lie]),
            'last_distance_to_pin': np.array([self.last_distance_to_pin], dtype=np.float32),
            'distance_from_pin': np.array([self.distance_from_pin], dtype=np.float32),
            'club_used': self.wrap_observation(self.club_mapping.get(self.club_used, 1)),  # 1 or appropriate value for 'None'
            "line_of_sight_lake": self.wrap_observation(self.line_of_sight_lake),
            "line_of_sight_tree": self.wrap_observation(self.line_of_sight_tree),
        }

        flattened_observation = self.flatten_observation(new_observation)
        return flattened_observation, reward, done, truncated, info
        

    # Update the state based on the current attributes
    def update_state(self):
        self.state = {
            'position': self.ball_position,
            'lie': self.lie,
            'last_distance_to_pin': self.last_distance_to_pin,
            'distance_from_pin': self.distance_from_pin,
            "line_of_sight_lake": self.line_of_sight_lake,
            "line_of_sight_tree": self.line_of_sight_tree,
            'last_shot_distance': self.last_shot_distance,
            'club_used' : self.club_used,
            'shots_number' : self.shots_number,

        }
    
    
    # DEfine the render mechanism (Visualization of a given state)
    def render(self, mode='human'):     
        if mode == 'matplot':
            drawHole(self.golf_hole, self.state)
        elif mode == 'human':
            # Ensure Pygame is initialized before rendering
            self.init_pygame()
            drawHolePyGame(self.golf_hole, self.state, self.screen)
            # Handle Pygame events here to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
        elif mode == 'rgb_array':
            # For rgb_array mode, the plor is drawn into the in-memory buffer and return it as an RGB array
            fig, ax = drawHole(self.golf_hole, self.state, return_fig=True)  # Assuming drawHole is adjusted to return fig, ax

            # Save the figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Open the image and convert to an RGB array
            image = Image.open(buf)
            rgb_array = np.array(image)

            # Close the image and buffer
            image.close()
            buf.close()

            # Also close the figure to prevent memory leakage
            plt.close(fig)

            return rgb_array
        else:
            raise ValueError(f"Unsupported render mode '{mode}'")


    # Function to calculate the total reward of a state/action pair
    def calculate_reward(self, action, new_state, done):
        reward = 0

        # 1. Rewarding closer distance to the hole
        distance_improvement = self.last_distance_to_pin - new_state['distance_from_pin']
        # Assuming the maximum reward for getting the ball in the hole is significantly higher
        distance_reward = max(0, distance_improvement) * 10  # Adjust multiplier to scale reward appropriately
        reward += distance_reward
        self.reward_information["distance_reward"] = distance_reward
        
        # 2. Avoiding penalties
        penalty_reward = -10 * new_state['penalties']  # Penalize for each penalty incurred
        reward += penalty_reward
        self.reward_information["penalty_reward"] = penalty_reward

        
        # 3. Rewarding fewer number of shots when the hole is done
        if done:
            shots_number = self.shots_number
            distance_left = new_state['distance_from_pin']
            if distance_left*3<=10:
                shots_number +=  1
            elif distance_left*3 <= 60:
                shots_number += 2
            else:
                shots_number += 3
            shots_over_par = shots_number - self.par
            shots_reward = -5 * shots_over_par  # Penalize for shots over par, adjust multiplier as needed
            reward += shots_reward
            reward += 100  # Bonus for completing the hole
            if shots_over_par<0:
                reward += 2000 * (self.par-shots_over_par)


            self.reward_information["shots_reward"] = shots_reward

        

         # 4. Rewarding slightly aim near 0.5 which means aim at the flag
        _, aim, _ = self.translate_action(action)        
    
        # Calculate deviation from perfect aim (0.5), assuming action is normalized between 0 and 1
        aim_deviation = abs(0.5 - aim)
        aim_reward = - 2  * aim_deviation  # Penalize based on deviation, adjust multiplier as needed
        reward += aim_reward
        self.reward_information["aim_reward"] = aim_reward
        
        #6 Reward lie:
        reward += self.reward_lie_quality(new_state["lie"])
        self.reward_information["reward_lie"] = self.reward_lie_quality(new_state["lie"])

        return reward
    
    # Reward the lie
    def reward_lie_quality(self, lie):
        lie_rewards = {'Fairway': 2, 'Green': 5, 'Rough': -1, 'Bunker': -2, 'Tree': -4}
        return lie_rewards.get(lie, 0)  # Default to 0 if lie not in dictionar
    
    # Penalize the out of bounds/lake penalties
    def penalize_penalties(self, penalties):
        return -5 * penalties

    """
    Other reward functions used during development 
    """
    def reward_power(self, club, power, last_to_pin, last_shot_distance):
        club_max = self.all_club_data[club]["total"] 
        ideal_power_factor = self.calculate_power_factor(club_max, last_to_pin)
        power_accuracy = ideal_power_factor - power
        return power_accuracy * 10

    def calculate_power_factor(self, club_max_distance, target_distance):
        ideal_power_factor = min(target_distance / club_max_distance, 1.0)
        return ideal_power_factor
    
    def reward_club_change(self, current_club):
        if self.club_used is not None and self.club_used != current_club:
            return 5  
        return 0


    def reward_closer_distance(self, new_distance_from_pin):
        distance_improvement = self.distance_from_pin - new_distance_from_pin
        normalized_improvement = distance_improvement / self.distance_from_pin
        return normalized_improvement * 10  # Scale the reward
    

    def reward_club_use(self, club, prev_distance_to_pin):
        clubs = self.all_clubs
        
        driver_total = self.all_club_data[clubs[-1]]["total"]
        if (club == "Dr" and driver_total < prev_distance_to_pin ):
            return (1 - (driver_total - prev_distance_to_pin) / driver_total) * 5
        
        if (club == self.all_club_data[clubs[0]] and  self.all_club_data[clubs[0]]["total"]> prev_distance_to_pin):
            return 0 
        
        threshold = .98
        clubs = list(self.all_club_data.keys())
        current_index = clubs.index(club)
        club_max = self.all_club_data[club]["total"] * threshold

        if club_max >= prev_distance_to_pin:

            prev_club = clubs[current_index - 1]
            prev_club_max = self.all_club_data[prev_club]["total"] * threshold

            if prev_club_max  < prev_distance_to_pin:
                return (1 - (club_max - prev_distance_to_pin) / club_max) * 5  # Normalize and scale
        return (1 - (club_max - prev_distance_to_pin) / club_max) *  - 5  # Normalize and scale 

    def penalize_line_of_sight(self, line_of_sight):
        if line_of_sight['lake']:
            return -3  # Higher penalty for lake in line of sight
        elif line_of_sight['tree']:
            return -2  # Lower penalty for trees in line of sight
        return 0
    
    
    """
    End of testing reward functions
    """
    
    # Check before the reward if the episode has ended 
    def prev_check_done(self, state):
        if state["lie"] == "Green":
            self.done = True
            return True
        return False

    # Standard check to determine if the episode has ended, and add the number of putts
    def check_if_done(self, state):
        if state["lie"] == "Green":
            self.add_putts(state["distance_from_pin"])

            # Save the final strategy
            self.translated_strategy = self.save_strategy()
            self.print_state_information()
            self.done = True
            return True
        
        return False
    
    # Get the translated strategy
    def get_strat(self):
        return self.translated_strategy

    # Translate the action from normalized form e.g([1, .5, 1] to [Driver, 0, 100])
    def translate_action(self, action):
        clubs = self.all_clubs
        n = action[0]
        n = max(0, min(n, 1))
        # Adjust calculation to ensure equal distribution
        index = int(n * len(clubs))
        # Ensure index is within the list bounds
        index = min(index, len(clubs) - 1)
        aim_choice = -90 + 180 * action[1]
        return clubs[index], aim_choice, action[2]

    # Return the distance between 2 points in the plane
    def calculate_distance_two_coords(self, a,b):
        if a!=None:
            a_lon, a_lat = a
        else:
            a_lon, a_lat = 0,0
        b_lon, b_lat = b
        return ((b_lon - a_lon)**2 + (b_lat - a_lat)**2)**0.5


    # Simulate a normal golf shot using the action data and the plater's personal data
    def simulate_shot(self, action):
        lake_flag = False
        ob_flag = False
        penalties = 0 

        # Simulate the shot based on the action and update ball position
        club, aim, power = self.translate_action(action)
        current_lie = self.state['lie']

        if current_lie != "Teebox":
            club = "3w"

        # Retrieve the club data, current lie, and flag coordinates
        club_stats = self.all_club_data[club]
        flag_coords = self.flag_position  # Assume this method returns (x, y) coordinates of the flag

        # Calculate the direct line angle to the flag from the current position
        dx = flag_coords[0] - self.ball_position[0]
        dy = flag_coords[1] - self.ball_position[1]
        direct_line_angle = np.arctan2(dy, dx)  # Angle in radians

        # Adjust carry and spin based on the lie
        if current_lie == "Bunker":
            carry_reduction_factor = np.random.uniform(0.05, 0.15)  # Randomly between 5% and 15%
            standard_deviation_multiplier = 1.2  # Example multiplier, adjust as needed

        elif current_lie == 'Rough':
            carry_reduction_factor = np.random.uniform(0.05, 0.15)
            standard_deviation_multiplier = 1.5  # Higher multiplier for rough lies

        elif current_lie == "Tree":
            carry_reduction_factor = np.random.uniform(0.05, 0.15)
            standard_deviation_multiplier = 1.3  # Example multiplier, adjust as needed
            if -70 <= aim <= 0:
                aim = -70  # Adjusting to -70 if the aim was negative or 0
            elif 0 < aim <= 70:
                aim = 70
        else:
            carry_reduction_factor = 0
            standard_deviation_multiplier = 1

        carry_adjustment = 1 - carry_reduction_factor

        # Simulate the shot with adjusted standard deviations for lie conditions
        carry = np.random.normal(club_stats['carry'], club_stats['standard carry deviation'] * standard_deviation_multiplier) * carry_adjustment * (power/1)
        side_effect = np.random.normal(0, club_stats['standard side deviation'] * standard_deviation_multiplier * power)
        roll = np.random.normal(club_stats['roll'], club_stats['standard roll deviation'] * standard_deviation_multiplier * power) * carry_adjustment

        # Convert aim_direction from degrees to radians and adjust it based on the direct line angle to the flag
        aim_direction_radians = np.deg2rad(-aim) + direct_line_angle

        # Calculate new position considering the aim, direct line to the flag, and the side effect
        # Note: The side effect is applied directly as a lateral movement from the direct path to the flag
        new_position_x = self.ball_position[0] + (carry + roll) * np.cos(aim_direction_radians)
        new_position_y = self.ball_position[1] + (carry + roll) * np.sin(aim_direction_radians)

        # Apply the side effect as a lateral deviation to the new position
        # Assuming side_effect represents a lateral distance, adjust perpendicular to the flag direction
        new_position_x += side_effect * np.cos(aim_direction_radians + np.pi/2)  # +90 degrees for perpendicular direction
        new_position_y += side_effect * np.sin(aim_direction_radians + np.pi/2)

        new_position = (new_position_x, new_position_y)
        new_lie = self.determine_point_location(new_position)

        # Check if the new lie is lake or out of Bounds, if so, process each case separately
        if (new_lie =="Lake"):
            lake_flag = True
            new_position, new_lie = self.process_lake_penalty(new_position, self.ball_position)
            penalties +=1 

        elif (new_lie == "Out of Bounds"):
            ob_flag = True
            # Return to the previous position as happens in golf
            new_position, new_lie = self.ball_position, self.lie
            penalties +=1
        
        # Double check if the solution worked, if not set the agent back to the previous position
        if (new_lie=="Lake" or new_lie == "Out of Bounds"):
            ob_flag = True
            new_position, new_lie = self.ball_position, self.lie
    
        new_distance_from_pin = self.calculate_distance_two_coords(new_position, self.flag_position)

        flag_lake, flag_tree = self.golf_hole.check_obstacles_in_line_of_sight(new_position, self.flag_position)

        # Create a new temporal state 
        new_state = {
            'position': new_position,
            'lie': new_lie,
            'last_distance_to_pin': self.last_distance_to_pin,
            'distance_from_pin': new_distance_from_pin,
            'last_shot_distance': self.calculate_distance_two_coords(self.ball_position, new_position),
            'club_used' : club,
            'penalties' : penalties,
            'line_of_sight_lake': flag_lake,
            'line_of_sight_tree': flag_tree,
            'lake_flag': lake_flag,
            'ob_flag': ob_flag
        }
        # Return the temporal state
        return new_state
    
    # Process the lake situation 
    def process_lake_penalty(self, current_position, last_position):
        
        # Identify which lake the ball kies into
        current_point = Point(current_position)
        identified_lake_polygon = None
        for lake_coords in self.golf_hole.lakes_coordinates:
            lake_polygon = Polygon(lake_coords)
            if lake_polygon.contains(current_point):
                identified_lake_polygon = lake_polygon
                break
        
        # Create line between both ball positions
        trajectory_line = LineString([last_position, current_position])

        # Create vector from the last position to the current position
        vector = [last_position[0] - current_position[0], last_position[1] - current_position[1]]

        # Find the entry point to the lake by calculating the intersection of the trajectory line with the lake's boundary
        intersection_points = identified_lake_polygon.boundary.intersection(trajectory_line)
        
        if isinstance(intersection_points, Point):
            entry_point = intersection_points

            # Use the vector to adjust the entry point slightly towards the previous position direction
            adjusted_entry_x = entry_point.x + vector[0] * 0.0001  # Small adjustment factor
            adjusted_entry_y = entry_point.y + vector[1] * 0.0001
            
            new_lie = self.determine_point_location((adjusted_entry_x, adjusted_entry_y))
            if new_lie == "Out of Bounds" or new_lie == "Lake":
                return last_position, self.lie
            return (adjusted_entry_x, adjusted_entry_y), new_lie

        elif isinstance(intersection_points, MultiPoint):  # Handle MultiPoint or similar
            flag = False
            points = [p for p in intersection_points.geoms]
            final_point = Point(current_position)

            f_distance = final_point.distance(points[0])
            entry_point = points[0]
            
            # Get the closest point to where the ball finished
            for point in points:

                # Use the vector to adjust the entry point slightly towards the previous position direction
                adjusted_entry_x = point.x + vector[0] * 0.0001  # Small adjustment factor
                adjusted_entry_y = point.y + vector[1] * 0.0001
                adjusted_point = (adjusted_entry_x, adjusted_entry_y)
                
                new_lie = self.determine_point_location(adjusted_point)

                if new_lie == "Out of Bounds" or new_lie == "Lake":
                    distance = final_point.distance(point)

                    if distance <= f_distance:
                        entry_point = adjusted_point
                        f_distance = distance
                        flag = True
            
            if flag:
                return entry_point, new_lie
            else:
                return last_position, self.lie
        else: 
            return last_position, self.lie
        
    # Determine the exact the ball lies    
    def determine_point_location(self, point):
        
        # Check teebox
        for teebox_path in self.golf_hole.teebox_paths:
            if teebox_path.contains_point(point):
                return 'Teebox'
            
        # Check bunkers
        for bunker_path in self.golf_hole.bunker_paths:
            if bunker_path.contains_point(point):
                return 'Bunker'

        # Check lake
        for lake_path in self.golf_hole.lake_paths:
            if lake_path.contains_point(point):
                return 'Lake'
            
        # Check green    
        if self.golf_hole.green_path.contains_point(point):
            return 'Green'    
        
        # Check fairway
        for fairway_path in self.golf_hole.fairways_paths:
            if fairway_path.contains_point(point):
                return 'Fairway'
            
        # Check tree
        for tree_path in self.golf_hole.trees_paths:
            if tree_path.contains_point(point):
                return 'Tree'
            
        # Check rough
        if self.golf_hole.rough_path.contains_point(point):
            return 'Rough'
        
        # If none of the above, then the ball must be out of bounds
        return 'Out of Bounds'

    
    # Unused function to simulate the putting
    def simulate_putt(self, distance):
        distance *= 3 # Convert yards to feet
        # Example probabilities for making the putt in 1, 2, or 3+ attempts
        # These would need to be adjusted based on your actual data
        # Convert the provided data into a Python dictionary for use in simulations
        # Find the closest distance in the dictionary to the input distance
        closest_distance = min(probabilities.keys(), key=lambda x: abs(x-distance))
        one_putt_prob, two_putt_prob, _ = probabilities[closest_distance]
        num_putts = 0
        
        # Generate a random number between 0 and 1
        rand_num = random.random()
        # Determine the outcome based on the generated number and the probabilities
        if rand_num <= one_putt_prob:
            num_putts =  1  # One putt
        elif rand_num <= one_putt_prob + two_putt_prob:
            num_putts = 2  # Two putts
        else:
            num_putts = 3  # Three or more putts
        self.shots_number += num_putts

    # Function to add the number of putts to the total shots given the distance to the pin
    def add_putts(self, distance):
        if distance*3<=10:
            self.number_of_putts =  1
        elif distance*3 <= 60:
            self.number_of_putts = 2
        else:
            self.number_of_putts = 3

        self.shots_number += self.number_of_putts
        self.club_used = "Putter"
        self.last_shot_distance = self.distance_from_pin
        self.distance_from_pin = 0
        self.reward = None

        # Update the state after adding the putts
        self.update_state()
        

     #Function to print the state related info
    def print_state_information(self):
        print("\nCurrent State Information:")
        for key, value in self.state.items():
            print(f"{key}: {value}")

        try:
            if self.reward != None:
                print(f'Reward is: {self.reward}')
                for key, value in self.reward_information.items():
                    print(f"{key}: {value}")

        except AttributeError:
            pass

    # Translate each shot action, in order to save the final strategy
    def translate_history(self):
        history = self.action_history
        final_list = []
        for e in history:

            club,aim, power = e
            clubs = self.all_clubs  # Example list of clubs
            # Correctly handling normalization for n = 1 to map to last club
            if club >= 1:
                index = len(clubs) - 1
            else:
                # For values of n < 1, calculate the index as before
                index = int(club * len(clubs))
                # Ensure index is within list bounds, considering how int truncation affects index calculation
                index = min(index, len(clubs) - 1)

            fclub = clubs[index]
            
            # Normalize the aim choice to a value between 0 and 1
            normalized_aim = round(-90 + (180*aim),2)
            
            # The power is already normalized
            normalized_power = power * 100
            
            # Recreate the action array with normalized values
            action = [fclub, normalized_aim, normalized_power]
            
            final_list.append(action)
        return final_list        


    # Prin the strategy in a readable way
    def print_visual_strategy(self):
        shots = self.translate_history()
        for i in range(len(shots)):
            club,aim,power = shots[i]
            
            b = self.position_history
            f_point = Point(b[i])
            s_point = Point(b[i+1])
            dist = f_point.distance(s_point)
            
            #Calculate the lateral aim from the pin
            f_aim = round(dist * sin((aim* pi)/180),2)

            print(f"\n{i+1}.- shot: ")
            print(f'Club: {club} \nAim: {f_aim}\nIntensity: {int(power)}%')


    # Save the strategy in a more readable way 
    def save_strategy(self):
        strategy = {}
        shots = self.translate_history()
        b = self.position_history
        # Ensure shots and position_history have compatible lengths
        if len(shots) > len(b) - 1:
            raise ValueError("Not enough position history data for the number of shots")

        for i in range(len(shots)):
            club, aim, power = shots[i]
            
            f_point = Point(b[i])
            s_point = Point(b[i+1])
            dist = f_point.distance(s_point)
            
            # Calculate the lateral aim from the pin
            f_aim = round(dist * sin((aim * pi) / 180), 2)

            strategy[i] = {}
            strategy[i]["club"] = club
            strategy[i]["aim"] = f_aim
            strategy[i]["power"] = int(power)

        return strategy  # Assuming you might want to use the generated strategy


        
