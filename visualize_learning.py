import ast
import pandas as pd
import io

from environment import Environment
from golf_hole_plot import drawHoleStrategy
import pygame
from py_game_golf import drawHolePyGame

import pandas as pd
import numpy as np

environment = Environment()

## IMPORTANT: In order to visualize the learning, the additional info file must have been created during the training phase. 
## Make sure the functionality is not commented in the train_model.py

# Split the datafram into parts, so you can see the learning by chunks
def split_dataframe_into_equal_parts(dataframe, number_of_parts):

    # Calculate the number of rows in each split, rounding up to ensure all data is included
    rows_per_split = np.ceil(len(dataframe) / number_of_parts).astype(int)
    
    # Initialize a list to hold the split dataframes
    splits = []
    
    for i in range(number_of_parts):
        start_row = i * rows_per_split
        # If it's the last split, take all remaining rows to avoid missing any data
        if i == number_of_parts - 1:
            split = dataframe.iloc[start_row:]
        else:
            end_row = start_row + rows_per_split
            split = dataframe.iloc[start_row:end_row]
        splits.append(split)
    
    return splits


def find_length_values_with_highest_rewards(dataframe):
    # Worst Length Value with Highest Reward
    max_length = dataframe['length'].max()
    filtered_df_max = dataframe[dataframe['length'] == max_length]
    index_of_max_reward_max_length = filtered_df_max['Reward'].idxmax()
    worst_length_with_highest_reward = filtered_df_max.loc[index_of_max_reward_max_length]

    # Mean Length (50% Percentile) with Highest Reward
    median_length = dataframe['length'].median()
    # Finding the closest length to the median
    closest_median_length = dataframe.iloc[(dataframe['length']-median_length).abs().argsort()[:1]]['length'].values[0]
    filtered_df_median = dataframe[dataframe['length'] == closest_median_length]
    index_of_max_reward_median_length = filtered_df_median['Reward'].idxmax()
    mean_length_with_highest_reward = filtered_df_median.loc[index_of_max_reward_median_length]

    # 15th Percentile Length with Highest Reward
    percentile_15_length = dataframe['length'].quantile(0.10)
    # Finding the closest length to the 15th percentile
    closest_percentile_15_length = dataframe.iloc[(dataframe['length']-percentile_15_length).abs().argsort()[:1]]['length'].values[0]
    filtered_df_percentile_15 = dataframe[dataframe['length'] == closest_percentile_15_length]
    index_of_max_reward_percentile_15_length = filtered_df_percentile_15['Reward'].idxmax()
    percentile_15_length_with_highest_reward = filtered_df_percentile_15.loc[index_of_max_reward_percentile_15_length]

    # Best Length Value with Highest Reward
    min_length = dataframe['length'].min()
    filtered_df_min = dataframe[dataframe['length'] == min_length]
    index_of_max_reward_min_length = filtered_df_min['Reward'].idxmax()
    best_length_with_highest_reward = filtered_df_min.loc[index_of_max_reward_min_length]

    return worst_length_with_highest_reward, mean_length_with_highest_reward, percentile_15_length_with_highest_reward, best_length_with_highest_reward


#Visualizes the position and action history in a pygame window.
def visualize_position_history_pygame(value, environment, fps=15): 
    height = environment.hole_distance * 1.15
    width = height/1.5
    window_size = (width, height)

    position_history_string = value["Position History"]
    action_history_string = value["Action History"]


    position_history_list = ast.literal_eval(position_history_string)
    action_history_list = ast.literal_eval(action_history_string)

    # Convert the list of lists to a list of tuples using a list comprehension
    position_history = [tuple(lst) for lst in position_history_list]
    action_history = [tuple(lst) for lst in action_history_list]
    
    # Assuming Environment() and drawHolePyGame() are defined elsewhere
    environment = Environment()

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption('Golf Hole Visualization')
    clock = pygame.time.Clock()

    running = True
    for position in position_history:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        state = {"position": position}
        drawHolePyGame(environment.golf_hole, state, screen)
        pygame.display.flip()
        clock.tick(fps)

    
    return position_history


# Loop through all the small dataframes obtained from the original dataframe 
def loop_dataframes(dataframes):
    for dt in dataframes:
        print("Displaying training: ")
        max, mean, ten, min = find_length_values_with_highest_rewards(dt)
        position_history = visualize_position_history_pygame(ten, environment)
        #drawHoleStrategy(environment.golf_hole, position_history)

    return position_history

# Read the CSV file into a pandas DataFrame (Change the direction of the addtional_info file if needed)
df = pd.read_csv("tmp/final_model/additional_info.csv")
dataframes = split_dataframe_into_equal_parts(df, 20)

position_history = loop_dataframes(dataframes=dataframes)
drawHoleStrategy(environment.golf_hole, position_history)
