from matplotlib import patches, pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random
from matplotlib.path import Path
import numpy as np


# Draw the hole using Matplotlib
def drawHole(golfHole, state={'position': (0,0)}, return_fig=False):
    fairways_coordinates = golfHole.fairways_coordinates
    rough_coordinates = golfHole.rough_coordinates
    lakes_coordinates = golfHole.lakes_coordinates
    teeboxes_coordinates = golfHole.teeboxes_coordinates
    green_coordinates = golfHole.green_coordinates
    bunkers_coordinates = golfHole.bunkers_coordinates
    trees_coordinates = golfHole.trees_coordinates
    flag_coordinates = golfHole.flag_coordinates
    ball_coordinates = state["position"]


    # Create a figure and an axis
    fig, ax = plt.subplots(figsize = (10,10))

    # Function to convert coordinates and add a polygon to the plot
    def add_polygon(coordinates, facecolor, edgecolor, alpha=1):
        # Convert coordinates to (longitude, latitude)
        coordinates = [(lon, lat) for lon, lat in coordinates]
        polygon = patches.Polygon(coordinates, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(polygon)


    # Function to calculate polygon area using the shoelace formula
    def calculate_polygon_area(coordinates):
        x = [p[0] for p in coordinates]
        y = [p[1] for p in coordinates]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Function to plot tree icons, scaling the number of trees by the area of the polygon
    def plot_tree_icons(coordinates, tree_icon_path, area_per_tree):
        area = calculate_polygon_area(coordinates)
        num_trees = int(area / area_per_tree)  # Determine number of trees based on area
        
        if num_trees == 0:
            num_trees = 1  # Ensure at least one tree if the area is not zero
        
        path = Path(coordinates)  # Create a path from the polygon coordinates
        bbox = path.get_extents()
        minx, miny, maxx, maxy = bbox.x0, bbox.y0, bbox.x1, bbox.y1

        plotted_trees = 0
        attempts = 0
        while plotted_trees < num_trees and attempts < 1000:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy) + 10
            if path.contains_points([(x, y)]):
                img = plt.imread(tree_icon_path)  # Read the tree icon image
                imagebox = OffsetImage(img, zoom=0.05)  # Adjust zoom as necessary
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=2) #lower zorder so the ball can be visible 
                ax.add_artist(ab)
                plotted_trees += 1
            attempts += 1

    #Add teeshot
    ax.plot(0,0,'o', color='#FFFFFF', markersize=3)

    

    flag_color = '#db2c18'
    # Plot the flagpole
    plt.plot(flag_coordinates[0], flag_coordinates[1], '|', color=flag_color, markersize=10)

    # Annotate with a custom flag symbol near the top of the flagpole
    # Adjust the xytext to position the flag correctly relative to your flagpole
    plt.annotate('►', xy=(flag_coordinates[0], flag_coordinates[1]), 
                color=flag_color, fontsize=10, 
                textcoords="offset points", xytext=(0,5), va='center')


    # Add each element as a polygon
    add_polygon(rough_coordinates, '#38761D', '#38761D')

    # Add lakes
    for lake in lakes_coordinates:
        add_polygon(lake, '#0B5394', '#0B5394')
    
    for tree in trees_coordinates:
        add_polygon(tree, "#734222", None, 1)
        plot_tree_icons(tree, 'assets/tree.png', 160)  # Replace 'tree_icon.png' with your icon file path


    # Add bunkers
    for bunker in bunkers_coordinates:
        add_polygon(bunker, '#B79E88', '#AF947B')


    for teebox in teeboxes_coordinates:
        add_polygon(teebox, "#6AA84F", "#6AA84F")

    for fairway in fairways_coordinates:
        add_polygon(fairway, "#6AA84F", "#6AA84F")

    add_polygon(green_coordinates, '#7FD35A', '#7FD35A')

    # Add ball
    ball_color = '#000000'  # Red color hex code for the ball
    ax.plot(ball_coordinates[0], ball_coordinates[1], 'o', color=ball_color, markersize=3, zorder=5)


    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    default = "#f2f2f2"
    white = "#FFFFFF"
    # Set the background (canvas) color
    fig.patch.set_facecolor(default)
    ax.set_facecolor(default)

    # Automatically adjust the view to the data
    #ax.autoscale_view()


    if return_fig:
        return fig, ax  # Only change: return the figure and axes for further processing
    
    else:
        # Show the plot
        plt.show()

# Draw all the shots from the position history
def drawHoleStrategy(golfHole, ball_positions, return_fig=False):
    fairways_coordinates = golfHole.fairways_coordinates
    rough_coordinates = golfHole.rough_coordinates
    lakes_coordinates = golfHole.lakes_coordinates
    teeboxes_coordinates = golfHole.teeboxes_coordinates
    green_coordinates = golfHole.green_coordinates
    bunkers_coordinates = golfHole.bunkers_coordinates
    trees_coordinates = golfHole.trees_coordinates
    flag_coordinates = golfHole.flag_coordinates
    

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize = (10,10))

    # Function to convert coordinates and add a polygon to the plot
    def add_polygon(coordinates, facecolor, edgecolor, alpha=1):
        # Convert coordinates to (longitude, latitude)
        coordinates = [(lon, lat) for lon, lat in coordinates]
        polygon = patches.Polygon(coordinates, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(polygon)


    # Function to calculate polygon area using the shoelace formula
    def calculate_polygon_area(coordinates):
        x = [p[0] for p in coordinates]
        y = [p[1] for p in coordinates]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Function to plot tree icons, scaling the number of trees by the area of the polygon
    def plot_tree_icons(coordinates, tree_icon_path, area_per_tree):
        area = calculate_polygon_area(coordinates)
        num_trees = int(area / area_per_tree)  # Determine number of trees based on area
        
        if num_trees == 0:
            num_trees = 1  # Ensure at least one tree if the area is not zero
        
        path = Path(coordinates)  # Create a path from the polygon coordinates
        bbox = path.get_extents()
        minx, miny, maxx, maxy = bbox.x0, bbox.y0, bbox.x1, bbox.y1

        plotted_trees = 0
        attempts = 0
        while plotted_trees < num_trees and attempts < 1000:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy) + 10
            if path.contains_points([(x, y)]):
                img = plt.imread(tree_icon_path)  # Read the tree icon image
                imagebox = OffsetImage(img, zoom=0.05)  # Adjust zoom as necessary
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=2) #lower zorder so the ball can be visible 
                ax.add_artist(ab)
                plotted_trees += 1
            attempts += 1

    #Add teeshot
    ax.plot(0,0,'o', color='#FFFFFF', markersize=3)

    

    flag_color = '#db2c18'
    # Plot the flagpole
    plt.plot(flag_coordinates[0], flag_coordinates[1], '|', color=flag_color, markersize=10)

    # Annotate with a custom flag symbol near the top of the flagpole
    # Adjust the xytext to position the flag correctly relative to your flagpole
    plt.annotate('►', xy=(flag_coordinates[0], flag_coordinates[1]), 
                color=flag_color, fontsize=10, 
                textcoords="offset points", xytext=(0,5), va='center')


    # Add each element as a polygon
    add_polygon(rough_coordinates, '#38761D', '#38761D')
    
    # Add lakes
    for lake in lakes_coordinates:
        add_polygon(lake, '#0B5394', '#0B5394')


    for tree in trees_coordinates:
        add_polygon(tree, "#734222", None, 1)
        plot_tree_icons(tree, 'assets/tree.png', 160)  # Replace 'tree_icon.png' with your icon file path


    # Add bunkers
    for bunker in bunkers_coordinates:
        add_polygon(bunker, '#B79E88', '#AF947B')

   
    for teebox in teeboxes_coordinates:
        add_polygon(teebox, "#6AA84F", "#6AA84F")

    for fairway in fairways_coordinates:
        add_polygon(fairway, "#6AA84F", "#6AA84F")

    add_polygon(green_coordinates, '#7FD35A', '#7FD35A')
    

    # Add ball for each position
    ball_color = '#000000'  # Black color hex code for the ball
    for ball_coordinate in ball_positions:
        ax.plot(ball_coordinate[0], ball_coordinate[1], 'o', color=ball_color, markersize=4, zorder=5)
    
    # Draw a grey line between all ball positions, if there are at least two positions
    if len(ball_positions) > 1:
        # Extracting x and y coordinates separately for plotting
        x_coords, y_coords = zip(*ball_positions)  # This unzips into two tuples: all x's and all y's
        ax.plot(x_coords, y_coords, '-', color='#808080', zorder=4)  # Grey line


    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    default = "#f2f2f2"
    white = "#FFFFFF"
    # Set the background (canvas) color
    fig.patch.set_facecolor(white)
    ax.set_facecolor(white)

    # Automatically adjust the view to the data
    #ax.autoscale_view()


    if return_fig:
        return fig, ax  # Only change: return the figure and axes for further processing
    
    else:
        # Show the plot
        plt.show()


    
