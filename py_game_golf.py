import pygame
import sys


# Transform a list of tuples or a list of list of tuples to Pygame window coordinates."""
def transform_coordinates(coordinates, window_width, window_height):
    transformed = []
    if not coordinates:
        return transformed  # Return empty list if coordinates are empty

    if isinstance(coordinates[0], list):
        # Handle list of list of tuples (multiple polygons)
        for polygon in coordinates:
            transformed_polygon = [(window_width / 2 + x, window_height - y - 20) for x, y in polygon]
            transformed.append(transformed_polygon)
    else:
        # Handle a single list of tuples (single polygon)
        transformed = [(window_width / 2 + x, window_height - y- 20) for x, y in coordinates]

    return transformed


def transform_single_tuple(tuplee, window_width, window_height):
    x = tuplee[0]
    y = tuplee[1]

    x = window_width / 2 + x
    y = window_height - y - 20
    return x, y


def drawHolePyGame(golfHole, state, screen):
    ball_coordinates = state["position"]
    width, height = screen.get_size()

    # Define colors for different terrain features
    colors = {
        'rough': '#38761D',  # Forest green
        'fairway': "#6AA84F",  # Dark green
        'lake': '#0B5394',  # Deep sky blue
        'bunkers': "#B79E88",  # Golden rod
        'teebox': "#6AA84F",  # Green
        'green': '#7FD35A',  # Lawn green
        'trees': "#734222",  # Saddle brown
    }


    # Terrain features coordinates 
    features = {
        'rough': golfHole.rough_coordinates,         
        'fairway': golfHole.fairways_coordinates,          
        'bunkers': golfHole.bunkers_coordinates,      
        'lake': golfHole.lakes_coordinates,       
        'trees': golfHole.trees_coordinates,        
        'green': golfHole.green_coordinates,       
        'teebox': golfHole.teeboxes_coordinates,    
  
        # Add other features here
    }

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Fill background
    screen.fill((255, 255, 255))  # White background

    # Draw terrain features
    for feature_type, coordinates in features.items():
        color = colors.get(feature_type, (255, 255, 255))  # Use default color if not found
        if coordinates:  # Check if coordinates list is not empty
            transformed_coordinates = transform_coordinates(coordinates, width, height)
            
            if isinstance(transformed_coordinates[0], list):
                # Handle as multiple polygons
                for polygon in transformed_coordinates:
                    pygame.draw.polygon(screen, color, polygon)
            else:
                # Handle as a single polygon
                pygame.draw.polygon(screen, color, transformed_coordinates)

    pygame.draw.circle(screen, BLACK, transform_single_tuple(ball_coordinates, width, height), 3)
    pygame.draw.circle(screen, WHITE, transform_single_tuple((0,0), width, height), 3)

    # Update display
    pygame.display.flip()

    clock = pygame.time.Clock()
    fps = 30
    clock.tick(fps)






