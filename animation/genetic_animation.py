import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsp_solver import TSPSolver
from config.style_config import DEFAULT_STYLE

def visualize_tsp(tsp_solver, path=None, save_path=None):
    """
    Visualize the TSP solution for a given TSPSolver instance.
    
    Args:
        tsp_solver: Instance of TSPSolver with coordinates and solution data
        path: Optional specific path to visualize (defaults to tsp_solver.best_path)
        save_path: Optional file path to save the plot (e.g., 'tsp_solution.png')
    """
    # Initialize style with defaults
    style = DEFAULT_STYLE.copy()
    
    plt.figure(figsize=style['fig_size'])
    
    # Plot cities as scatter points
    plt.scatter(tsp_solver.coordinates[:, 0], tsp_solver.coordinates[:, 1], 
                c=style['city_color'], s=style['city_size'], label='Cities')
    
    # Annotate city indices
    for i, (x, y) in enumerate(tsp_solver.coordinates):
        plt.annotate(f"{i}", (x, y), fontsize=style['font_size'])
    
    # Plot the path (either provided or best_path)
    if path is not None:
        path_coords = tsp_solver.coordinates[list(path)]
        plt.plot(path_coords[:, 0], path_coords[:, 1], 
                color=style['path_color'], linewidth=2, label='Tour')
    elif tsp_solver.best_path is not None:
        path_coords = tsp_solver.coordinates[list(tsp_solver.best_path)]
        plt.plot(path_coords[:, 0], path_coords[:, 1], 
                color=style['path_color'], linewidth=2, label='Tour')
    
    # Set title and labels
    plt.title(f'TSP Solution - {tsp_solver.num_cities} Cities - Distance: {tsp_solver.best_distance:.2f}',
             fontsize=style['title_size'])
    plt.xlabel('X Coordinate', fontsize=style['label_size'])
    plt.ylabel('Y Coordinate', fontsize=style['label_size'])
    plt.legend(fontsize=style['legend_size'])
    plt.grid(True, linestyle=style['grid_linestyle'], alpha=style['grid_alpha'], color=style['grid_color'])
    
    # Save or show the plot
    if save_path:
        print(f"Saving static plot to {save_path}")
        plt.savefig(save_path)
    plt.show()

def animate_tsp(tsp_solver, save_path=None):
    """
    Animate the TSP solution process, showing movement between cities and the evolution of the genetic algorithm.
    
    Args:
        tsp_solver: Instance of TSPSolver with all_paths and path_distances
        save_path: Optional file path to save the animation (e.g., 'tsp_genetic.gif')
    """
    if not tsp_solver.all_paths:
        print("No paths available for animation. Run solve_genetic with visualize_steps=True.")
        return
    
    # Initialize style with defaults
    style = DEFAULT_STYLE.copy()
    
    # Set up the figure with two subplots: main plot for the tour, small plot for fitness
    fig = plt.figure(figsize=style['fig_size'])
    ax_tour = fig.add_subplot(121)
    ax_fitness = fig.add_subplot(122)

    # Initialize the fitness plot (distance over generations)
    generations = list(range(len(tsp_solver.path_distances)))
    distances = tsp_solver.path_distances
    ax_fitness.plot(generations, distances, 'b-', label='Best Distance')
    ax_fitness.set_xlabel('Sampled Generation', fontsize=style['label_size'])
    ax_fitness.set_ylabel('Distance', fontsize=style['label_size'])
    ax_fitness.set_title('Genetic Algorithm Convergence', fontsize=style['title_size'])
    ax_fitness.grid(True, linestyle=style['grid_linestyle'], alpha=style['grid_alpha'], color=style['grid_color'])
    ax_fitness.legend(fontsize=style['legend_size'])

    # Calculate frames: for each path, animate city-to-city movement
    frames_per_path = tsp_solver.num_cities + 1  # +1 for the return to start
    total_frames = len(tsp_solver.all_paths) * frames_per_path

    def update(frame):
        # Clear the tour plot for the new frame
        ax_tour.clear()
        
        # Determine which path and segment we're animating
        path_idx = frame // frames_per_path
        segment_idx = frame % frames_per_path

        # Get the current path and coordinates
        path = tsp_solver.all_paths[path_idx]
        path_coords = tsp_solver.coordinates[list(path)]

        # Identify the current city index in the path
        current_city_idx = path[segment_idx] if segment_idx < len(path) else None

        # Create a mask to exclude the current city from the scatter plot
        mask = np.ones(len(tsp_solver.coordinates), dtype=bool)
        if current_city_idx is not None:
            mask[current_city_idx] = False

        # Plot all cities except the current one as blue circles
        ax_tour.scatter(tsp_solver.coordinates[mask, 0], tsp_solver.coordinates[mask, 1], 
                        c=style['city_color'], s=style['city_size'], label='Cities')
        
        # Annotate city indices
        for i, (x, y) in enumerate(tsp_solver.coordinates):
            ax_tour.annotate(f"{i}", (x, y), fontsize=style['font_size'])
        
        # Calculate color based on generation (lighter for early, darker for later)
        progress = path_idx / (len(tsp_solver.all_paths) - 1) if len(tsp_solver.all_paths) > 1 else 1
        color = (1 - progress * 0.5, 0, progress * 0.5)

        # Draw the path
        if segment_idx == frames_per_path - 1:  # Last frame: close the loop
            # Append the first city to the end to close the loop
            closed_path_coords = np.vstack([path_coords, path_coords[0]])
            ax_tour.plot(closed_path_coords[:, 0], closed_path_coords[:, 1], 
                         color=color, linestyle='-', linewidth=2, label='Tour')
        elif segment_idx > 0:  # Intermediate frames: draw partial path
            partial_path_coords = path_coords[:segment_idx]
            ax_tour.plot(partial_path_coords[:, 0], partial_path_coords[:, 1], 
                         color=color, linestyle='-', linewidth=2, label='Tour')

        # Plot the current city as a green star only, but not on the last frame
        if segment_idx < len(path):  # Only show current city marker if not on the last frame
            current_city = path_coords[segment_idx]
            ax_tour.scatter(current_city[0], current_city[1], c=style['next_city_color'], s=style['city_size']*1.5, marker='*', 
                            label='Current City')

        # Update title with generation and distance
        ax_tour.set_title(f'Generation {path_idx+1}/{len(tsp_solver.all_paths)} - '
                         f'Distance: {tsp_solver.path_distances[path_idx]:.2f}',
                         fontsize=style['title_size'])
        ax_tour.set_xlabel('X Coordinate', fontsize=style['label_size'])
        ax_tour.set_ylabel('Y Coordinate', fontsize=style['label_size'])
        ax_tour.grid(True, linestyle=style['grid_linestyle'], alpha=style['grid_alpha'], color=style['grid_color'])
        ax_tour.legend(fontsize=style['legend_size'])

        # Highlight the current generation on the fitness plot
        ax_fitness.clear()
        ax_fitness.plot(generations, distances, 'b-', label='Best Distance')
        ax_fitness.scatter([path_idx], [distances[path_idx]], c='red', s=style['city_size'], marker='o', 
                          label='Current Generation')
        ax_fitness.set_xlabel('Sampled Generation', fontsize=style['label_size'])
        ax_fitness.set_ylabel('Distance', fontsize=style['label_size'])
        ax_fitness.set_title('Genetic Algorithm Convergence', fontsize=style['title_size'])
        ax_fitness.grid(True, linestyle=style['grid_linestyle'], alpha=style['grid_alpha'], color=style['grid_color'])
        ax_fitness.legend(fontsize=style['legend_size'])

    # Create the animation
    anim = FuncAnimation(fig, update, frames=total_frames, interval=style['animation_interval'], repeat=style['repeat'])

    # Show the animation
    plt.show()

if __name__ == "__main__": 
    # Create and solve a TSP instance
    tsp = TSPSolver(num_cities=10)
    tsp.solve_genetic(pop_size=100, generations=500, mutation_rate=0.01, visualize_steps=True, max_paths_to_store=20)
    
    # Visualize the final solution
    visualize_tsp(tsp, save_path="tsp_solution.png")
    
    # Animate the solution process
    animate_tsp(tsp, save_path="tsp_genetic.gif")