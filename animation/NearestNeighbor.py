import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TSPSolver import TSPSolver

# Default style configuration for TSP visualizations
DEFAULT_STYLE = {
    'fig_size': (8, 6),
    'background_color': 'white',
    'plot_bg_color': '#f8f8f8',
    'city_color': '#4a86e8',  # accent color
    'path_color': '#2ecc71',  # nearest_neighbor color
    'current_city_color': 'red',
    'next_city_color': '#28a745',  # success color
    'city_size': 120,
    'city_edge_color': 'white',
    'city_edge_width': 2,
    'city_alpha': 0.8,
    'font_size': 9,
    'animation_interval': 500,
    'repeat': False,
    'grid_alpha': 0.3,
    'grid_color': 'gray',
    'grid_linestyle': '--',
    'x_min': -5,
    'x_max': 105,
    'y_min': -5,
    'y_max': 105,
    'title_size': 18,
    'title': "TSP Nearest Neighbor Solution",
    'label_size': 14,
    'legend_size': 8,
    'legend_loc': 'lower right',
    'text_size': 9
}

def visualize_nearest_neighbor(coordinates=None, custom_style=None, show_plot=False, fig=None, ax=None, update_status_callback=None):
    """
    Visualize the Nearest Neighbor algorithm solving the TSP problem
    
    Args:
        coordinates: Optional list of city coordinates. If None, defaults to a star pattern.
        custom_style: Optional dictionary with custom styling parameters.
        show_plot: Whether to call plt.show() at the end (True for standalone, False for embedding)
        fig: Optional matplotlib Figure to use instead of creating a new one
        ax: Optional matplotlib Axes to use instead of creating new ones
        update_status_callback: Optional callback function to update UI status with (title, distance, time) parameters
    
    Returns:
        The animation object for further manipulation
    """
    # Use provided coordinates or create default example
    if coordinates is None:
        coordinates = [
            (30, 30),  # center
            (10, 30),  # left
            (50, 30),  # right
            (30, 10),  # bottom
            (30, 50),  # top
        ]
    
    # Initialize style with defaults
    style = DEFAULT_STYLE.copy()
    
    # Update style with custom settings if provided
    if custom_style:
        style.update(custom_style)
    
    coords_array = np.array(coordinates)

    solver = TSPSolver(coordinates=coordinates)
    solver.solve_nearest_neighbor(visualize_steps=True)

    if not solver.all_paths:
        print("No paths recorded. Check TSPSolver implementation.")
        return None

    # Create or use figure and axes
    if fig is None or ax is None:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=style['fig_size'])
        fig.patch.set_facecolor(style['background_color'])
    else:
        # Clear the axes for embedded mode
        ax.clear()
    
    ax.set_facecolor(style['plot_bg_color'])

    city_dot = mpatches.Patch(color=style['city_color'], label='Cities')
    path_line = mpatches.Patch(color=style['path_color'], label='Current Path')
    current_city = mpatches.Patch(color=style['current_city_color'], label='Current City')
    next_city = mpatches.Patch(color=style['next_city_color'], label='Next City')

    scatter = ax.scatter(coords_array[:, 0], coords_array[:, 1], 
                       c=style['city_color'], s=style['city_size'], zorder=10,
                       edgecolor=style['city_edge_color'], linewidth=style['city_edge_width'],
                       alpha=style['city_alpha'])
    
    # Create completely separate artists for the path line and markers
    # Important: use explicit empty arrays rather than [], [] to avoid any automatic markers
    path_line = ax.add_line(plt.Line2D(np.array([]), np.array([]), 
                          color=style['path_color'], linewidth=3.5, zorder=5, solid_capstyle='round'))
    
    # Initialize empty markers for current and next cities
    current_marker = ax.scatter(np.array([]), np.array([]), color=style['current_city_color'], 
                              s=250, zorder=15, marker='o', edgecolors='white', linewidth=2)
    next_marker = ax.scatter(np.array([]), np.array([]), color=style['next_city_color'], 
                           s=250, zorder=15, marker='o', edgecolors='white', linewidth=2)
    
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, 
                       fontsize=style['text_size'], fontweight='bold', ha='center')
    progress_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                          fontsize=style['text_size'], ha='left', va='bottom',
                          bbox=dict(facecolor='white', alpha=0.7, 
                                   edgecolor='#cccccc', boxstyle='round,pad=0.2'))

    def init():
        x_min, x_max = coords_array[:, 0].min(), coords_array[:, 0].max()
        y_min, y_max = coords_array[:, 1].min(), coords_array[:, 1].max()
        margin = max(10, (x_max - x_min) * 0.1)  # Adaptive margin
        
        # Use provided limits if available, otherwise calculate based on data
        ax_x_min = style['x_min'] if style['x_min'] is not None else x_min - margin
        ax_x_max = style['x_max'] if style['x_max'] is not None else x_max + margin
        ax_y_min = style['y_min'] if style['y_min'] is not None else y_min - margin
        ax_y_max = style['y_max'] if style['y_max'] is not None else y_max + margin
        
        ax.set_xlim(ax_x_min, ax_x_max)
        ax.set_ylim(ax_y_min, ax_y_max)
        ax.set_xlabel('X Coordinate', fontsize=style['label_size'])
        ax.set_ylabel('Y Coordinate', fontsize=style['label_size'])
        
        # Apply grid styling
        ax.grid(True, linestyle=style['grid_linestyle'], alpha=style['grid_alpha'], color=style['grid_color'])
        
        # Set title
        ax.set_title(style['title'], fontsize=style['title_size'], fontweight='bold')
        
        # Remove spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create new legend objects to ensure clean display
        city_dot_legend = mpatches.Patch(color=style['city_color'], label='Cities')
        path_line_legend = mpatches.Patch(color=style['path_color'], label='Current Path')
        current_city_legend = mpatches.Patch(color=style['current_city_color'], label='Current City')
        next_city_legend = mpatches.Patch(color=style['next_city_color'], label='Next City')
        
        ax.legend(handles=[city_dot_legend, path_line_legend, current_city_legend, next_city_legend], 
                loc=style['legend_loc'], framealpha=0.9, fontsize=style['legend_size'])

        # Reset all elements to initial state
        path_line.set_data(np.array([]), np.array([]))
        current_marker.set_offsets(np.empty((0, 2)))
        next_marker.set_offsets(np.empty((0, 2)))
        title_text.set_text('')
        progress_text.set_text('')
        
        return path_line, current_marker, next_marker, title_text, progress_text

    def update(frame):
        path = solver.all_paths[frame]
        distance = solver.path_distances[frame]
        current_path_length = min(frame + 2, len(path))
        current_path = list(path[:current_path_length])
        path_coords = coords_array[current_path]

        # Update path line data
        path_line.set_data(path_coords[:, 0], path_coords[:, 1])

        # Clear all markers first to avoid artifacts
        current_marker.set_offsets(np.empty((0, 2)))
        next_marker.set_offsets(np.empty((0, 2)))

        # Only then set new marker positions
        if len(current_path) >= 2:
            if frame < len(solver.all_paths) - 1:
                current_city_idx = current_path[-1]
                current_pos = coords_array[current_city_idx].reshape(1, 2)
                current_marker.set_offsets(current_pos)
            else:
                # On the last frame, do not show any markers
                current_marker.set_offsets(np.empty((0, 2)))
                next_marker.set_offsets(np.empty((0, 2)))
                return path_line, current_marker, next_marker, title_text, progress_text
        
        if frame < len(solver.all_paths) - 1:
            next_path = solver.all_paths[frame + 1]
            next_city_idx = next_path[current_path_length] if current_path_length < len(next_path) else path[0]
            next_pos = coords_array[next_city_idx].reshape(1, 2)
            next_marker.set_offsets(next_pos)
        else:
            # On the last frame, do not show any markers
            next_marker.set_offsets(np.empty((0, 2)))

        # Set title text and format
        step_text = "Final Path" if frame == len(solver.all_paths) - 1 else f"Step {frame+1}/{len(solver.all_paths)-1}"
        title_text.set_text(f"{step_text} - Distance: {distance:.2f}")
        progress_text.set_text(f"Step {frame+1} of {len(solver.all_paths)}")

        # Call the status update callback if provided
        if update_status_callback:
            status_title = f"Nearest Neighbor - {step_text}"
            update_status_callback(status_title, distance, frame * 0.2)  # Estimated time

        # Return all the artists that need to be updated
        return path_line, current_marker, next_marker, title_text, progress_text

    ani = FuncAnimation(fig, update, frames=len(solver.all_paths), 
                     init_func=init, blit=True, 
                     interval=style['animation_interval'], 
                     repeat=style['repeat'])
    
    if show_plot:
        plt.suptitle('TSP Using Nearest Neighbor', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.0, 1, 0.95])
        plt.show()
    
    return ani

if __name__ == "__main__":
    # When run as a standalone script, show the plot
    visualize_nearest_neighbor(show_plot=True)