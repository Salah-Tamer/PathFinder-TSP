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
    'path_color': '#e74c3c',  # brute_force color
    'best_path_color': '#2ecc71',  # success color
    'new_best_color': '#28a745',  # success color
    'city_size': 200,
    'city_edge_color': 'white',
    'city_edge_width': 2,
    'city_alpha': 0.8,
    'font_size': 9,
    'animation_interval': 1000,
    'repeat': False,
    'grid_alpha': 0.3,
    'grid_color': 'gray',
    'grid_linestyle': '--',
    'x_min': -5,
    'x_max': 105,
    'y_min': -5,
    'y_max': 105,
    'title_size': 18,
    'title': "TSP Brute Force Solution",
    'label_size': 14,
    'legend_size': 8,
    'legend_loc': 'lower right',
    'text_size': 9
}

def visualize_brute_force(coordinates=None, custom_style=None, show_plot=False, fig=None, ax=None, update_status_callback=None):
    """
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
    
    # Initialize style with defaults
    style = DEFAULT_STYLE.copy()
    
    # Update style with custom settings if provided
    if custom_style:
        style.update(custom_style)
    
    coords_array = np.array(coordinates)  # Precompute NumPy array

    # Create and solve TSP
    solver = TSPSolver(coordinates=coordinates)
    solver.solve_brute_force(visualize_steps=True)

    # Validate solver output
    if not solver.all_paths or len(solver.all_paths) != len(solver.path_distances):
        raise ValueError("No valid paths or mismatched paths and distances from solver")

    # Process paths for animation
    paths_to_show = []
    best_so_far = float('inf')
    
    # Process the paths tracked by the solver
    for i, (path, distance) in enumerate(zip(solver.all_paths, solver.path_distances)):
        # Check if this is a new best path
        is_new_best = distance < best_so_far
        
        if is_new_best:
            best_so_far = distance
            title = "NEW BEST PATH"
        else:
            title = f"Path Option {i+1}"
            
        # The last path should be the optimal path
        if i == len(solver.all_paths) - 1:
            title = "OPTIMAL PATH"
            
        paths_to_show.append((path, title, distance))

    # Create figure and axis or use provided ones
    if fig is None or ax is None:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=style['fig_size'])
        fig.patch.set_facecolor(style['background_color'])
        
        # Create metrics box for standalone mode
        metrics_box = plt.axes([0.3, 0.85, 0.4, 0.08], frameon=True)
        metrics_box.axis('off')
        metrics_bg = plt.Rectangle((0, 0), 1, 1, transform=metrics_box.transAxes,
                                facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.9,
                                linewidth=1.5, zorder=1)
        metrics_box.add_patch(metrics_bg)
        
        # Text elements for metrics box
        current_label = metrics_box.text(0.2, 0.3, 'Distance:', fontsize=style['text_size'], 
                                      fontweight='bold', ha='right', zorder=2)
        distance_text = metrics_box.text(0.25, 0.3, '', fontsize=style['text_size'], ha='left', zorder=2)
        
        best_label = metrics_box.text(0.6, 0.3, 'Best :', fontsize=style['text_size'], 
                                   fontweight='bold', ha='right', zorder=2)
        best_text = metrics_box.text(0.65, 0.3, '', fontsize=style['text_size'], ha='left', zorder=2)
    else:
        # Clear the axes for embedded mode
        ax.clear()
        # No metrics box in embedded mode since we'll use the app's status display
        distance_text = None
        best_text = None
    
    # Main plot area styling
    ax.set_facecolor(style['plot_bg_color'])
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    # Legend
    city_dot = mpatches.Patch(color=style['city_color'], label='Cities')
    path_line = mpatches.Patch(color=style['path_color'], label='Current Path')
    best_path_line = mpatches.Patch(color=style['best_path_color'], label='New Best/Optimal Path')

    # Plot elements
    line, = ax.plot([], [], '-', color=style['path_color'], linewidth=2.5, zorder=5)  # Unpack Line2D directly
    scatter = ax.scatter([c[0] for c in coordinates], [c[1] for c in coordinates], 
                       c=style['city_color'], s=style['city_size'], zorder=10,
                       edgecolor=style['city_edge_color'], linewidth=style['city_edge_width'],
                       alpha=style['city_alpha'])
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(f"{i}", (x, y), fontsize=style['font_size'], fontweight='bold', 
                   ha='center', va='center', color='white')
    
    # Progress indicator
    progress_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=style['text_size'],
                          ha='left', va='bottom', zorder=2,
                          bbox=dict(facecolor='white', alpha=0.7, 
                                   edgecolor='#cccccc', boxstyle='round,pad=0.2'))

    # Title text
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, fontsize=style['text_size'],
                        fontweight='bold', ha='center')

    # Dynamic elements
    city_markers = []
    
    # Arrow directions for current path (will be updated)
    arrows = []
    
    # Track the best distance found so far
    best_so_far = float('inf')

    def init():
        # Dynamic axis limits
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
        
        ax.legend(handles=[city_dot, path_line, best_path_line], 
                loc=style['legend_loc'], framealpha=0.9, frameon=True, fontsize=style['legend_size'])
        line.set_data([], [])
        
        # Clear any existing arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        
        # Clear any existing city markers
        for marker in city_markers:
            marker.remove()
        city_markers.clear()
        
        # Initialize text elements
        title_text.set_text('')
        if distance_text:
            distance_text.set_text('')
        if best_text:
            best_text.set_text('')
        progress_text.set_text('')
        
        return_elements = [line, title_text, progress_text]
        if distance_text:
            return_elements.append(distance_text)
        if best_text:
            return_elements.append(best_text)
            
        return return_elements

    def update(frame):
        nonlocal best_so_far
        path, title, distance = paths_to_show[frame]
        
        # Check if this is a new best path
        is_new_best = distance < best_so_far
        if is_new_best:
            best_so_far = distance

        # Color and linewidth based on path type
        if "OPTIMAL" in title:
            color = style['best_path_color']
            lw = 3.5
        elif is_new_best:
            color = style['new_best_color']
            lw = 3
        else:
            color = style['path_color']
            lw = 2.5

        # Update path (close the loop)
        path_coords = coords_array[list(path) + [path[0]]]
        line.set_data(path_coords[:, 0], path_coords[:, 1])
        line.set_color(color)
        line.set_linewidth(lw)

        # Clear arrows and markers
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        for marker in city_markers:
            marker.remove()
        city_markers.clear()

        # Add arrows and markers
        for i in range(len(path)-1):
            x1, y1 = path_coords[i]
            x2, y2 = path_coords[i+1]
            dx, dy = x2-x1, y2-y1
            
            # Calculate length for scaling
            length = np.sqrt(dx**2 + dy**2)
            scale = 0.8 if length > 20 else 0.6
            
            # Create and store arrow
            arrow = ax.arrow(x1, y1, dx*scale, dy*scale, 
                           head_width=2, head_length=3, 
                           fc=color, ec=color, zorder=15)
            arrows.append(arrow)
            if i > 0:
                # Add small circle markers at city points without numeric labels
                marker = ax.add_patch(plt.Circle((x1, y1), 1.5, color=color, zorder=20))
                city_markers.append(marker)

        # Update text
        title_text.set_text(title)
        if "OPTIMAL" in title or is_new_best:
            title_text.set_color(style['best_path_color'])
        else:
            title_text.set_color('#212529')
            
        # Update metrics display for standalone mode
        if distance_text:
            distance_text.set_text(f"{distance:.2f}")
            if "OPTIMAL" in title or is_new_best:
                distance_text.set_color(style['best_path_color'])
                distance_text.set_fontweight('bold')
            else:
                distance_text.set_color('#212529')
                distance_text.set_fontweight('normal')
                
        if best_text:
            if best_so_far < float('inf'):
                best_text.set_text(f"{best_so_far:.2f}")
                best_text.set_color(style['best_path_color'])
                best_text.set_fontweight('bold')
            else:
                best_text.set_text("N/A")
                best_text.set_color('#6c757d')
                
        progress_text.set_text(f"Step {frame+1} of {len(paths_to_show)}")

        # Call the status update callback if provided
        if update_status_callback:
            update_status_callback(title, distance, frame * 0.5)  # Estimated time based on frame

        return_elements = [line, title_text, progress_text]
        if distance_text:
            return_elements.append(distance_text)
        if best_text:
            return_elements.append(best_text)
            
        # Add all arrows to return elements
        return_elements.extend(arrows)
        
        return return_elements

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(paths_to_show),
                     init_func=init, blit=True, interval=style['animation_interval'], 
                     repeat=style['repeat'])
    
    if show_plot:
        plt.suptitle('TSP Using Brute Force Algorithm', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.0, 1, 0.95])
        plt.show()
        
    return ani

def main():
    fig, ax, ani = visualize_brute_force()
    plt.show()

if __name__ == "__main__":
    # When run as a standalone script, show the plot
    visualize_brute_force(show_plot=True)