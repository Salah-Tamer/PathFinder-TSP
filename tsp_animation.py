from tsp_solver import TSPSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

def main():
    # Create example with 5 cities in a star pattern
    coordinates = [
        (30, 30),  # center
        (10, 30),  # left
        (50, 30),  # right
        (30, 10),  # bottom
        (30, 50),  # top
    ]
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

    # Create figure and axis
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    
    # Main plot area styling
    ax.set_facecolor('#f8f8f8')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    # Legend
    city_dot = mpatches.Patch(color='blue', label='Cities')
    path_line = mpatches.Patch(color='red', label='Current Path')
    best_path_line = mpatches.Patch(color='green', label='New Best/Optimal Path')

    # Plot elements
    line, = ax.plot([], [], 'r-', linewidth=2.5, zorder=5)  # Unpack Line2D directly
    scatter = ax.scatter([c[0] for c in coordinates], [c[1] for c in coordinates], 
                        c='blue', s=200, zorder=10)
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(f"{i}", (x, y), fontsize=16, fontweight='bold', 
                    ha='center', va='center', color='white')

    # Metrics display
    metrics_box = plt.axes([0.3, 0.85, 0.4, 0.08], frameon=True)
    metrics_box.axis('off')
    metrics_bg = plt.Rectangle((0, 0), 1, 1, transform=metrics_box.transAxes,
                              facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.9,
                              linewidth=1.5, zorder=1)
    metrics_box.add_patch(metrics_bg)

    # Text elements
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, fontsize=12,
                         fontweight='bold', ha='center')
    current_label = metrics_box.text(0.2, 0.3, 'Distance:', fontsize=12, 
                                    fontweight='bold', ha='right', zorder=2)
    distance_text = metrics_box.text(0.25, 0.3, '', fontsize=12, ha='left', zorder=2)
    
    best_label = metrics_box.text(0.6, 0.3, 'Best :', fontsize=12, 
                                fontweight='bold', ha='right', zorder=2)
    best_text = metrics_box.text(0.65, 0.3, '', fontsize=12, ha='left', zorder=2)
    
    # Progress indicator
    progress_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                           ha='left', va='bottom', zorder=2,
                           bbox=dict(facecolor='white', alpha=0.7, 
                                     edgecolor='#cccccc', boxstyle='round,pad=0.2'))

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
        margin = 10
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(handles=[city_dot, path_line, best_path_line], 
                 loc='upper right', framealpha=0.9, frameon=True, fontsize=10)
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
        distance_text.set_text('')
        best_text.set_text('')
        progress_text.set_text('')
        return [line, title_text, distance_text, best_text, progress_text]

    def update(frame):
        nonlocal best_so_far
        path, title, distance = paths_to_show[frame]
        
        # Check if this is a new best path
        is_new_best = distance < best_so_far
        if is_new_best:
            best_so_far = distance

        # Color and linewidth
        if "OPTIMAL" in title:
            color = '#28a745'
            lw = 3.5
        elif is_new_best:
            color = '#5cb85c'
            lw = 3
        else:
            color = '#dc3545'
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
                marker = ax.add_patch(plt.Circle((x1, y1), 1.5, color=color, zorder=20))
                city_markers.append(marker)
                step_label = ax.annotate(f"{i}", 
                                       (x1, y1), 
                                       xytext=(x1+2, y1+2),
                                       fontsize=10, 
                                       color='white',
                                       fontweight='bold',
                                       bbox=dict(facecolor=color, alpha=1, boxstyle='circle'),
                                       zorder=25)
                city_markers.append(step_label)

        # Update text
        title_text.set_text(title)
        if "OPTIMAL" in title or is_new_best:
            title_text.set_color('#28a745')
        else:
            title_text.set_color('#212529')
        distance_text.set_text(f"{distance:.2f}")
        if "OPTIMAL" in title or is_new_best:
            distance_text.set_color('#28a745')
            distance_text.set_fontweight('bold')
        else:
            distance_text.set_color('#212529')
            distance_text.set_fontweight('normal')
        if best_so_far < float('inf'):
            best_text.set_text(f"{best_so_far:.2f}")
            best_text.set_color('#28a745')
            best_text.set_fontweight('bold')
        else:
            best_text.set_text("N/A")
            best_text.set_color('#6c757d')
        progress_text.set_text(f"Step {frame+1} of {len(paths_to_show)}")

        return [line, title_text, distance_text, best_text, progress_text]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(paths_to_show),
                       init_func=init, blit=True, interval=2000, repeat=True)
    plt.suptitle('TSP Using Nearest Neighbor', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()