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
from tsp_solver import TSPSolver

def visualize_nearest_neighbor():
    coordinates = [
        (30, 30), 
        (10, 30), 
        (50, 30), 
        (30, 10), 
        (30, 50)
    ]
    coords_array = np.array(coordinates)

    solver = TSPSolver(coordinates=coordinates)
    solver.solve_nearest_neighbor(visualize_steps=True)

    if not solver.all_paths:
        print("No paths recorded. Check TSPSolver implementation.")
        return

    print(f"Number of steps: {len(solver.all_paths)}")
    print(f"Final distance: {solver.best_distance:.2f}")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')

    city_dot = mpatches.Patch(color='blue', label='Cities')
    path_line = mpatches.Patch(color='#ff7f0e', label='Current Path')
    current_city = mpatches.Patch(color='red', label='Current City')
    next_city = mpatches.Patch(color='green', label='Next City')

    scatter = ax.scatter(coords_array[:, 0], coords_array[:, 1], c='blue', s=200, zorder=10)
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(f"{i}", (x, y), fontsize=16, fontweight='bold', ha='center', va='center', color='white')

    line, = ax.plot([], [], 'o-', color='#ff7f0e', linewidth=2.5, zorder=5)
    current_marker = ax.scatter([], [], color='red', s=300, zorder=15, marker='o', edgecolors='white', linewidth=2)
    next_marker = ax.scatter([], [], color='green', s=300, zorder=15, marker='o', edgecolors='white', linewidth=2)
    arrow_marker = ax.arrow(0, 0, 0, 0, head_width=2, head_length=3, fc='#ff7f0e', ec='#ff7f0e', zorder=15)
    arrow_marker.set_visible(False)

    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, fontsize=12, fontweight='bold', ha='center')
    progress_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10, ha='left', va='bottom',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='#cccccc', boxstyle='round,pad=0.2'))

    def init():
        x_min, x_max = coords_array[:, 0].min(), coords_array[:, 0].max()
        y_min, y_max = coords_array[:, 1].min(), coords_array[:, 1].max()
        margin = 10
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(handles=[city_dot, path_line, current_city, next_city], loc='upper right', framealpha=0.9, fontsize=10)

        line.set_data([], [])
        current_marker.set_offsets(np.empty((0, 2)))
        next_marker.set_offsets(np.empty((0, 2)))
        arrow_marker.set_visible(False)
        title_text.set_text('')
        progress_text.set_text('')
        return line, current_marker, next_marker, arrow_marker, title_text, progress_text

    def update(frame):
        path = solver.all_paths[frame]
        distance = solver.path_distances[frame]
        current_path_length = min(frame + 2, len(path))
        current_path = list(path[:current_path_length])
        path_coords = coords_array[current_path]

        line.set_data(path_coords[:, 0], path_coords[:, 1])

        if len(current_path) >= 2:
            current_city_idx = current_path[-1] if frame < len(solver.all_paths) - 1 else current_path[-2]
            current_marker.set_offsets([coords_array[current_city_idx]])
        else:
            current_marker.set_offsets(np.empty((0, 2)))

        if frame < len(solver.all_paths) - 1:
            next_path = solver.all_paths[frame + 1]
            next_city_idx = next_path[current_path_length] if current_path_length < len(next_path) else path[0]
        else:
            next_city_idx = path[0]
        next_marker.set_offsets([coords_array[next_city_idx]])

        if len(current_path) >= 2:
            x1, y1 = path_coords[-2]
            x2, y2 = path_coords[-1]
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            scale = 0.8 if length > 20 else 0.6
            arrow_marker.set_data(x=x1, y=y1, dx=dx*scale, dy=dy*scale)
            arrow_marker.set_visible(True)
        else:
            arrow_marker.set_visible(False)

        title_text.set_text(f"Building Path - Step {frame+1}/{len(solver.all_paths)-1}" if frame < len(solver.all_paths) - 1 else "Final Path")
        progress_text.set_text(f"Step {frame+1} of {len(solver.all_paths)}")

        return line, current_marker, next_marker, arrow_marker, title_text, progress_text

    ani = FuncAnimation(fig, update, frames=len(solver.all_paths), init_func=init, blit=True, interval=500, repeat=False)
    plt.suptitle('TSP Using Nearest Neighbor', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    visualize_nearest_neighbor()