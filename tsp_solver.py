import numpy as np
import math 
import matplotlib.pyplot as plt
import itertools
import time
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

class TSPSolver:
    def __init__(self, coordinates=None, num_cities=None):
        """
        Initialize the TSP Solver
        
        Args:
            coordinates: List of (x, y) coordinates for each city
            num_cities: Number of cities to generate randomly if coordinates not provided
        """
        if coordinates is not None:
            self.coordinates = np.array(coordinates)
            self.num_cities = len(coordinates)
        elif num_cities is not None:
            # Generate random coordinates
            self.num_cities = num_cities
            self.coordinates = np.random.rand(num_cities, 2) * 100
        else:
            raise ValueError("Either coordinates or num_cities must be provided")
            
        # Calculate the distance matrix
        self.distances = self._calculate_distances()
        
        # Best solution found
        self.best_distance = float('inf')
        self.best_path = None
        
        # For visualization
        self.all_paths = []
        self.path_distances = []
        
    def _calculate_distances(self):
        """Calculate the distance matrix between all pairs of cities"""
        n = self.num_cities
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Euclidean distance between cities i and j
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def _calculate_path_distance(self, path):
        """Calculate the total distance of a path using pre-computed distances"""
        return sum(self.distances[path[i], path[i+1]] for i in range(len(path)-1))
    
    def solve_brute_force(self, visualize_steps=False, max_paths_to_store=100):
        """
        Solve TSP using brute force approach (all permutations)
        
        Args:
            visualize_steps: If True, store paths for step-by-step visualization
            max_paths_to_store: Maximum number of paths to store for visualization
        """
        cities = list(range(1, self.num_cities))  # Exclude city 0 (starting city)
        best_distance = float('inf')
        best_path = None
        start_time = time.time()
        
        # Clear previous paths
        self.all_paths = []
        self.path_distances = []
        
        # Generate all permutations but don't store them all in memory at once
       
        total_perms = math.factorial(self.num_cities - 1)
        
        if visualize_steps:
            # Keep track of best paths and a sample of others
            best_paths = []
            best_distances = []
            sample_indices = set(np.linspace(0, total_perms-1, min(max_paths_to_store//2, total_perms), 
                                         dtype=int))
            
        for idx, path in enumerate(itertools.permutations(cities)):
            # Complete path: start with 0, go through the permutation, and return to 0
            full_path = (0,) + path + (0,)
            
            # Calculate total distance
            distance = self._calculate_path_distance(full_path)
            
            # Update best solution if better
            if distance < best_distance:
                best_distance = distance
                best_path = full_path
                
                # Store new best paths
                if visualize_steps:
                    best_paths.append(full_path)
                    best_distances.append(distance)
            # Store some non-best paths too based on pre-calculated sample indices
            elif visualize_steps and idx in sample_indices:
                best_paths.append(full_path)
                best_distances.append(distance)
        
        if visualize_steps:
            # Combine best and sampled paths, sorting by discovery order
            path_data = [(p, d, i) for i, (p, d) in enumerate(zip(best_paths, best_distances))]
            path_data.sort(key=lambda x: x[2])  # Sort by original index
            
            # Get final paths for visualization
            self.all_paths = [p for p, _, _ in path_data]
            self.path_distances = [d for _, d, _ in path_data]
        
        end_time = time.time()
        
        self.best_distance = best_distance
        self.best_path = best_path
        
        print(f"Best time {end_time - start_time:.4f} seconds")
        print(f"Best distance: {best_distance:.2f}")
        
        return best_path, best_distance
    
    def solve_nearest_neighbor(self, start_city=0, visualize_steps=False):
        """
        Solve TSP using the Nearest Neighbor heuristic.
        
        Args:
            start_city: Index of the starting city (default is 0)
            visualize_steps: If True, store paths for step-by-step visualization
        """
        unvisited = set(range(self.num_cities))
        path = [start_city]
        current_city = start_city
        unvisited.remove(current_city)
        total_distance = 0
        
        if visualize_steps:
            self.all_paths = []
            self.path_distances = []

        while unvisited:
            next_city = min(unvisited, key=lambda city: self.distances[current_city][city])
            total_distance += self.distances[current_city][next_city]
            current_city = next_city
            path.append(current_city)
            unvisited.remove(current_city)
            
            if visualize_steps:
                temp_path = path + [start_city]
                self.all_paths.append(tuple(temp_path))
                self.path_distances.append(total_distance + self.distances[current_city][start_city])
        
        # Return to start city
        path.append(start_city)
        total_distance += self.distances[current_city][start_city]

        self.best_path = tuple(path)
        self.best_distance = total_distance
        
        print(f"Nearest Neighbor solution from city {start_city}:")
        print(f"Path: {self.best_path}")
        print(f"Total Distance: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance
    
    def visualize(self, path=None, save_path=None):
        """Visualize the TSP solution"""
        plt.figure(figsize=(10, 8))
        
        # Plot all cities
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                    c='blue', s=100, label='Cities')
        
        # Add city labels
        for i, (x, y) in enumerate(self.coordinates):
            plt.annotate(f"{i}", (x, y), fontsize=12)
        
        # If a path is provided, plot the tour
        if path is not None:
            path_coords = self.coordinates[list(path)]
            plt.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2, label='Tour')
        elif self.best_path is not None:
            path_coords = self.coordinates[list(self.best_path)]
            plt.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2, label='Tour')
        
        plt.title(f'TSP Solution - {self.num_cities} Cities - Distance: {self.best_distance:.2f}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def visualize_step_by_step(self, interval=500, save_animation=None):
        """
        Create a step-by-step animation of the path search process
        
        Args:
            interval: Time interval between frames in milliseconds
            save_animation: Filename to save the animation as MP4 (requires ffmpeg)
        """
        if not self.all_paths:
            print("No path data available. Run solve_brute_force(visualize_steps=True) first.")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set axis limits with padding
        x_min, x_max = self.coordinates[:, 0].min(), self.coordinates[:, 0].max()
        y_min, y_max = self.coordinates[:, 1].min(), self.coordinates[:, 1].max()
        
        padding = max((x_max - x_min), (y_max - y_min)) * 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Plot cities
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                  c='blue', s=100, zorder=10)
        
        # Add city labels
        for i, (x, y) in enumerate(self.coordinates):
            ax.annotate(f"{i}", (x, y), fontsize=12, weight='bold')
        
        # Add title and labels
        title = ax.set_title('TSP Step-by-Step Visualization', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True)
        
        # Create legend
        red_patch = mpatches.Patch(color='red', label='Current Path')
        green_patch = mpatches.Patch(color='green', label='Best Path')
        blue_dot = mpatches.Patch(color='blue', label='Cities')
        ax.legend(handles=[blue_dot, red_patch, green_patch], loc='upper right')
        
        # Info text box
        info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Progress information
        progress_text = ax.text(0.5, 0.02, '', transform=ax.transAxes, fontsize=12,
                              ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Create line for the path
        line, = ax.plot([], [], 'r-', linewidth=2.5)
        
        # Track best distance so far to colorize new best paths
        path_with_status = []
        best_so_far = float('inf')
        
        for idx, (path, distance) in enumerate(zip(self.all_paths, self.path_distances)):
            is_new_best = distance < best_so_far
            if is_new_best:
                best_so_far = distance
            path_with_status.append((path, distance, is_new_best))
        
        def init():
            """Initialize the animation"""
            line.set_data([], [])
            info_text.set_text('')
            progress_text.set_text('')
            return line, info_text, progress_text, title
            
        def update(frame):
            """Update function for animation frames"""
            path, distance, is_new_best = path_with_status[frame]
            
            # Update path visualization
            path_coords = self.coordinates[list(path)]
            line.set_data(path_coords[:, 0], path_coords[:, 1])
            
            # Color best paths in green, others in red
            if is_new_best:
                line.set_color('green')
                status = "NEW BEST PATH!"
            else:
                line.set_color('red')
                status = "Evaluating path"
            
            # Update info text
            info_str = f"Path: {' → '.join(map(str, path))}\nDistance: {distance:.2f}\n{status}"
            info_text.set_text(info_str)
            
            # Update progress text
            total_frames = len(path_with_status)
            progress_str = f"Progress: {frame+1}/{total_frames} paths ({(frame+1)/total_frames*100:.1f}%)"
            progress_text.set_text(progress_str)
            
            # Update title
            title.set_text(f'TSP Step-by-Step - Path {frame+1}/{total_frames}')
            
            return line, info_text, progress_text, title
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(path_with_status),
                           init_func=init, blit=True, interval=interval)
        
        # Save animation if requested
        if save_animation:
            anim.save(save_animation, writer='ffmpeg', fps=1000/interval)
        
        plt.tight_layout()
        plt.show()
    
    def get_solution_info(self):
        """Return a dictionary with solution information"""
        if self.best_path is None:
            return {"error": "No solution available. Run solve_brute_force() first."}
            
        return {
            "best_path": self.best_path,
            "best_distance": self.best_distance,
            "num_cities": self.num_cities,
            "coordinates": self.coordinates.tolist()
        }
