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
        
        
        return self.best_path, self.best_distance
    
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
