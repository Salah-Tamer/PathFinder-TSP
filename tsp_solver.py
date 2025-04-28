
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import time
import random
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
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def _calculate_path_distance(self, path):
        """Calculate the total distance of a path using pre-computed distances"""
        return sum(self.distances[path[i], path[i+1]] for i in range(len(path)-1))
    
    def create_population(self, pop_size):
        """Create an initial population of random routes"""
        population = []
        for _ in range(pop_size):
            route = random.sample(range(self.num_cities), self.num_cities)
            population.append(route)
        return population
    
    def select_parents(self, population):
        """Select two parents using fitness-based probability"""
        fitness_scores = [1 / self._calculate_path_distance(list(route) + [route[0]]) for route in population]
        total_fitness = sum(fitness_scores)
        probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
        parents = random.choices(population, probabilities, k=2)
        return parents
    
    def crossover(self, parent1, parent2):
        """Perform crossover to create an offspring"""
        start, end = sorted(random.sample(range(len(parent1)), 2))
        offspring = [-1] * len(parent1)
        offspring[start:end+1] = parent1[start:end+1]
        
        idx = 0
        for i in range(len(parent2)):
            if parent2[i] not in offspring:
                while offspring[idx] != -1:
                    idx += 1
                offspring[idx] = parent2[i]
        return offspring
    
    def mutate(self, route, mutation_rate=0.01):
        """Mutate a route by swapping two cities with a given probability"""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route
    
    def genetic_algorithm(self, pop_size=100, generations=500, mutation_rate=0.01, visualize_steps=False, max_paths_to_store=100):
        """
        Run the genetic algorithm to solve TSP
        
        Args:
            pop_size: Size of the population
            generations: Number of generations to run
            mutation_rate: Probability of mutation
            visualize_steps: If True, store paths for visualization
            max_paths_to_store: Maximum number of paths to store for visualization
        """
        population = self.create_population(pop_size)
        start_time = time.time()
        
        if visualize_steps:
            self.all_paths = []
            self.path_distances = []
            sample_indices = set(np.linspace(0, generations-1, min(max_paths_to_store, generations), dtype=int))
        
        for generation in range(generations):
            # Sort population by fitness
            population = sorted(population, key=lambda route: self._calculate_path_distance(list(route) + [route[0]]))
            
            # Update best solution
            current_best_route = population[0]
            current_best_distance = self._calculate_path_distance(list(current_best_route) + [current_best_route[0]])
            
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_path = tuple(list(current_best_route) + [current_best_route[0]])
            
            # Store paths for visualization
            if visualize_steps and generation in sample_indices:
                self.all_paths.append(tuple(list(current_best_route) + [current_best_route[0]]))
                self.path_distances.append(current_best_distance)
            
            # Stop if optimal solution found (unlikely)
            if current_best_distance == 0:
                break
            
            # Create new population
            new_population = population[:2]  # Elitism: keep top 2
            while len(new_population) < pop_size:
                parents = self.select_parents(population)
                offspring = self.crossover(parents[0], parents[1])
                offspring = self.mutate(offspring, mutation_rate)
                new_population.append(offspring)
            
            population = new_population
        
        end_time = time.time()
        
        print(f"Genetic Algorithm time: {end_time - start_time:.4f} seconds")
        print(f"Best distance: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance
    
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
        
        total_perms = math.factorial(self.num_cities - 1)
        
        if visualize_steps:
            best_paths = []
            best_distances = []
            sample_indices = set(np.linspace(0, total_perms-1, min(max_paths_to_store//2, total_perms), 
                                         dtype=int))
            
        for idx, path in enumerate(itertools.permutations(cities)):
            full_path = (0,) + path + (0,)
            distance = self._calculate_path_distance(full_path)
            
            if distance < best_distance:
                best_distance = distance
                best_path = full_path
                
                if visualize_steps:
                    best_paths.append(full_path)
                    best_distances.append(distance)
            elif visualize_steps and idx in sample_indices:
                best_paths.append(full_path)
                best_distances.append(distance)
        
        if visualize_steps:
            path_data = [(p, d, i) for i, (p, d) in enumerate(zip(best_paths, best_distances))]
            path_data.sort(key=lambda x: x[2])
            self.all_paths = [p for p, _, _ in path_data]
            self.path_distances = [d for _, d, _ in path_data]
        
        end_time = time.time()
        
        self.best_distance = best_distance
        self.best_path = best_path
        
        print(f"Brute Force time: {end_time - start_time:.4f} seconds")
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
        
        path.append(start_city)
        total_distance += self.distances[current_city][start_city]

        self.best_path = tuple(path)
        self.best_distance = total_distance
        
        print(f"Nearest Neighbor distance: {total_distance:.2f}")
        
        return self.best_path, self.best_distance
    
    def solve_genetic(self, pop_size=100, generations=500, mutation_rate=0.01, visualize_steps=False, max_paths_to_store=100):
        """
        Solve TSP using the genetic algorithm
        
        Args:
            pop_size: Size of the population
            generations: Number of generations to run
            mutation_rate: Probability of mutation
            visualize_steps: If True, store paths for visualization
            max_paths_to_store: Maximum number of paths to store for visualization
        """
        return self.genetic_algorithm(pop_size, generations, mutation_rate, visualize_steps, max_paths_to_store)
    
    def get_solution_info(self):
        """Return a dictionary with solution information"""
        if self.best_path is None:
            return {"error": "No solution available. Run a solve method first."}
            
        return {
            "best_path": self.best_path,
            "best_distance": self.best_distance,
            "num_cities": self.num_cities,
            "coordinates": self.coordinates.tolist()
        }
    
    def visualize(self, path=None, save_path=None):
        """Visualize the TSP solution"""
        plt.figure(figsize=(10, 8))
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='blue', s=100, label='Cities')

        for i, (x, y) in enumerate(self.coordinates):
            plt.annotate(f"{i}", (x, y), fontsize=12)

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

    def animate_solution(self, save_path=None):
        """Create an animation of the solution process"""
        if not self.all_paths:
            print("No paths available for animation. Run solve with visualize_steps=True.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='blue', s=100)
            
            for i, (x, y) in enumerate(self.coordinates):
                ax.annotate(f"{i}", (x, y), fontsize=12)
            
            path = self.all_paths[frame]
            path_coords = self.coordinates[list(path)]
            ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2)
            ax.set_title(f'TSP - Step {frame+1}/{len(self.all_paths)} - Distance: {self.path_distances[frame]:.2f}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True)
        
        anim = FuncAnimation(fig, update, frames=len(self.all_paths), interval=200, repeat=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        plt.show()