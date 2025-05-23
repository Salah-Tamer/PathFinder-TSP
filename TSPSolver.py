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
        if coordinates is not None:
            self.coordinates = np.array(coordinates)
            self.num_cities = len(coordinates)
        elif num_cities is not None:
            self.num_cities = num_cities
            self.coordinates = np.random.rand(num_cities, 2) * 100
        else:
            raise ValueError("Either coordinates or num_cities must be provided")
            
        self.distances = self._calculate_distances()
        
        self.best_distance = float('inf')
        self.best_path = None
        
        self.all_paths = []
        self.path_distances = []
        self.best_path_over_time = []
        
    def _calculate_distances(self):
        n = self.num_cities
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def _calculate_path_distance(self, path):
        return sum(self.distances[path[i], path[i+1]] for i in range(len(path)-1))
    
    def calculate_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i]][route[i + 1]]
        total_distance += self.distances[route[-1]][route[0]]
        return total_distance
    
    def create_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            route = random.sample(range(self.num_cities), self.num_cities)
            population.append(route)
        return population
    
    def select_parents(self, population, method='tournament'):
        if method == 'tournament':
            tournament_size = max(2, min(8, int(len(population) * 0.1)))
            parents = []
            for _ in range(2):
                contestants = random.sample(population, tournament_size)
                if random.random() < 0.2:
                    contestants = sorted(contestants, key=lambda x: self.calculate_distance(x))
                    parents.append(contestants[1])
                else:
                    best = min(contestants, key=lambda x: self.calculate_distance(x))
                    parents.append(best)
        return parents
    
    def crossover(self, parent1, parent2):
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
    
    def mutate(self, route, mutation_rate=0.1):
        new_route = route[:]
        if random.random() < mutation_rate:
            # Use different mutation operators
            mutation_type = random.choice(['swap', 'inversion', 'scramble'])
            
            if mutation_type == 'swap':
                # Swap two random cities
                idx1, idx2 = random.sample(range(len(new_route)), 2)
                new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
            elif mutation_type == 'inversion':
                # Invert a random segment
                start, end = sorted(random.sample(range(len(new_route)), 2))
                new_route[start:end+1] = new_route[start:end+1][::-1]
            else:  # scramble
                # Scramble a random segment
                start, end = sorted(random.sample(range(len(new_route)), 2))
                segment = new_route[start:end+1]
                random.shuffle(segment)
                new_route[start:end+1] = segment
                
        return new_route
    
    def genetic_algorithm(self, pop_size=100, generations=500, mutation_rate=0.05, 
                         visualize_steps=False, convergence_threshold=0.001):
        population = self.create_population(pop_size)
        start_time = time.time()
        
        if visualize_steps:
            self.all_paths = []
            self.path_distances = []
            self.all_discovered_paths = set()
            self.path_generations = []
            self.generation_best_paths = []
            self.generation_best_distances = []
            self.optimal_paths = set()
        
        # Track convergence
        no_improvement_count = 0
        last_best_distance = float('inf')
        
        for generation in range(generations):
            population = sorted(population, key=lambda route: self.calculate_distance(route))
            
            current_best_route = population[0]
            current_best_distance = self.calculate_distance(current_best_route)
            
            # Check for convergence
            if abs(current_best_distance - last_best_distance) < convergence_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            last_best_distance = current_best_distance
            
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_path = tuple(list(current_best_route) + [current_best_route[0]])
                self.optimal_paths = set()
                self.optimal_paths.add(tuple(list(current_best_route) + [current_best_route[0]]))
            elif current_best_distance == self.best_distance:
                self.optimal_paths.add(tuple(list(current_best_route) + [current_best_route[0]]))
            
            if visualize_steps:
                for route in population:
                    closed_route = tuple(list(route) + [route[0]])
                    if closed_route not in self.all_discovered_paths:
                        self.all_discovered_paths.add(closed_route)
                        self.all_paths.append(closed_route)
                        self.path_distances.append(self.calculate_distance(route))
                        self.path_generations.append(generation)
                self.generation_best_paths.append(tuple(list(current_best_route) + [current_best_route[0]]))
                self.generation_best_distances.append(current_best_distance)
            
            # Early stopping if no improvement for too long
            if no_improvement_count >= 50:
                print(f"Early stopping at generation {generation} due to convergence")
                break
            
            # Keep top 5% of population for elitism
            elite_size = max(1, int(pop_size * 0.05))
            new_population = population[:elite_size]
            
            # Add some random solutions to maintain diversity
            num_random = max(1, int(pop_size * 0.05))
            new_population.extend(self.create_population(num_random))
            
            # Adaptive mutation rate based on diversity
            current_diversity = len(set(tuple(route) for route in population)) / pop_size
            adaptive_mutation_rate = mutation_rate * (1 - current_diversity)
            
            while len(new_population) < pop_size:
                parents = self.select_parents(population, method='tournament')
                offspring = self.crossover(parents[0], parents[1])
                offspring = self.mutate(offspring, adaptive_mutation_rate)
                new_population.append(offspring)
            
            population = new_population
        
        end_time = time.time()
        
        print(f"\nFound {len(self.optimal_paths)} different paths with the optimal distance of {self.best_distance:.2f}")
        print(f"Algorithm ran for {generations} generations in {end_time - start_time:.2f} seconds")
        if len(self.optimal_paths) > 1:
            print("This demonstrates that there are multiple paths that achieve the same optimal distance!")
        
        return self.best_path, self.best_distance
    
    def solve_brute_force(self, visualize_steps=False, max_paths_to_store=100):
        cities = list(range(1, self.num_cities))
        best_distance = float('inf')
        best_path = None
        start_time = time.time()
        
        self.all_paths = []
        self.path_distances = []
        
        total_perms = math.factorial(self.num_cities - 1)
        
        if visualize_steps:
            best_paths = []
            best_distances = []
            count = min(max_paths_to_store//2, total_perms)
            if count > 0:
                sample_indices = set(np.linspace(0, total_perms-1, count, dtype=int))
            else:
                sample_indices = set()
            
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
    
    # Nearest Neighbor Method
    def solve_nearest_neighbor(self, start_city=0, visualize_steps=False):
        unvisited = set(range(self.num_cities))
        path = [start_city]
        current_city = start_city
        unvisited.remove(current_city)
        total_distance = 0
        start_time = time.time()
        
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
                # For visualization, create a temporary path that includes return to start
                temp_path = path.copy()
                temp_distance = total_distance
                
                # Add return to start city for visualization
                temp_path.append(start_city)
                temp_distance += self.distances[current_city][start_city]
                
                self.all_paths.append(tuple(temp_path))
                self.path_distances.append(temp_distance)
        
        # Add the return to start city to complete the circuit
        total_distance += self.distances[current_city][start_city]
        path.append(start_city)
                
        end_time = time.time()
        self.best_path = tuple(path)
        self.best_distance = total_distance
        
        print(f"Nearest Neighbor time: {end_time - start_time:.4f} seconds")
        print(f"Nearest Neighbor distance: {total_distance:.2f}")
        
        return self.best_path, self.best_distance
    
    # Unified solve method
    def solve(self, method='genetic', **kwargs):
        if method == 'genetic':
            return self.solve_genetic(**kwargs)
        elif method == 'brute_force':
            return self.solve_brute_force(**kwargs)
        elif method == 'nearest_neighbor':
            return self.solve_nearest_neighbor(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def solve_genetic(self, pop_size=100, generations=500, mutation_rate=0.01, 
                     visualize_steps=False):
        """Wrapper for genetic algorithm"""
        return self.genetic_algorithm(pop_size, generations, mutation_rate, 
                                    visualize_steps)
    
    def get_solution_info(self):
        if self.best_path is None:
            return {"error": "No solution available. Run a solve method first."}
            
        return {
            "best_path": self.best_path,
            "best_distance": self.best_distance,
            "num_cities": self.num_cities,
            "coordinates": self.coordinates.tolist()
        }