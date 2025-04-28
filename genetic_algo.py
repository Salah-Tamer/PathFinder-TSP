import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsp_solver import TSPSolver

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
        
        # Calculate the distance matrix
        self.distances = self._calculate_distances()
        self.best_distance = float('inf')
        self.best_path = None
        self.best_path_over_time = []  # To store best paths for animation

    def _calculate_distances(self):
        n = self.num_cities
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def calculate_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i]][route[i + 1]]
        total_distance += self.distances[route[-1]][route[0]]  # Return to the starting city
        return total_distance

    def create_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            route = random.sample(range(self.num_cities), self.num_cities)
            population.append(route)
        return population
    
    def select_parents(self, population):
        fitness_scores = [1 / self.calculate_distance(route) for route in population]
        total_fitness = sum(fitness_scores)
        probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
        parents = random.choices(population, probabilities, k=2)
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
    
    def mutate(self, route, mutation_rate=0.01):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route
    
    def genetic_algorithm(self, pop_size, generations, mutation_rate=0.01):
        population = self.create_population(pop_size)
        best_path_over_time = []  # To store best paths for animation
        
        for generation in range(generations):
            population = sorted(population, key=lambda route: self.calculate_distance(route))
            print(f"Generation {generation}: Best Distance = {self.calculate_distance(population[0])}")
            
            best_path_over_time.append(population[0])  # Store the best path for this generation
            
            if self.calculate_distance(population[0]) == 0:
                break

            new_population = population[:2]  # Elitism: Keep the best 2 solutions
            while len(new_population) < pop_size:
                parents = self.select_parents(population)
                offspring = self.crossover(parents[0], parents[1])
                offspring = self.mutate(offspring, mutation_rate)
                new_population.append(offspring)
            
            population = new_population
        
        best_route = population[0]
        return best_route, self.calculate_distance(best_route), best_path_over_time

    def solve(self, pop_size=100, generations=500, mutation_rate=0.01):
        best_route, best_distance, best_path_over_time = self.genetic_algorithm(pop_size, generations, mutation_rate)
        self.best_path = best_route
        self.best_distance = best_distance
        self.best_path_over_time = best_path_over_time  # Save for animation
        print(f"Best route: {best_route}")
        print(f"Total distance: {best_distance}")
        return best_route, best_distance

    def visualize(self, path=None, save_path=None):
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



        