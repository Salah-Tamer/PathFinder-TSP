# PathFinder-TSP

## Overview
The Traveling Salesman Problem (TSP) is an NP-hard problem that is computationally challenging, making it practically impossible to find an optimal solution for large instances. The TSP can be represented as a complete graph, where each vertex represents a city, and every pair of vertices is connected by a direct edge with an associated weight, typically representing the distance or cost of travel. The objective is to find the shortest possible route that visits each city exactly once and returns to the starting city, forming a Hamiltonian cycle. 

Due to its complexity, exact solutions for large TSP instances are infeasible with current computational resources. However, various approaches exist to tackle the problem, including exact algorithms like dynamic programming for smaller instances, and heuristic or approximation algorithms, such as nearest neighbor, genetic algorithms for larger instances. These methods trade optimality for providing near-optimal solutions in reasonable time.

## Solutions
We used two of approximation algorithms nearest neighbor and genetic algorithms where it tries to find near-optimal solution and we also tried the brute force solution that gurantees to find the optimal path

### 1. Brute Force

The algorithm generates all permutations of the cities, computes the cost of each route, and selects the one with the minimum cost. Due to the factorial growth in permutations, this approach becomes computationally infeasible for large inputs, especially when the number of cities exceeds 10.

The number of unique routes for `n` cities is `(n-1)! / 2`, making brute force impractical for `n > 10`. Assuming a computer evaluates 1 million routes per second, the times are:

| Number of Cities ( n ) | Number of Routes ( (n-1)! / 2 ) | Estimated Time (at 1M routes/sec) |
|----------------------------|-------------------------------------|------------------------------------|
| 10                         | 181,440                             | ~0.18 seconds                     |
| 11                         | 1,814,400                           | ~1.81 seconds                     |
| 12                         | 19,958,400                          | ~19.96 seconds                    |
| 15                         | ~1.307 × 10¹²                       | ~1,307,674 seconds (~15 days)     |
| 20                         | ~1.216 × 10¹⁵                       | ~1,216,451,004 seconds (~38,551 years) |

We also optimized the visualization by limiting the number of excessively large paths, storing some of the best discovered paths while filtering out poor random ones to improve clarity and avoid overwhelming the user.

### 2. Nearest Neighbour

A greedy heuristic used to find a quick and often reasonably good solution to the TSP. The algorithm starts at a randomly chosen city and repeatedly visits the nearest unvisited city until all cities have been visited. Finally, it returns to the starting city to complete the cycle.

Despite its simplicity, the Nearest Neighbour algorithm can produce a decent approximation in a fraction of the time required for brute-force methods. However, its greedy nature means it may not always produce the shortest possible tour. It is prone to getting stuck in local optima, especially in instances where the closest city is not part of the optimal long-term route.

### 3. Genetic Algorithm

A population-based metaheuristic inspired by the principles of natural selection and genetics. It is particularly effective for solving optimization problems like the TSP, where exhaustive search is infeasible. GA operates on a population of possible solutions, evolving them over generations to find better and better approximations of the optimal path.

Rather than working with a single candidate like Nearest Neighbour, GA maintains diversity and explores multiple regions of the solution space simultaneously, helping it escape local optima.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Salah-Tamer/PathFinder-TSP.git
cd PathFinder-TSP
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the program

```bash
python Init.py
```

## Features 

- Step-by-step visualization of the Nearest Neighbor algorithm.
- Animated movement between cities.
- Displays current distance and best path.
- Includes graphical enhancements for clarity and better user experience.
