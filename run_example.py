from tsp_solver import TSPSolver
import numpy as np
import time

def run_example(example_number=1):
    print("TSP Solver Demo with Improved Step-by-Step Visualization")
    print("------------------------------------------------------")
    
    if example_number == 1:
        # Example 1: Small example
        print("\nRunning Example 1: 5 random cities with step-by-step visualization")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create solver
        solver = TSPSolver(num_cities=5)
        print("Cities generated at random positions.")
        
        # Show initial city layout
        solver.visualize()
        
        # Solve and store paths for visualization
        print("\nSolving TSP with brute force approach...")
        solver.solve_brute_force(visualize_steps=True, max_paths_to_store=20)
        
        # Show final solution
        print("\nShowing final solution:")
        solver.visualize()
        
        # Show step-by-step animation
        print("\nNow showing step-by-step visualization...")
        solver.visualize_step_by_step(interval=1000)  # 1 second between frames
    
    elif example_number == 2:
        # Example 2: Predefined cities
        print("\nRunning Example 2: Predefined cities in a pattern")
        
        # Define cities in a pattern
        coordinates = [
            (20, 20),   # City 0
            (40, 30),   # City 1
            (60, 20),   # City 2
            (40, 10),   # City 3
            (30, 40),   # City 4
            (50, 40),   # City 5
        ]
        
        solver = TSPSolver(coordinates=coordinates)
        print("Using predefined city locations.")
        
        # Show initial city layout
        solver.visualize()
        
        # Solve and store paths for visualization
        print("\nSolving TSP with brute force approach...")
        solver.solve_brute_force(visualize_steps=True, max_paths_to_store=30)
        
        # Show final solution
        print("\nShowing final solution:")
        solver.visualize()
        
        # Show step-by-step animation
        print("\nNow showing step-by-step visualization...")
        solver.visualize_step_by_step(interval=800)  # 0.8 seconds between frames
    
    elif example_number == 3:
        # Example 3: Slightly larger problem
        print("\nRunning Example 3: 7 cities (larger problem)")
        print("Note: This will generate a lot of permutations!")
        
        # Set random seed for reproducibility
        np.random.seed(123)
        
        # Create solver
        solver = TSPSolver(num_cities=7)
        print("Cities generated at random positions.")
        
        # Show initial city layout
        solver.visualize()
        
        # Solve and store paths for visualization
        print("\nSolving TSP with brute force approach...")
        print("(This might take a moment as there are 720 possible paths)")
        solver.solve_brute_force(visualize_steps=True, max_paths_to_store=40)
        
        # Show final solution
        print("\nShowing final solution:")
        solver.visualize()
        
        # Show step-by-step animation
        print("\nNow showing step-by-step visualization...")
        solver.visualize_step_by_step(interval=500)  # 0.5 seconds between frames

if __name__ == "__main__":
    # Run example 1 (change to 2 or 3 to run different examples)
    run_example(1) 