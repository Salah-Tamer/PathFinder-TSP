from tsp_solver import TSPSolver
import numpy as np

def main():
    print("TSP Solver Demo with Improved Visualization")
    print("------------------------------------------")
    
    # Ask user which example to run
    print("\nSelect an example to run:")
    print("1. Small example (5 cities)")
    print("2. Predefined cities in a pattern (6 cities)")
    print("3. Slightly larger problem (7 cities)")
    print("4. Quit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        # Example 1: Small example
        print("\nRunning Example 1: 5 random cities")
        
        # Create solver
        solver = TSPSolver(num_cities=5)
        
        # Show initial city layout
        solver.visualize()
        
        # Solve and store paths for visualization
        solver.solve_brute_force(visualize_steps=True, max_paths_to_store=20)
        
        # Show final solution
        solver.visualize()
        
        # Show animation
        solver.visualize_step_by_step(interval=800)
    
    elif choice == '2':
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
        
        # Show initial city layout
        solver.visualize()
        
        # Solve and store paths for visualization
        solver.solve_brute_force(visualize_steps=True, max_paths_to_store=30)
        
        # Show final solution
        solver.visualize()
        
        # Show animation
        solver.visualize_step_by_step(interval=600)
    
    elif choice == '3':
        # Example 3: Slightly larger problem
        print("\nRunning Example 3: 7 cities")
        
        # Create solver
        solver = TSPSolver(num_cities=7)
        
        # Show initial city layout
        solver.visualize()
        
        # Solve and store paths for visualization
        print("\nSolving TSP with brute force approach...")
        solver.solve_brute_force(visualize_steps=True, max_paths_to_store=40)
        
        # Show final solution
        solver.visualize()
        
        # Show animation
        solver.visualize_step_by_step(interval=400)
    
    elif choice == '4':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main() 