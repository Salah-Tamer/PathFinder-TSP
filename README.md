# PathFinder-TSP

A Python implementation of the Traveling Salesman Problem (TSP) solver using a brute force algorithm with improved step-by-step visualization.

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

### Running the demo

```bash
python tsp_demo.py
```

## Limitations

The brute force approach has O(n!) complexity, making it feasible only for small problem instances (typically n ≤ 10). For larger problems, consider implementing more advanced algorithms like:

- Nearest Neighbor
- Dynamic Programming
- Genetic Algorithms
- Simulated Annealing