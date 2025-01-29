import pulp
import numpy as np
import time

# [Previous VRPTWOptimizer class implementation goes here]

def create_test_instance():
    """Create a small test instance with 3 customers"""
    # Customer locations
    locations = {
        0: (0, 0),   # Depot start
        1: (2, 4),   # Customer 1
        2: (-1, 3),  # Customer 2
        3: (4, 1),   # Customer 3
        4: (0, 0)    # Depot end
    }
    
    # Calculate costs (Euclidean distances * 10)
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 10)
    
    # Time windows [earliest, latest]
    time_windows = {
        0: (0, 200),   # Depot start
        1: (20, 80),   # Customer 1
        2: (40, 100),  # Customer 2
        3: (60, 120),  # Customer 3
        4: (0, 200)    # Depot end
    }
    
    # Customer demands
    demands = {
        0: 0,   # Depot
        1: 10,  # Customer 1
        2: 15,  # Customer 2
        3: 20,  # Customer 3
        4: 0    # Depot end
    }
    
    # Vehicle capacity
    vehicle_capacity = 50
    
    # Local area neighbors
    local_area_neighbors = {
        1: [2],    # Customer 1's neighbors
        2: [1, 3], # Customer 2's neighbors
        3: [2]     # Customer 3's neighbors
    }
    
    # Capacity buckets
    capacity_buckets = {}
    for i in range(1, 4):
        demand = demands[i]
        capacity_buckets[i] = [
            (demand, demand + 15),
            (demand + 16, vehicle_capacity)
        ]
    
    # Time buckets
    time_buckets = {}
    for i in range(1, 4):
        earliest, latest = time_windows[i]
        mid = (earliest + latest) // 2
        time_buckets[i] = [
            (earliest, mid),
            (mid + 1, latest)
        ]
    
    return {
        'customers': list(range(1, 4)),
        'depot_start': 0,
        'depot_end': 4,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': vehicle_capacity,
        'local_area_neighbors': local_area_neighbors,
        'time_buckets': time_buckets,
        'capacity_buckets': capacity_buckets
    }

def main():
    instance = create_test_instance()
    optimizer = VRPTWOptimizer(**instance)
    
    print("Solving VRPTW instance...")
    solution = optimizer.solve(time_limit=300)
    
    print(f"\nSolution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if 'routes' in solution:
        print("\nRoutes found:")
        print(f"Total Cost: {solution['objective']:.2f}")
        for i, route in enumerate(solution['routes'], 1):
            print(f"Route {i}: {' -> '.join(str(node) for node in [0] + route + [4])}")

if __name__ == "__main__":
    main()