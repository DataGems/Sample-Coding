import pulp
import numpy as np
import time

# [Previous VRPTWOptimizer class implementation goes here]

def create_large_test_instance():
    """Create a test instance with 7 customers"""
    locations = {
        0: (0, 0),     # Depot start
        1: (2, 4),     # Customer 1
        2: (-1, 3),    # Customer 2
        3: (4, 1),     # Customer 3
        4: (-2, -3),   # Customer 4
        5: (1, -2),    # Customer 5
        6: (3, -1),    # Customer 6
        7: (-3, 2),    # Customer 7
        8: (0, 0)      # Depot end
    }
    
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 10)
    
    time_windows = {
        0: (0, 280),    # Depot start
        1: (20, 80),    # Customer 1
        2: (40, 100),   # Customer 2
        3: (60, 120),   # Customer 3
        4: (80, 140),   # Customer 4
        5: (100, 160),  # Customer 5
        6: (120, 180),  # Customer 6
        7: (140, 200),  # Customer 7
        8: (0, 280)     # Depot end
    }
    
    demands = {
        0: 0,   # Depot
        1: 10,  # Customer 1
        2: 15,  # Customer 2
        3: 20,  # Customer 3
        4: 12,  # Customer 4
        5: 18,  # Customer 5
        6: 14,  # Customer 6
        7: 16,  # Customer 7
        8: 0    # Depot end
    }
    
    vehicle_capacity = 60
    
    local_area_neighbors = {
        1: [2, 3],      # Customer 1's neighbors
        2: [1, 7],      # Customer 2's neighbors
        3: [1, 6],      # Customer 3's neighbors
        4: [5, 7],      # Customer 4's neighbors
        5: [4, 6],      # Customer 5's neighbors
        6: [3, 5],      # Customer 6's neighbors
        7: [2, 4]       # Customer 7's neighbors
    }
    
    capacity_buckets = {}
    for i in range(1, 8):
        demand = demands[i]
        capacity_buckets[i] = [
            (demand, demand + 20),
            (demand + 21, demand + 40),
            (demand + 41, vehicle_capacity)
        ]
    
    time_buckets = {}
    for i in range(1, 8):
        earliest, latest = time_windows[i]
        interval = (latest - earliest) // 3
        time_buckets[i] = [
            (earliest, earliest + interval),
            (earliest + interval + 1, earliest + 2 * interval),
            (earliest + 2 * interval + 1, latest)
        ]
    
    return {
        'customers': list(range(1, 8)),
        'depot_start': 0,
        'depot_end': 8,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': vehicle_capacity,
        'local_area_neighbors': local_area_neighbors,
        'time_buckets': time_buckets,
        'capacity_buckets': capacity_buckets
    }

def main():
    instance = create_large_test_instance()
    optimizer = VRPTWOptimizer(**instance)
    
    print("Solving VRPTW instance with 7 customers...")
    solution = optimizer.solve(time_limit=600)  # Increased time limit to 10 minutes
    
    print(f"\nSolution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if 'routes' in solution:
        print("\nRoutes found:")
        print(f"Total Cost: {solution['objective']:.2f}")
        for i, route in enumerate(solution['routes'], 1):
            print(f"Route {i}: {' -> '.join(str(node) for node in [0] + route + [8])}")

if __name__ == "__main__":
    main()