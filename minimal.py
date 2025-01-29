def create_simple_test_instance():
    """Create a simple test instance with 3 customers that is guaranteed to be feasible"""
    # Customer locations in a small area
    locations = {
        0: (0, 0),    # Depot start
        1: (1, 1),    # Customer 1 (nearby)
        2: (-1, 1),   # Customer 2 (nearby)
        3: (0, 2),    # Customer 3 (nearby)
        4: (0, 0)     # Depot end
    }
    
    # Calculate costs (Euclidean distances * 10)
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 10)
    
    # Time windows with plenty of overlap
    time_windows = {
        0: (0, 1000),    # Depot start (very wide window)
        1: (0, 100),     # Customer 1
        2: (0, 100),     # Customer 2
        3: (0, 100),     # Customer 3
        4: (0, 1000)     # Depot end (very wide window)
    }
    
    # Small demands relative to capacity
    demands = {
        0: 0,    # Depot
        1: 10,   # Customer 1
        2: 10,   # Customer 2
        3: 10,   # Customer 3
        4: 0     # Depot end
    }
    
    # Large vehicle capacity
    vehicle_capacity = 100
    
    # Simple local area neighbors (everyone is neighbors)
    local_area_neighbors = {
        1: [2, 3],    # Customer 1's neighbors
        2: [1, 3],    # Customer 2's neighbors
        3: [1, 2]     # Customer 3's neighbors
    }
    
    # Simple capacity buckets (just two per customer)
    capacity_buckets = {}
    for i in range(1, 4):
        demand = demands[i]
        capacity_buckets[i] = [
            (demand, vehicle_capacity // 2),
            (vehicle_capacity // 2 + 1, vehicle_capacity)
        ]
    
    # Simple time buckets (just two per customer)
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
    # Create test instance
    print("Creating simple test instance...")
    instance = create_simple_test_instance()
    
    # Create and solve the model
    print("\nInitializing optimizer...")
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        local_area_neighbors=instance['local_area_neighbors'],
        time_buckets=instance['time_buckets'],
        capacity_buckets=instance['capacity_buckets']
    )
    
    print("\nSolving VRPTW instance with 3 customers...")
    solution = optimizer.solve(time_limit=300)  # 5 minute time limit
    
    # Print results
    print("\nResults:")
    print(f"Solution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        if 'routes' in solution:
            print("\nRoutes:")
            for i, route in enumerate(solution['routes'], 1):
                print(f"Route {i}: {' -> '.join(['0'] + [str(c) for c in route] + ['4'])}")
    else:
        print("\nNo optimal solution found within time limit")

if __name__ == "__main__":
    main()