def create_small_test_instance():
    """Create a small test instance with 3 customers"""
    instance = {
        'customers': [1, 2, 3],
        'depot_start': 0,
        'depot_end': 4,
        'costs': {
            (0,1): 10, (0,2): 15, (0,3): 20, 
            (1,2): 12, (1,3): 18, (1,4): 10,
            (2,1): 12, (2,3): 14, (2,4): 15,
            (3,1): 18, (3,2): 14, (3,4): 20,
            (1,0): 10, (2,0): 15, (3,0): 20
        },
        'time_windows': {
            0: (0, 100),    # Depot start
            1: (10, 40),    # Customer 1
            2: (30, 70),    # Customer 2
            3: (50, 80),    # Customer 3
            4: (0, 100)     # Depot end
        },
        'demands': {
            0: 0,    # Depot start
            1: 5,    # Customer 1
            2: 7,    # Customer 2
            3: 6,    # Customer 3
            4: 0     # Depot end
        },
        'vehicle_capacity': 15
    }
    return instance

def run_small_test():
    """Run test on small instance"""
    print("Creating small test instance...")
    instance = create_small_test_instance()
    
    print("\nProblem characteristics:")
    print(f"Number of customers: {len(instance['customers'])}")
    print(f"Vehicle capacity: {instance['vehicle_capacity']}")
    print(f"Total demand: {sum(instance['demands'][i] for i in instance['customers'])}")
    
    print("\nCustomer Details:")
    for i in sorted(instance['customers']):
        print(f"Customer {i}: Window {instance['time_windows'][i]}, Demand: {instance['demands'][i]}")
    
    print("\nInitializing optimizer...")
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        K=2,
        capacity_granularity=3,
        time_granularity=3
    )
    
    print("\nSolving model...")
    solution = optimizer.solve(time_limit=300)
    
    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        print("\nRoutes:")
        total_cost = 0
        for idx, route in enumerate(solution['routes'], 1):
            route_demand = sum(instance['demands'][c] for c in route)
            route_cost = sum(instance['costs'][i,j] for i, j in zip([0] + route, route + [4]))
            total_cost += route_cost
            
            print(f"\nRoute {idx}: {' -> '.join(['0'] + [str(c) for c in route] + ['4'])}")
            print(f"  Total demand: {route_demand}")
            print(f"  Schedule:")
            current_time = 0
            current_loc = 0
            for stop in route:
                travel_time = instance['costs'][current_loc, stop] / 5
                arrival_time = max(current_time + travel_time, instance['time_windows'][stop][0])
                print(f"    Customer {stop}: Arrive at {arrival_time:.1f} "
                      f"(Window: {instance['time_windows'][stop]}, "
                      f"Demand: {instance['demands'][stop]})")
                current_time = arrival_time
                current_loc = stop
        
        print(f"\nTotal Cost: {total_cost}")
    
    return optimizer, solution

if __name__ == "__main__":
    optimizer, solution = run_small_test()
    