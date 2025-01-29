def create_test_instance():
    """Create test instance with 5 customers"""
    locations = {
        0: (0, 0),     # Depot start
        1: (2, 4),     # Customer 1
        2: (-1, 3),    # Customer 2
        3: (4, 1),     # Customer 3
        4: (-2, -3),   # Customer 4
        5: (1, -2),    # Customer 5
        6: (0, 0)      # Depot end
    }
    
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 5)
    
	time_windows = {
        0: (0, 100),    # Depot
        1: (10, 40),    # Customer 1
        2: (30, 70),    # Customer 2
        3: (50, 80),    # Customer 3
        4: (20, 60),    # Customer 4
        5: (40, 90),    # Customer 5
        6: (0, 100)     # Depot end
    }
    
    demands = {
        0: 0,     # Depot
        1: 5,     # Customer 1
        2: 7,     # Customer 2
        3: 6,     # Customer 3
        4: 4,     # Customer 4
        5: 8,     # Customer 5
        6: 0      # Depot end
    }
    
    return {
        'customers': list(range(1, 6)),
        'depot_start': 0,
        'depot_end': 6,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 15
    }