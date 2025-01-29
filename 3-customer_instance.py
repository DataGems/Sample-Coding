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
