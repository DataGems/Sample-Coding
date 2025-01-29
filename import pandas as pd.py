import pandas as pd
import numpy as np

def load_solomon_instance(filename, num_customers=None):
    """
    Load Solomon VRPTW instance from CSV file
    
    Args:
        filename: Path to CSV file
        num_customers: Number of customers to include (None for all)
    
    Returns:
        Dictionary with problem instance data
    """
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Convert customer numbers to 0-based indexing
    df['CUST_NUM'] = df['CUST_NUM'].astype(int)
    
    # Limit number of customers if specified
    if num_customers is not None:
        df = df.iloc[:num_customers+1]  # +1 to include depot
    
    # Extract coordinates
    coords = {row['CUST_NUM']: (row['XCOORD.'], row['YCOORD.']) 
             for _, row in df.iterrows()}
    
    # Calculate distances (rounded down to 1 decimal as per paper)
    costs = {}
    for i in coords:
        for j in coords:
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = np.floor(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 10) / 10
                costs[i,j] = dist * 5  # Convert to costs as per paper
    
    # Create customer list (excluding depot)
    customers = sorted(list(set(df['CUST_NUM']) - {0}))
    
    # Extract time windows and demands
    time_windows = {row['CUST_NUM']: (row['READY_TIME'], row['DUE_DATE'])
                   for _, row in df.iterrows()}
    
    demands = {row['CUST_NUM']: row['DEMAND']
              for _, row in df.iterrows()}
    
    # Create instance dictionary
    instance = {
        'customers': customers,
        'depot_start': 0,
        'depot_end': len(customers) + 1,  # Create virtual end depot
        'costs': costs,
        'time_windows': time_windows | {len(customers) + 1: time_windows[0]},  # End depot has same time window as start
        'demands': demands | {len(customers) + 1: 0},  # Zero demand for end depot
        'vehicle_capacity': 200  # Standard capacity for Solomon instances
    }
    
    return instance

def run_solomon_instance(filename, num_customers, 
                        K=3, max_iterations=5, time_limit=300):
    """
    Load and solve a Solomon instance
    
    Args:
        filename: Path to Solomon instance file
        num_customers: Number of customers to include
        K: Initial neighborhood size
        max_iterations: Maximum iterations for LA-Discretization
        time_limit: Time limit in seconds
    """
    print(f"Loading Solomon instance with {num_customers} customers...")
    instance = load_solomon_instance(filename, num_customers)
    
    print("\nProblem characteristics:")
    print(f"Number of customers: {len(instance['customers'])}")
    print(f"Vehicle capacity: {instance['vehicle_capacity']}")
    print(f"Total demand: {sum(instance['demands'][i] for i in instance['customers'])}")
    
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        K=K,
        max_iterations=max_iterations
    )
    
    print("\nSolving...")
    solution = optimizer.solve_with_parsimony(time_limit=time_limit)
    
    print("\nSolution Results:")
    print(f"Status: {solution['status']}")
    print(f"Objective: {solution['objective']}")
    print(f"Computation time: {solution['computation_time']:.2f} seconds")
    print("\nRoutes:")
    for i, route in enumerate(solution['routes'], 1):
        print(f"Route {i}: {' -> '.join(str(x) for x in [0] + route + [optimizer.depot_end])}")
    
    # Validate solution
    is_valid = optimizer.validate_solution(solution)
    print(f"\nSolution is {'valid' if is_valid else 'invalid'}")
    
    return optimizer, solution