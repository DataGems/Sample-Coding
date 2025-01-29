import pulp
import numpy as np
import time

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, vehicle_capacity):
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs
        self.time_windows = time_windows
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        
        # Initialize model
        self.model = pulp.LpProblem("VRPTW", pulp.LpMinimize)
        
        # Create valid edges
        self.E_star = [(i,j) for i in [self.depot_start] + self.customers 
                      for j in self.customers + [self.depot_end] if i != j]
        
        print(f"Number of edges: {len(self.E_star)}")
        print("Edges:", self.E_star)
        
        self._create_variables()
        self._add_constraints()

    def _create_variables(self):
        # Route variables x_{ij}
        self.x = pulp.LpVariable.dicts("x", 
                                     self.E_star,
                                     cat='Binary')
        
        # Time variables
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)

    def _add_constraints(self):
        """Add only essential routing constraints"""
        # Objective function: minimize total cost
        self.model += pulp.lpSum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star)
        
        # Each customer must be visited exactly once
        for u in self.customers:
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == 1
            print(f"Customer {u} inbound edges:", [(i,j) for i,j in self.E_star if j == u])
        
        # Each customer must be left exactly once
        for u in self.customers:
            self.model += pulp.lpSum(self.x[u,j] for i,j in self.E_star if i == u) == 1
            print(f"Customer {u} outbound edges:", [(i,j) for i,j in self.E_star if i == u])
        
        # Time window constraints
        for (i,j) in self.E_star:
            if j != self.depot_end:
                M = max(tw[1] for tw in self.time_windows.values())
                self.model += self.tau[j] >= self.tau[i] + self.costs[i,j] - M * (1 - self.x[i,j])
        
        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model += self.tau[i] >= self.time_windows[i][0]
            self.model += self.tau[i] <= self.time_windows[i][1]

    def solve(self, time_limit=None):
        """Solve the VRPTW instance"""
        print("\nSolving model...")
        if time_limit:
            status = self.model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        else:
            status = self.model.solve(pulp.PULP_CBC_CMD())
            
        print(f"Status: {pulp.LpStatus[status]}")
        
        solution = {
            'status': pulp.LpStatus[status],
            'computation_time': time.time(),
            'objective': pulp.value(self.model.objective) if status == pulp.LpStatusOptimal else None
        }
        
        return solution

def create_tiny_instance():
    """Create a tiny test instance with 2 customers"""
    costs = {
        (0, 1): 1, (0, 2): 1, (0, 3): 100,  # From depot start
        (1, 2): 1, (1, 3): 1,                # From customer 1
        (2, 1): 1, (2, 3): 1,                # From customer 2
        (3, 0): 0                            # To depot end
    }
    
    time_windows = {
        0: (0, 100),    # Depot start
        1: (0, 100),    # Customer 1
        2: (0, 100),    # Customer 2
        3: (0, 100)     # Depot end
    }
    
    demands = {
        0: 0,    # Depot start
        1: 1,    # Customer 1
        2: 1,    # Customer 2
        3: 0     # Depot end
    }
    
    return {
        'customers': [1, 2],
        'depot_start': 0,
        'depot_end': 3,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 10
    }

def main():
    print("Creating tiny test instance...")
    instance = create_tiny_instance()
    
    print("\nInitializing optimizer...")
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity']
    )
    
    solution = optimizer.solve(time_limit=300)
    
    print("\nFinal Results:")
    print(f"Solution Status: {solution['status']}")
    
    if solution['status'] == 'Optimal':
        print(f"Optimal Solution Cost: {solution['objective']:.2f}")

if __name__ == "__main__":
    main()