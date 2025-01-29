import pulp
import numpy as np
import time

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, vehicle_capacity):
        """Simplified initialization with only essential parameters"""
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs
        self.time_windows = time_windows
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        
        # Initialize model
        self.model = pulp.LpProblem("VRPTW", pulp.LpMinimize)
        
        # Create edges
        self.E_star = [(i,j) for i in [self.depot_start] + self.customers 
                      for j in self.customers + [self.depot_end] if i != j]
        
        self._create_variables()
        self._add_constraints()

    def _create_variables(self):
        """Create basic decision variables"""
        # Route variables (x_{ij})
        self.x = pulp.LpVariable.dicts("x", 
                                     self.E_star,
                                     cat='Binary')
        
        # Time variables
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)

    def _add_constraints(self):
        """Add basic VRPTW constraints"""
        # Objective function: minimize total cost
        self.model += pulp.lpSum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star)
        
        # Each customer must be visited exactly once
        for u in self.customers:
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == 1
            self.model += pulp.lpSum(self.x[u,j] for i,j in self.E_star if i == u) == 1
        
        # Time window constraints
        M = max(tw[1] for tw in self.time_windows.values())  # Big M value
        for (i,j) in self.E_star:
            if j != self.depot_end:
                self.model += self.tau[j] >= self.tau[i] + self.costs[i,j] - M * (1 - self.x[i,j])
        
        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model += self.tau[i] >= self.time_windows[i][0]
            self.model += self.tau[i] <= self.time_windows[i][1]

    def solve(self, time_limit=None):
        """Solve the VRPTW instance"""
        if time_limit:
            self.model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        else:
            self.model.solve(pulp.PULP_CBC_CMD())
            
        solution = {
            'status': pulp.LpStatus[self.model.status],
            'computation_time': time.time() - self.model.solutionTime,
            'objective': pulp.value(self.model.objective) if self.model.status == pulp.LpStatusOptimal else None
        }
        
        if self.model.status == pulp.LpStatusOptimal:
            solution['routes'] = self._extract_routes()
            
        return solution

    def _extract_routes(self):
        """Extract routes from solution"""
        routes = []
        current_route = []
        current = self.depot_start
        visited = set()
        
        while len(visited) < len(self.customers):
            for j in self.customers + [self.depot_end]:
                if (current, j) in self.x and pulp.value(self.x[current,j]) > 0.5:
                    if j == self.depot_end:
                        if current_route:
                            routes.append(current_route)
                        current_route = []
                        current = self.depot_start
                    else:
                        current_route.append(j)
                        visited.add(j)
                        current = j
                    break
        
        if current_route:
            routes.append(current_route)
        
        return routes

def create_minimal_test_instance():
    """Create a very simple test instance with 2 customers"""
    locations = {
        0: (0, 0),   # Depot start
        1: (1, 0),   # Customer 1
        2: (0, 1),   # Customer 2
        3: (0, 0)    # Depot end
    }
    
    # Calculate costs (Manhattan distance * 10 for simplicity)
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = (abs(x2-x1) + abs(y2-y1)) * 10
    
    # Very relaxed time windows
    time_windows = {
        0: (0, 100),   # Depot start
        1: (0, 100),   # Customer 1
        2: (0, 100),   # Customer 2
        3: (0, 100)    # Depot end
    }
    
    # Small demands
    demands = {
        0: 0,   # Depot start
        1: 5,   # Customer 1
        2: 5,   # Customer 2
        3: 0    # Depot end
    }
    
    return {
        'customers': [1, 2],
        'depot_start': 0,
        'depot_end': 3,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 20
    }

def main():
    print("Creating minimal test instance...")
    instance = create_minimal_test_instance()
    
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
    
    print("\nSolving VRPTW instance with 2 customers...")
    solution = optimizer.solve(time_limit=300)
    
    print("\nResults:")
    print(f"Solution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        if 'routes' in solution:
            print("\nRoutes:")
            for i, route in enumerate(solution['routes'], 1):
                print(f"Route {i}: {' -> '.join(['0'] + [str(c) for c in route] + ['3'])}")
    else:
        print("\nNo optimal solution found")

if __name__ == "__main__":
    main()