import pulp
import numpy as np
import time

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, vehicle_capacity):
        """
        Initialize VRPTW optimizer with basic parameters
        
        Parameters:
        customers: List of customer indices
        depot_start: Starting depot index (α)
        depot_end: Ending depot index (ᾱ)
        costs: Dictionary of travel costs/times between locations {(i,j): cost}
        time_windows: Dictionary of time windows {i: (earliest, latest)}
        demands: Dictionary of customer demands {i: demand}
        vehicle_capacity: Maximum vehicle capacity (d0)
        """
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs
        self.time_windows = time_windows
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        
        # Initialize model
        self.model = pulp.LpProblem("VRPTW", pulp.LpMinimize)
        
        # Create valid edges E*
        self.E_star = self._create_valid_edges()
        
        # Create variables and constraints
        self._create_variables()
        self._add_constraints()

    def _create_valid_edges(self):
        """Create set of valid edges considering time windows"""
        E_star = []
        for i in self.customers + [self.depot_start]:
            for j in self.customers + [self.depot_end]:
                if i != j:
                    if self._is_time_feasible(i, j):
                        E_star.append((i, j))
        return E_star

    def _is_time_feasible(self, i, j):
        """Check if edge (i,j) is feasible with respect to time windows"""
        earliest_i, latest_i = self.time_windows[i]
        earliest_j, latest_j = self.time_windows[j]
        travel_time = self.costs[(i,j)]
        return earliest_i + travel_time <= latest_j

    def _create_variables(self):
        """Create decision variables"""
        # Route variables x_{ij} - binary vehicle flow variables
        self.x = pulp.LpVariable.dicts("x", 
                                     self.E_star,
                                     cat='Binary')
        
        # Resource variables
        # δ_i - remaining capacity at location i
        self.delta = pulp.LpVariable.dicts("delta",
                                         self.customers + [self.depot_start, self.depot_end],
                                         lowBound=0,
                                         upBound=self.vehicle_capacity)
        
        # τ_i - time of service at location i
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)

    def _add_constraints(self):
        """Add constraints from original compact formulation (6b-6f)"""
        # Objective function (6a)
        self.model += pulp.lpSum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star)
        
        # Each customer must be visited exactly once (6b)
        for u in self.customers:
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == 1
        
        # Each customer must be left exactly once (6c)
        for u in self.customers:
            self.model += pulp.lpSum(self.x[u,j] for i,j in self.E_star if i == u) == 1
        
        # Capacity propagation constraints (6d)
        for u in self.customers:
            for v,j in self.E_star:
                if v != u:
                    M = self.vehicle_capacity + self.demands[j]  # Big-M value
                    self.model += (self.delta[j] - self.demands[j] >= 
                                 self.delta[u] - M * (1 - self.x[v,j]))
        
        # Time window propagation constraints (6e)
        for u in self.customers:
            for v,j in self.E_star:
                if v != u:
                    M = self.time_windows[u][1] + self.costs[v,j]  # Big-M value
                    self.model += (self.tau[j] - self.costs[v,j] >= 
                                 self.tau[u] - M * (1 - self.x[v,j]))
        
        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model += self.tau[i] >= self.time_windows[i][0]
            self.model += self.tau[i] <= self.time_windows[i][1]
        
        # Minimum number of vehicles constraint (6f)
        total_demand = sum(self.demands[u] for u in self.customers)
        min_vehicles = int(np.ceil(total_demand / self.vehicle_capacity))
        self.model += (pulp.lpSum(self.x[self.depot_start,j] 
                                for i,j in self.E_star if i == self.depot_start) >= min_vehicles)

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
            # Find next customer in route
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
            
            # Break if no feasible next customer found
            else:
                if current_route:
                    routes.append(current_route)
                break
        
        return routes

def create_test_instance():
    """Create a very simple test instance with 3 customers"""
    # Customer locations
    locations = {
        0: (0, 0),    # Depot start
        1: (1, 0),    # Customer 1
        2: (0, 1),    # Customer 2
        3: (1, 1),    # Customer 3
        4: (0, 0)     # Depot end
    }
    
    # Calculate costs/times (Manhattan distance)
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = abs(x2-x1) + abs(y2-y1)
    
    # Very relaxed time windows
    time_windows = {
        0: (0, 1000),   # Depot start
        1: (0, 100),    # Customer 1
        2: (0, 100),    # Customer 2
        3: (0, 100),    # Customer 3
        4: (0, 1000)    # Depot end
    }
    
    # Small demands
    demands = {
        0: 0,   # Depot
        1: 5,   # Customer 1
        2: 5,   # Customer 2
        3: 5,   # Customer 3
        4: 0    # Depot end
    }
    
    return {
        'customers': [1, 2, 3],
        'depot_start': 0,
        'depot_end': 4,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 20
    }

def main():
    # Create and solve test instance
    print("Creating test instance...")
    instance = create_test_instance()
    
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
    
    print("\nSolving VRPTW instance...")
    solution = optimizer.solve(time_limit=300)
    
    print("\nResults:")
    print(f"Solution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        print("\nRoutes:")
        for i, route in enumerate(solution['routes'], 1):
            print(f"Route {i}: {' -> '.join(['0'] + [str(c) for c in route] + ['4'])}")
            
if __name__ == "__main__":
    main()