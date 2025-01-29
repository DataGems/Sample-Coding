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
        
        # Time variables τ_i
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)
        
        # Capacity variables δ_i
        self.delta = pulp.LpVariable.dicts("delta",
                                         self.customers + [self.depot_start, self.depot_end],
                                         lowBound=0,
                                         upBound=self.vehicle_capacity)

    def _add_constraints(self):
        """Add routing and capacity constraints"""
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
            
        # Capacity constraints
        # Initialize depot capacity
        self.model += self.delta[self.depot_start] == self.vehicle_capacity
        self.model += self.delta[self.depot_end] >= 0
        
        # Capacity propagation
        for (i,j) in self.E_star:
            if j != self.depot_end:
                M = self.vehicle_capacity + max(self.demands.values())
                self.model += self.delta[j] <= self.delta[i] - self.demands.get(j, 0) + M * (1 - self.x[i,j])
                self.model += self.delta[j] >= self.demands.get(j, 0)
        
        # Minimum number of vehicles (based on total demand)
        total_demand = sum(self.demands[u] for u in self.customers)
        min_vehicles = int(np.ceil(total_demand / self.vehicle_capacity))
        self.model += pulp.lpSum(self.x[self.depot_start,j] 
                               for i,j in self.E_star if i == self.depot_start) >= min_vehicles

    def solve(self, time_limit=None):
        """Solve the VRPTW instance"""
        print("\nSolving model...")
        start_time = time.time()
        
        if time_limit:
            status = self.model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        else:
            status = self.model.solve(pulp.PULP_CBC_CMD())
        
        solve_time = time.time() - start_time
        print(f"Status: {pulp.LpStatus[status]}")
        
        solution = {
            'status': pulp.LpStatus[status],
            'computation_time': solve_time,
            'objective': pulp.value(self.model.objective) if status == pulp.LpStatusOptimal else None
        }
        
        if status == pulp.LpStatusOptimal:
            # Print variable values for debugging
            print("\nDecision Variables:")
            for (i,j) in self.E_star:
                val = pulp.value(self.x[i,j])
                if val > 0.5:
                    print(f"x_{i}_{j} = {val}")
            
            solution['routes'] = self._extract_routes()
            
            print("\nSolution Details:")
            for i in self.customers + [self.depot_start, self.depot_end]:
                if i in self.delta:
                    print(f"Node {i} - Capacity: {pulp.value(self.delta[i]):.2f}, Time: {pulp.value(self.tau[i]):.2f}")
        
        return solution

    def _extract_routes(self):
        """Extract routes from solution using active edges"""
        print("\nDebug: Route Extraction")
        routes = []
        active_edges = [(i,j) for (i,j) in self.E_star 
                       if pulp.value(self.x[i,j]) > 0.5]
        print("Active edges:", active_edges)
        
        if not active_edges:
            return routes
            
        # Build routes from active edges
        current_route = []
        current = self.depot_start
        visited_edges = set()
        
        while len(visited_edges) < len(active_edges):
            # Find the next edge from current node
            for edge in active_edges:
                if edge[0] == current and edge not in visited_edges:
                    visited_edges.add(edge)
                    if edge[1] == self.depot_end:
                        if current_route:
                            routes.append(current_route)
                            current_route = []
                            current = self.depot_start
                    else:
                        current_route.append(edge[1])
                        current = edge[1]
                    break
            else:
                if current_route:
                    routes.append(current_route)
                break
                
        # Add any remaining route
        if current_route:
            routes.append(current_route)
            
        return routes

def create_complex_instance():
    """Create a complex test instance with 5 customers, with relaxed constraints"""
    # Locations in a smaller 5x5 grid to reduce travel times
    locations = {
        0: (0, 0),     # Depot start
        1: (1, 4),     # Customer 1 - north
        2: (4, 1),     # Customer 2 - east
        3: (1, 2),     # Customer 3 - west
        4: (2, 2),     # Customer 4 - central
        5: (3, 3),     # Customer 5 - northeast
        6: (0, 0)      # Depot end
    }
    
    # Calculate costs based on Euclidean distance * 5 (reduced scaling)
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 5)
    
    # Time windows - more relaxed with overlapping windows
    time_windows = {
        0: (0, 200),     # Depot start - long horizon
        1: (10, 80),     # Customer 1 - early window
        2: (20, 100),    # Customer 2 - mid window
        3: (15, 90),     # Customer 3 - early-mid window
        4: (30, 120),    # Customer 4 - mid-late window
        5: (40, 140),    # Customer 5 - late window
        6: (0, 200)      # Depot end - long horizon
    }
    
    # Demands - reduced to total 25 units
    demands = {
        0: 0,     # Depot
        1: 6,     # Customer 1
        2: 7,     # Customer 2
        3: 3,     # Customer 3
        4: 4,     # Customer 4
        5: 5,     # Customer 5
        6: 0      # Depot end
    }
    
    # Vehicle capacity of 15 units (will still need at least 2 vehicles)
    vehicle_capacity = 15
    
    return {
        'customers': [1, 2, 3, 4, 5],
        'depot_start': 0,
        'depot_end': 6,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': vehicle_capacity,
        'locations': locations  # Added for visualization
    }

def main():
    print("Creating modified complex test instance with 5 customers...")
    instance = create_complex_instance()
    
    print("\nProblem characteristics:")
    print(f"Number of customers: {len(instance['customers'])}")
    print(f"Vehicle capacity: {instance['vehicle_capacity']}")
    print(f"Total demand: {sum(instance['demands'][i] for i in instance['customers'])}")
    print("\nTime Windows and Demands:")
    for i in sorted(instance['customers']):
        print(f"Customer {i}: Window {instance['time_windows'][i]}, Demand: {instance['demands'][i]}")
    
    print("\nTravel times (sample):")
    sample_paths = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    for i,j in sample_paths:
        print(f"From {i} to {j}: {instance['costs'][i,j]/5:.1f} time units")  # Divide by 5 to get actual time
    
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
    
    print("\nSolving model...")
    solution = optimizer.solve(time_limit=300)
    
    print("\nFinal Results:")
    print(f"Solution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if solution['status'] == 'Optimal':
        print(f"Optimal Solution Cost: {solution['objective']:.2f}")
        print("\nRoutes:")
        for i, route in enumerate(solution['routes'], 1):
            print(f"\nRoute {i}: {' -> '.join(['0'] + [str(c) for c in route] + ['6'])}")
            # Calculate route statistics
            route_demand = sum(instance['demands'][c] for c in route)
            print(f"  Total demand: {route_demand}")
            print(f"  Time window feasibility:")
            current_time = 0
            current_location = 0
            for c in route:
                travel_time = instance['costs'][current_location, c] / 5  # Convert back to actual time
                current_time = max(current_time + travel_time, instance['time_windows'][c][0])
                print(f"    Customer {c}: Arrive at {current_time:.1f} (Window: {instance['time_windows'][c]})")
                current_location = c

if __name__ == "__main__":
    main()