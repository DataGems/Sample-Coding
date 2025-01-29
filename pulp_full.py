import pulp
import numpy as np
import time

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, local_area_neighbors, time_buckets, capacity_buckets):
        """
        Initialize VRPTW optimizer with problem parameters
        """
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs
        self.time_windows = time_windows
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.local_area_neighbors = local_area_neighbors
        self.time_buckets = time_buckets
        self.capacity_buckets = capacity_buckets
        
        # Initialize model
        self.model = pulp.LpProblem("VRPTW", pulp.LpMinimize)
        
        # Create all the graph structures needed
        self.E_star = self._create_valid_edges()
        self.R_u = self._generate_orderings()
        self.G_D, self.E_D = self._create_capacity_graph()
        self.G_T, self.E_T = self._create_time_graph()
        
        # Create variables and add constraints
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

    def _generate_orderings(self):
        """Generate efficient orderings for LA arcs"""
        R_u = {}
        for u in self.customers:
            R_u[u] = []
            neighbors = self.local_area_neighbors[u]
            
            # Generate simple orderings (direct connections)
            for v in neighbors:
                if self._is_time_feasible(u, v):
                    R_u[u].append({
                        'sequence': [u, v],
                        'a_wv': {(u,v): 1},
                        'a_star': {v: 1}
                    })
        return R_u

    def _create_capacity_graph(self):
        """Create capacity graph G_D and edge set E_D"""
        G_D = []
        
        # Create nodes for customers
        for u in self.customers:
            for k, (d_min, d_max) in enumerate(self.capacity_buckets[u], 1):
                G_D.append((u, k, d_min, d_max))
        
        # Add depot nodes
        G_D.append((self.depot_start, 1, self.vehicle_capacity, self.vehicle_capacity))
        G_D.append((self.depot_end, 1, 0, self.vehicle_capacity))
        
        # Create edges
        E_D = []
        for i, k_i, d_min_i, d_max_i in G_D:
            for j, k_j, d_min_j, d_max_j in G_D:
                if i != j:
                    demand_i = self.demands.get(i, 0)
                    if d_max_i - demand_i >= d_min_j:
                        E_D.append(((i,k_i), (j,k_j)))
        
        return G_D, E_D

    def _create_time_graph(self):
        """Create time graph G_T and edge set E_T"""
        G_T = []
        
        # Create nodes for customers
        for u in self.customers:
            for k, (t_min, t_max) in enumerate(self.time_buckets[u], 1):
                G_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        earliest_start, latest_start = self.time_windows[self.depot_start]
        earliest_end, latest_end = self.time_windows[self.depot_end]
        G_T.append((self.depot_start, 1, earliest_start, latest_start))
        G_T.append((self.depot_end, 1, earliest_end, latest_end))
        
        # Create edges
        E_T = []
        for i, k_i, t_min_i, t_max_i in G_T:
            for j, k_j, t_min_j, t_max_j in G_T:
                if i != j:
                    travel_time = self.costs.get((i,j), 0)
                    if t_min_i + travel_time <= t_max_j:
                        E_T.append(((i,k_i), (j,k_j)))
        
        return G_T, E_T

    def _create_variables(self):
        """Create all decision variables for the model"""
        # Route variables (x_{ij})
        self.x = pulp.LpVariable.dicts("x", 
                                     ((i,j) for i,j in self.E_star),
                                     cat='Binary')
        
        # Ordering selection variables (y_r)
        self.y = pulp.LpVariable.dicts("y",
                                     ((u,r) for u in self.customers 
                                      for r in range(len(self.R_u[u]))),
                                     lowBound=0)
        
        # Capacity graph variables (z^D_{ij})
        self.z_D = pulp.LpVariable.dicts("z_D",
                                        self.E_D,
                                        lowBound=0)
        
        # Time graph variables (z^T_{ij})
        self.z_T = pulp.LpVariable.dicts("z_T",
                                        self.E_T,
                                        lowBound=0)
        
        # Resource variables (δ_u and τ_u)
        self.delta = pulp.LpVariable.dicts("delta",
                                         self.customers + [self.depot_start, self.depot_end],
                                         lowBound=0,
                                         upBound=self.vehicle_capacity)
        
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)

    def _add_constraints(self):
        """Add all constraints from Equation 6"""
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
                    self.model += (self.delta[j] - self.demands[j] >= 
                                 self.delta[u] - (self.vehicle_capacity + self.demands[j]) * 
                                 (1 - self.x[v,j]))
        
        # Time window propagation constraints (6e)
        for u in self.customers:
            for v,j in self.E_star:
                if v != u:
                    self.model += (self.tau[j] - self.costs[v,j] >= 
                                 self.tau[u] - (self.time_windows[u][1] + self.costs[v,j]) * 
                                 (1 - self.x[v,j]))
        
        # Minimum number of vehicles constraint (6f)
        total_demand = sum(self.demands[u] for u in self.customers)
        min_vehicles = int(np.ceil(total_demand / self.vehicle_capacity))
        self.model += (pulp.lpSum(self.x[self.depot_start,j] 
                                for i,j in self.E_star if i == self.depot_start) >= min_vehicles)
        
        # LA-arc movement consistency constraints
        # Select exactly one ordering for each customer (6g)
        for u in self.customers:
            self.model += pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))) == 1
        
        # Consistency between x and y variables (6h-6i)
        for u in self.customers:
            for w in self.local_area_neighbors[u] + [u]:
                # Constraint 6h
                for v in self.local_area_neighbors[u]:
                    if (w,v) in self.E_star:
                        self.model += (self.x[w,v] >= 
                                     pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u])) 
                                              if self.R_u[u][r]['a_wv'].get((w,v), 0) == 1))
                
                # Constraint 6i
                if w in self.local_area_neighbors[u] + [u]:
                    self.model += (pulp.lpSum(self.x[w,v] for i,v in self.E_star 
                                            if i == w and v not in self.local_area_neighbors[u] + [u]) >= 
                                 pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u])) 
                                          if self.R_u[u][r]['a_star'].get(w, 0) == 1))
        
        # Flow graph capacity constraints
        # Flow conservation for capacity (6j)
        for i, k, d_min, d_max in self.G_D:
            if i not in [self.depot_start, self.depot_end]:
                self.model += (pulp.lpSum(self.z_D[e] for e in self.E_D if e[0] == (i,k)) ==
                             pulp.lpSum(self.z_D[e] for e in self.E_D if e[1] == (i,k)))
        
        # Consistency between x and z_D variables (6k)
        for i, j in self.E_star:
            self.model += (self.x[i,j] == 
                         pulp.lpSum(self.z_D[e] for e in self.E_D 
                                  if e[0][0] == i and e[1][0] == j))
        
        # Flow graph time constraints
        # Flow conservation for time (6l)
        for i, k, t_min, t_max in self.G_T:
            if i not in [self.depot_start, self.depot_end]:
                self.model += (pulp.lpSum(self.z_T[e] for e in self.E_T if e[0] == (i,k)) ==
                             pulp.lpSum(self.z_T[e] for e in self.E_T if e[1] == (i,k)))
        
        # Consistency between x and z_T variables (6m)
        for i, j in self.E_star:
            self.model += (self.x[i,j] == 
                         pulp.lpSum(self.z_T[e] for e in self.E_T 
                                  if e[0][0] == i and e[1][0] == j))

    def solve(self, time_limit=None):
        """Solve the VRPTW instance"""
        if time_limit:
            self.model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        else:
            self.model.solve(pulp.PULP_CBC_CMD())
            
        # Prepare solution information
        solution = {
            'status': pulp.LpStatus[self.model.status],
            'computation_time': time.time() - self.model.solutionTime,
            'objective': pulp.value(self.model.objective)
        }
        
        # Extract routes if solution is found
        if self.model.status == pulp.LpStatusOptimal:
            solution['routes'] = self._extract_routes()
            
        return solution

    def _extract_routes(self):
        """Extract routes from solution"""
        routes = []
        current_route = []
        
        # Start from depot
        current = self.depot_start
        
        # Keep track of visited customers
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
        
        # Add final route if not empty
        if current_route:
            routes.append(current_route)
        
        return routes

def create_test_instance():
    """Create a test instance with 7 customers"""
    locations = {
        0: (0, 0),    # Depot start
        1: (2, 4),    # Customer 1
        2: (-1, 3),   # Customer 2
        3: (4, 1),    # Customer 3
        4: (-2, -3),  # Customer 4
        5: (1, -2),   # Customer 5
        6: (3, -1),   # Customer 6
        7: (-3, 2),   # Customer 7
        8: (0, 0)     # Depot end
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
        0: 0,   # Depot start
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
    
    # Create capacity buckets
    capacity_buckets = {}
    for i in range(1, 8):
        demand = demands[i]
        capacity_buckets[i] = [
            (demand, demand + 20),
            (demand + 21, demand + 40),
            (demand + 41, vehicle_capacity)
        ]
    
    # Create time buckets
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
    # Create test instance
    print("Creating test instance...")
    instance = create_test_instance()
    
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
    
    print("\nSolving VRPTW instance with 7 customers...")
    solution = optimizer.solve(time_limit=600)  # 10 minute time limit
    
    # Print results
    print("\nResults:")
    print(f"Solution Status: {solution['status']}")
    print(f"Computation Time: {solution['computation_time']:.2f} seconds")
    
    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        if 'routes' in solution:
            print("\nRoutes:")
            for i, route in enumerate(solution['routes'], 1):
                print(f"Route {i}: {' -> '.join(['0'] + [str(c) for c in route] + ['8'])}")
    else:
        print("\nNo optimal solution found within time limit")

if __name__ == "__main__":
    main()