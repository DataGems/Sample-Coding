import pulp
import numpy as np
import time
from collections import defaultdict

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, K=2, capacity_granularity=3):
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs
        self.time_windows = time_windows
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.K = K
        
        # Initialize model
        self.model = pulp.LpProblem("VRPTW", pulp.LpMinimize)
        
        # Create valid edges
        self.E_star = [(i,j) for i in [self.depot_start] + self.customers 
                      for j in self.customers + [self.depot_end] if i != j]
        
        # Generate LA neighborhoods
        self.la_neighbors = self._generate_la_neighbors()
        
        # Generate orderings
        self.R_u = self._generate_orderings()
        
        # Create capacity discretization
        self.capacity_buckets = self._create_capacity_buckets(capacity_granularity)
        self.nodes_D, self.edges_D = self._create_capacity_graph()
        
        self._create_variables()
        self._add_constraints()
        
        print(f"\nModel initialized with:")
        print(f"Number of customers: {len(customers)}")
        print(f"Number of edges: {len(self.E_star)}")
        print(f"Number of capacity nodes: {len(self.nodes_D)}")
        print(f"Number of capacity edges: {len(self.edges_D)}")

    def _create_capacity_buckets(self, granularity):
        """Create capacity buckets for each customer"""
        buckets = {}
        for u in self.customers:
            demand = self.demands[u]
            remaining_capacity = self.vehicle_capacity - demand
            bucket_size = remaining_capacity / granularity
            
            customer_buckets = []
            for i in range(granularity):
                lower = demand + i * bucket_size
                upper = demand + (i + 1) * bucket_size if i < granularity - 1 else self.vehicle_capacity
                customer_buckets.append((lower, upper))
                
            buckets[u] = customer_buckets
        return buckets

    def _create_capacity_graph(self):
        """Create directed graph GD for capacity flow"""
        nodes_D = []  # (u, k, d⁻, d⁺)
        edges_D = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and bucket
        for u in self.customers:
            for k, (d_min, d_max) in enumerate(self.capacity_buckets[u]):
                nodes_D.append((u, k, d_min, d_max))
        
        # Add depot nodes
        nodes_D.append((self.depot_start, 0, self.vehicle_capacity, self.vehicle_capacity))
        nodes_D.append((self.depot_end, 0, 0, self.vehicle_capacity))
        
        # Create edges
        for i, k_i, d_min_i, d_max_i in nodes_D:
            for j, k_j, d_min_j, d_max_j in nodes_D:
                if i != j:
                    # Check capacity feasibility
                    if d_max_i - self.demands.get(i,0) >= d_min_j:
                        edges_D.append(((i,k_i), (j,k_j)))
        
        return nodes_D, edges_D

    def _generate_la_neighbors(self):
        """Generate K closest neighbors for each customer"""
        la_neighbors = {}
        for u in self.customers:
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            distances.sort(key=lambda x: x[1])
            neighbors = []
            for j, _ in distances:
                if len(neighbors) >= self.K:
                    break
                if self._is_reachable(u, j):
                    neighbors.append(j)
            la_neighbors[u] = neighbors
        return la_neighbors
    
    def _is_reachable(self, i, j):
        """Check if j is reachable from i considering time windows and capacity"""
        earliest_i, latest_i = self.time_windows[i]
        earliest_j, latest_j = self.time_windows[j]
        travel_time = self.costs[i,j] / 5
        
        if earliest_i + travel_time > latest_j:
            return False
            
        if self.demands[i] + self.demands[j] > self.vehicle_capacity:
            return False
            
        return True

    def _generate_orderings(self):
        """Generate efficient orderings for each customer"""
        R_u = defaultdict(list)
        
        for u in self.customers:
            # Base ordering
            R_u[u].append({
                'sequence': [u],
                'a_wv': {},
                'a_star': {u: 1}
            })
            
            # Add single neighbor orderings
            for v in self.la_neighbors[u]:
                if self._is_reachable(u, v):
                    R_u[u].append({
                        'sequence': [u, v],
                        'a_wv': {(u,v): 1},
                        'a_star': {v: 1}
                    })
            
            # Add two-neighbor orderings
            for v1 in self.la_neighbors[u]:
                for v2 in self.la_neighbors[u]:
                    if v1 != v2 and self._is_sequence_feasible([u, v1, v2]):
                        R_u[u].append({
                            'sequence': [u, v1, v2],
                            'a_wv': {(u,v1): 1, (v1,v2): 1},
                            'a_star': {v2: 1}
                        })
        
        return R_u

    def _is_sequence_feasible(self, sequence):
        """Check if a sequence of customers is feasible"""
        total_demand = sum(self.demands[i] for i in sequence)
        if total_demand > self.vehicle_capacity:
            return False
            
        current_time = 0
        for i in range(len(sequence)-1):
            current = sequence[i]
            next_customer = sequence[i+1]
            current_time = max(current_time + self.costs[current,next_customer]/5,
                             self.time_windows[next_customer][0])
            
            if current_time > self.time_windows[next_customer][1]:
                return False
        
        return True

    def _create_variables(self):
        """Create all decision variables"""
        # Route variables x_{ij}
        self.x = pulp.LpVariable.dicts("x", 
                                     self.E_star,
                                     cat='Binary')
        
        # Time variables τ_i
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)
        
        # LA-arc variables y_r
        self.y = {}
        for u in self.customers:
            for r in range(len(self.R_u[u])):
                self.y[u,r] = pulp.LpVariable(f"y_{u}_{r}", cat='Binary')
        
        # Capacity flow variables z_D
        self.z_D = pulp.LpVariable.dicts("z_D",
                                        self.edges_D,
                                        lowBound=0)

    def _add_constraints(self):
        """Add all constraints"""
        # Original routing constraints
        self._add_routing_constraints()
        
        # LA-arc constraints
        self._add_la_arc_constraints()
        
        # Capacity flow constraints
        self._add_capacity_flow_constraints()

    def _add_routing_constraints(self):
        """Add basic routing constraints"""
        # Objective function
        self.model += pulp.lpSum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star)
        
        # Visit each customer once
        for u in self.customers:
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == 1
            self.model += pulp.lpSum(self.x[u,j] for i,j in self.E_star if i == u) == 1
        
        # Time window constraints
        for (i,j) in self.E_star:
            if j != self.depot_end:
                M = max(tw[1] for tw in self.time_windows.values())
                self.model += self.tau[j] >= self.tau[i] + self.costs[i,j]/5 - M * (1 - self.x[i,j])
        
        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model += self.tau[i] >= self.time_windows[i][0]
            self.model += self.tau[i] <= self.time_windows[i][1]

    def _add_la_arc_constraints(self):
        """Add LA-arc movement consistency constraints"""
        # Select one ordering per customer
        for u in self.customers:
            self.model += pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))) == 1
        
        # Consistency between x and y
        for u in self.customers:
            for w in [u] + self.la_neighbors[u]:
                for v in self.la_neighbors[u]:
                    if (w,v) in self.E_star:
                        self.model += self.x[w,v] >= pulp.lpSum(
                            self.y[u,r] for r in range(len(self.R_u[u]))
                            if self.R_u[u][r]['a_wv'].get((w,v), 0) == 1
                        )
        
        # Consistency for final customer
        for u in self.customers:
            for w in [u] + self.la_neighbors[u]:
                outside_neighbors = [j for j in self.customers + [self.depot_end] 
                                   if j not in self.la_neighbors[u] + [u]]
                if outside_neighbors:
                    self.model += pulp.lpSum(self.x[w,j] for j in outside_neighbors 
                                           if (w,j) in self.E_star) >= \
                                pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))
                                         if self.R_u[u][r]['a_star'].get(w, 0) == 1)

    def _add_capacity_flow_constraints(self):
        """Add capacity flow constraints"""
        # Flow conservation (4a)
        for i, k, d_min, d_max in self.nodes_D:
            if i not in [self.depot_start, self.depot_end]:
                self.model += (
                    pulp.lpSum(self.z_D[e] for e in self.edges_D if e[0] == (i,k)) ==
                    pulp.lpSum(self.z_D[e] for e in self.edges_D if e[1] == (i,k))
                )
        
        # Consistency with route variables (4b)
        for u,v in self.E_star:
            self.model += (
                self.x[u,v] == pulp.lpSum(
                    self.z_D[e] for e in self.edges_D 
                    if e[0][0] == u and e[1][0] == v
                )
            )

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
            print("\nDecision Variables:")
            for (i,j) in self.E_star:
                val = pulp.value(self.x[i,j])
                if val is not None and val > 0.5:
                    print(f"x_{i}_{j} = {val}")
            
            solution['routes'] = self._extract_routes()
            
            print("\nSolution Details:")
            for i in self.customers + [self.depot_start, self.depot_end]:
                tau_val = pulp.value(self.tau[i])
                if tau_val is not None:
                    print(f"Node {i} - Time: {tau_val:.2f}")
        
        return solution

    def _extract_routes(self):
        """Extract routes from solution"""
        active_edges = [(i,j) for (i,j) in self.E_star 
                       if pulp.value(self.x[i,j]) is not None 
                       and pulp.value(self.x[i,j]) > 0.5]
        print("\nActive edges:", active_edges)
        
        routes = []
        depot_starts = [(i,j) for (i,j) in active_edges if i == self.depot_start]
        
        for start_edge in depot_starts:
            route = []
            current = start_edge[1]
            route.append(current)
            
            while current != self.depot_end:
                next_edges = [(i,j) for (i,j) in active_edges if i == current]
                if not next_edges:
                    break
                current = next_edges[0][1]
                if current != self.depot_end:
                    route.append(current)
            
            routes.append(route)
        
        return routes

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