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

def create_large_test_instance():
    """Create test instance with 25 customers"""
    # Create a 10x10 grid, with depot in center
    locations = {
        0: (5, 5),     # Depot start
        1: (2, 8),     # Customers 1-25
        2: (3, 9),
        3: (4, 7),
        4: (6, 8),
        5: (7, 9),
        6: (8, 7),
        7: (9, 8),
        8: (1, 6),
        9: (3, 5),
        10: (7, 6),
        11: (8, 4),
        12: (9, 5),
        13: (1, 3),
        14: (2, 2),
        15: (4, 3),
        16: (6, 2),
        17: (8, 1),
        18: (2, 4),
        19: (3, 1),
        20: (5, 3),
        21: (7, 2),
        22: (1, 7),
        23: (4, 8),
        24: (6, 4),
        25: (8, 5),
        26: (5, 5)     # Depot end
    }
    
    # Calculate costs based on Euclidean distance * 5
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 5)
    
    # Time windows - morning (0-100) and afternoon (100-200) shifts
    # Spread customers across the day to make feasible routes more likely
    time_windows = {
        0: (0, 300),     # Depot start - long horizon
        26: (0, 300)     # Depot end - long horizon
    }
    
    # Create overlapping time windows
    for i in range(1, 26):
        if i <= 12:  # Morning shift customers
            earliest = max(20 * (i-1), 0)
            latest = earliest + 80
        else:  # Afternoon shift customers
            earliest = max(20 * (i-12), 100)
            latest = earliest + 80
        time_windows[i] = (earliest, latest)
    
    # Demands - mix of small (3-5), medium (6-8), and large (9-12) orders
    demands = {0: 0, 26: 0}  # Zero demand for depots
    for i in range(1, 26):
        if i % 3 == 0:
            # Large order
            demands[i] = np.random.randint(9, 13)
        elif i % 3 == 1:
            # Medium order
            demands[i] = np.random.randint(6, 9)
        else:
            # Small order
            demands[i] = np.random.randint(3, 6)
    
    # Set vehicle capacity to handle about 3-4 customers per route
    vehicle_capacity = 30
    
    return {
        'customers': list(range(1, 26)),
        'depot_start': 0,
        'depot_end': 26,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': vehicle_capacity
    }

def main():
    print("Creating instance with 25 customers...")
    instance = create_large_test_instance()
    
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
        K=3,  # Increased number of neighbors
        capacity_granularity=3
    )
    
    print("\nSolving model...")
    solution = optimizer.solve(time_limit=600)  # Increased time limit to 10 minutes
    
    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        print("\nRoutes:")
        total_cost = 0
        for idx, route in enumerate(solution['routes'], 1):
            route_demand = sum(instance['demands'][c] for c in route)
            route_cost = sum(instance['costs'][i,j] for i, j in zip([0] + route, route + [26]))
            total_cost += route_cost
            
            print(f"\nRoute {idx}: {' -> '.join(['0'] + [str(c) for c in route] + ['26'])}")
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

if __name__ == "__main__":
    main()