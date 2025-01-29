import pulp
import numpy as np
import time
from collections import defaultdict

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, K=2, capacity_granularity=3, time_granularity=3):
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
        
        # Create time discretization
        self.time_buckets = self._create_time_buckets(time_granularity)
        self.nodes_T, self.edges_T = self._create_time_graph()
        
        # Validate time discretization
        self._validate_time_discretization()
        
        self._create_variables()
        self._add_constraints()
        
        print(f"\nModel initialized with:")
        print(f"Number of customers: {len(customers)}")
        print(f"Number of edges: {len(self.E_star)}")
        print(f"Number of capacity nodes: {len(self.nodes_D)}")
        print(f"Number of capacity edges: {len(self.edges_D)}")
        print(f"Number of time nodes: {len(self.nodes_T)}")
        print(f"Number of time edges: {len(self.edges_T)}")

    def _create_capacity_buckets(self, granularity):
        """Create capacity buckets tracking remaining vehicle capacity"""
        buckets = {}
        
        # For each customer, create buckets based on remaining capacity
        for u in self.customers:
            demand = self.demands[u]
            if demand > self.vehicle_capacity:
                raise ValueError(f"Customer {u} demand {demand} exceeds vehicle capacity {self.vehicle_capacity}")
                
            # Create buckets for remaining capacity after customer's demand
            remaining_capacity = self.vehicle_capacity - demand
            bucket_size = remaining_capacity / granularity if granularity > 1 else remaining_capacity
            
            customer_buckets = []
            for i in range(granularity):
                # Buckets track remaining capacity after serving this customer
                lower = remaining_capacity - (granularity - i) * bucket_size
                upper = remaining_capacity - (granularity - i - 1) * bucket_size if i < granularity - 1 else remaining_capacity
                customer_buckets.append((max(0, lower), upper))
                
            buckets[u] = customer_buckets
        
        # Add depot buckets
        buckets[self.depot_start] = [(0, self.vehicle_capacity)]  # Full capacity available
        buckets[self.depot_end] = [(0, self.vehicle_capacity)]    # Any remaining capacity ok
        
        return buckets

    def _create_capacity_graph(self):
        """Create directed graph GD for capacity flow with cumulative demand tracking"""
        nodes_D = []  # (u, k, d⁻, d⁺)
        edges_D = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and bucket
        for u in self.customers + [self.depot_start, self.depot_end]:
            for k, (d_min, d_max) in enumerate(self.capacity_buckets[u]):
                nodes_D.append((u, k, d_min, d_max))
        
        def get_remaining_capacity(current_load, node):
            """Calculate remaining capacity after visiting a node"""
            i, k, d_min, d_max = node
            demand = self.demands.get(i, 0)
            new_load = current_load + demand
            return self.vehicle_capacity - new_load
        
        # Create edges with cumulative capacity tracking
        for i, k_i, d_min_i, d_max_i in nodes_D:
            # Track the load when reaching this node
            current_load = sum(self.demands.get(x, 0) for x in [i] if x != self.depot_start)
            
            for j, k_j, d_min_j, d_max_j in nodes_D:
                if i != j:  # No self-loops
                    # Calculate new load if we add customer j
                    next_load = current_load + self.demands.get(j, 0)
                    
                    # Check if edge is feasible
                    if i == self.depot_start:
                        # From depot - just check single customer
                        if self.demands.get(j, 0) <= self.vehicle_capacity:
                            edges_D.append(((i,k_i), (j,k_j)))
                    
                    elif j == self.depot_end:
                        # To depot - check current load is feasible
                        if current_load <= self.vehicle_capacity:
                            edges_D.append(((i,k_i), (j,k_j)))
                    
                    else:
                        # Between customers - check cumulative load
                        if (next_load <= self.vehicle_capacity and  # Total load feasible
                            d_min_j <= self.vehicle_capacity - next_load <= d_max_j):  # Capacity buckets align
                            edges_D.append(((i,k_i), (j,k_j)))
        
        print("\nDEBUG - Capacity Graph Details:")
        print(f"Nodes created: {len(nodes_D)}")
        for n in nodes_D:
            print(f"Node: {n}")
            
        print(f"\nEdges created: {len(edges_D)}")
        edge_details = []
        for ((i,ki), (j,kj)) in edges_D:
            if i in self.demands or j in self.demands:
                load_i = self.demands.get(i, 0)
                load_j = self.demands.get(j, 0)
                cum_load = sum(self.demands.get(x, 0) for x in [i, j] if x not in [self.depot_start, self.depot_end])
                edge_details.append(f"Edge ({i},{ki})->({j},{kj}): Load {load_i}+{load_j}={cum_load}")
        print("Edges with loads:")
        for detail in sorted(edge_details):
            print(detail)
        
        return nodes_D, edges_D

    def _create_time_buckets(self, granularity):
        """Create time buckets for each customer"""
        buckets = {}
        
        # Create buckets for each customer
        for u in self.customers:
            earliest_time, latest_time = self.time_windows[u]
            time_span = latest_time - earliest_time
            bucket_size = time_span / granularity
            
            customer_buckets = []
            for i in range(granularity):
                lower = earliest_time + i * bucket_size
                upper = (earliest_time + (i + 1) * bucket_size 
                        if i < granularity - 1 else latest_time)
                customer_buckets.append((lower, upper))
            
            buckets[u] = customer_buckets
        
        # Add single bucket for depot (start and end)
        depot_earliest, depot_latest = self.time_windows[self.depot_start]
        buckets[self.depot_start] = [(depot_earliest, depot_latest)]
        buckets[self.depot_end] = [(depot_earliest, depot_latest)]
        
        return buckets

    def _create_time_graph(self):
        """Create directed graph GT for time flow"""
        nodes_T = []  # (u, k, t⁻, t⁺)
        edges_T = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and their time buckets
        for u in self.customers:
            for k, (t_min, t_max) in enumerate(self.time_buckets[u]):
                nodes_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        depot_start_bucket = self.time_buckets[self.depot_start][0]
        depot_end_bucket = self.time_buckets[self.depot_end][0]
        nodes_T.append((self.depot_start, 0, depot_start_bucket[0], depot_start_bucket[1]))
        nodes_T.append((self.depot_end, 0, depot_end_bucket[0], depot_end_bucket[1]))
        
        # Create edges between nodes
        for i, k_i, t_min_i, t_max_i in nodes_T:
            for j, k_j, t_min_j, t_max_j in nodes_T:
                if i != j:
                    travel_time = self.costs[i,j] / 5
                    
                    earliest_arrival = t_min_i + travel_time
                    latest_arrival = t_max_i + travel_time
                    
                    if (earliest_arrival <= t_max_j and 
                        latest_arrival >= t_min_j and
                        earliest_arrival <= self.time_windows[j][1] and
                        latest_arrival >= self.time_windows[j][0]):
                        edges_T.append(((i,k_i), (j,k_j)))
        
        return nodes_T, edges_T

    def _validate_time_discretization(self):
        """Validate time discretization setup and data structures"""
        # Check time buckets
        for u in self.customers + [self.depot_start, self.depot_end]:
            buckets = self.time_buckets[u]
            tw_start, tw_end = self.time_windows[u]
            
            # Check bucket coverage
            if abs(buckets[0][0] - tw_start) > 1e-6:
                raise ValueError(f"First bucket for customer {u} doesn't start at time window start")
            if abs(buckets[-1][1] - tw_end) > 1e-6:
                raise ValueError(f"Last bucket for customer {u} doesn't end at time window end")
            
            # Check bucket continuity and ordering
            for i in range(len(buckets) - 1):
                if abs(buckets[i][1] - buckets[i+1][0]) > 1e-6:
                    raise ValueError(f"Gap between buckets {i} and {i+1} for customer {u}")
                if buckets[i][0] >= buckets[i][1]:
                    raise ValueError(f"Invalid bucket bounds for customer {u}, bucket {i}")
        
        # Check time graph structure
        node_dict = {(u,k): (t_min, t_max) for u,k,t_min,t_max in self.nodes_T}
        
        for (u1,k1), (u2,k2) in self.edges_T:
            # Verify nodes exist
            if (u1,k1) not in node_dict or (u2,k2) not in node_dict:
                raise ValueError(f"Edge references non-existent node: ({u1},{k1}) -> ({u2},{k2})")
            
            # Verify time feasibility
            t_min1, t_max1 = node_dict[(u1,k1)]
            t_min2, t_max2 = node_dict[(u2,k2)]
            travel_time = self.costs[u1,u2] / 5
            
            earliest_arrival = t_min1 + travel_time
            latest_arrival = t_max1 + travel_time
            
            if latest_arrival < t_min2 or earliest_arrival > t_max2:
                raise ValueError(f"Infeasible time edge: ({u1},{k1}) -> ({u2},{k2})")

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
        
        # Time flow variables z_T
        self.z_T = pulp.LpVariable.dicts("z_T",
                                        self.edges_T,
                                        lowBound=0)

    def _add_constraints(self):
        """Add all constraints"""
        # Original routing constraints
        self._add_routing_constraints()
        
        # LA-arc constraints with parsimony
        self._add_la_arc_constraints_with_parsimony()
        
        # Capacity flow constraints
        self._add_capacity_flow_constraints()
        
        # Time flow constraints
        self._add_time_flow_constraints()

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

    def _add_la_arc_constraints_with_parsimony(self):
        """Add LA-arc movement consistency constraints with parsimony penalties"""
        # Small positive value for parsimony penalties
        rho = 0.01  # ϱ in the paper
        
        # For each customer u, track customers by distance
        for u in self.customers:
            # Get distances to all other customers
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            distances.sort(key=lambda x: x[1])
            
            # Create k-indexed neighborhoods (N^k_u)
            N_k_u = {}  # k-indexed neighborhoods
            N_k_plus_u = {}  # N^k_+u from paper (includes u)
            for k in range(1, len(distances) + 1):
                N_k_u[k] = [j for j, _ in distances[:k]]
                N_k_plus_u[k] = [u] + N_k_u[k]
            
            # Select one ordering per customer (equation 6g)
            self.model += (
                pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))) == 1,
                f"one_ordering_{u}"
            )
            
            # Add k-indexed constraints from equation (8a)
            for k in range(1, len(distances) + 1):
                for w in N_k_plus_u[k]:
                    for v in N_k_u[k]:
                        if (w,v) in self.E_star:
                            self.model += (
                                rho * k + self.x[w,v] >= pulp.lpSum(
                                    self.y[u,r] for r in range(len(self.R_u[u]))
                                    if self._is_in_k_neighborhood_ordering(r, w, v, k, N_k_u[k])
                                ),
                                f"la_arc_consistency_{u}_{w}_{v}_{k}"
                            )
            
            # Add k-indexed constraints from equation (8b)
            for k in range(1, len(distances) + 1):
                for w in N_k_plus_u[k]:
                    outside_neighbors = [j for j in self.customers + [self.depot_end] 
                                      if j not in N_k_plus_u[k]]
                    if outside_neighbors:
                        self.model += (
                            rho * k + pulp.lpSum(
                                self.x[w,j] for j in outside_neighbors 
                                if (w,j) in self.E_star
                            ) >= pulp.lpSum(
                                self.y[u,r] for r in range(len(self.R_u[u]))
                                if self._is_final_in_k_neighborhood(r, w, k, N_k_plus_u[k])
                            ),
                            f"la_arc_final_{u}_{w}_{k}"
                        )

    def _is_in_k_neighborhood_ordering(self, r, w, v, k, N_k_u):
        """Check if w immediately precedes v in ordering r within k-neighborhood"""
        # r is an index, so we need to get the actual ordering from R_u[u][r]
        # Find out which customer u we're dealing with
        for u in self.R_u:
            if r < len(self.R_u[u]):
                ordering = self.R_u[u][r]
                sequence = ordering['sequence']
                
                # Both customers must be in k-neighborhood
                if w not in N_k_u or v not in N_k_u:
                    return False
                
                # Check if w immediately precedes v
                for i in range(len(sequence)-1):
                    if sequence[i] == w and sequence[i+1] == v:
                        return True
                return False
        return False

    def _is_final_in_k_neighborhood(self, r, w, k, N_k_plus_u):
        """Check if w is final customer in k-neighborhood for ordering r"""
        # r is an index, so we need to get the actual ordering from R_u[u][r]
        # Find out which customer u we're dealing with
        for u in self.R_u:
            if r < len(self.R_u[u]):
                ordering = self.R_u[u][r]
                sequence = ordering['sequence']
                
                # w must be in k-neighborhood
                if w not in N_k_plus_u:
                    return False
                
                # Find position of w in sequence
                try:
                    w_pos = sequence.index(w)
                except ValueError:
                    return False
                
                # Check if w is last in sequence or followed by customer outside k-neighborhood
                return (w_pos == len(sequence)-1 or 
                        sequence[w_pos+1] not in N_k_plus_u)
        return False

    def _add_capacity_flow_constraints(self):
        """Add capacity flow constraints with cumulative load tracking"""
        # Flow conservation (4a)
        for i, k, d_min, d_max in self.nodes_D:
            if i not in [self.depot_start, self.depot_end]:
                self.model += (
                    pulp.lpSum(self.z_D[e] for e in self.edges_D if e[0] == (i,k)) ==
                    pulp.lpSum(self.z_D[e] for e in self.edges_D if e[1] == (i,k)),
                    f"capacity_flow_conservation_{i}_{k}"
                )
        
        # Consistency with route variables (4b)
        for u,v in self.E_star:
            self.model += (
                self.x[u,v] == pulp.lpSum(
                    self.z_D[e] for e in self.edges_D 
                    if e[0][0] == u and e[1][0] == v
                ),
                f"capacity_flow_consistency_{u}_{v}"
            )
        
        # Track cumulative load along paths
        # For each customer, ensure incoming flow carries appropriate load
        for u in self.customers:
            # Sum of incoming flows must respect capacity
            incoming_edges = [(i,j) for i,j in self.edges_D if j[0] == u]
            self.model += (
                pulp.lpSum(
                    self.get_cumulative_load(e[0][0], u) * self.z_D[e]
                    for e in incoming_edges
                ) <= self.vehicle_capacity,
                f"cumulative_capacity_{u}"
            )

        # Enforce that each vehicle starts empty from depot
        depot_out_edges = [e for e in self.edges_D if e[0][0] == self.depot_start]
        self.model += (
            pulp.lpSum(self.z_D[e] for e in depot_out_edges) <= 1,
            "depot_start_capacity"
        )

    def get_cumulative_load(self, start_node, end_node):
        """Calculate cumulative load between two nodes"""
        if start_node == self.depot_start:
            return self.demands.get(end_node, 0)
        elif end_node == self.depot_end:
            return self.demands.get(start_node, 0)
        else:
            return self.demands.get(start_node, 0) + self.demands.get(end_node, 0)

    def calculate_route_load(self, route):
        """Helper function to calculate total load of a route"""
        return sum(self.demands[i] for i in route if i not in [self.depot_start, self.depot_end])

    def validate_solution(self, solution):
        """Validate solution feasibility"""
        if solution['status'] != 'Optimal':
            return False

        # Extract routes
        routes = solution['routes']
        print("\nValidating solution:")
        
        for idx, route in enumerate(routes, 1):
            print(f"\nRoute {idx}: {' -> '.join(['0'] + [str(c) for c in route] + ['4'])}")
            
            # Check capacity
            route_load = self.calculate_route_load(route)
            print(f"  Load: {route_load}/{self.vehicle_capacity}", end=" ")
            if route_load > self.vehicle_capacity:
                print("❌ Exceeds capacity!")
                return False
            print("✓")
            
            # Check time windows
            current_time = 0
            current_loc = self.depot_start
            for stop in route:
                travel_time = self.costs[current_loc, stop] / 5
                arrival_time = max(current_time + travel_time, self.time_windows[stop][0])
                window_start, window_end = self.time_windows[stop]
                
                print(f"  Customer {stop}: Arrive {arrival_time:.1f} Window [{window_start}, {window_end}]", end=" ")
                if arrival_time > window_end:
                    print("❌ Misses window!")
                    return False
                print("✓")
                
                current_time = arrival_time
                current_loc = stop
        
        return True

    def _add_time_flow_constraints(self):
        """Add time flow constraints"""
        # Flow conservation (5a)
        for i, k, t_min, t_max in self.nodes_T:
            if i not in [self.depot_start, self.depot_end]:
                self.model += (
                    pulp.lpSum(self.z_T[e] for e in self.edges_T if e[0] == (i,k)) ==
                    pulp.lpSum(self.z_T[e] for e in self.edges_T if e[1] == (i,k)),
                    f"time_flow_conservation_{i}_{k}"
                )
        
        # Consistency with route variables (5b)
        for u,v in self.E_star:
            self.model += (
                self.x[u,v] == pulp.lpSum(
                    self.z_T[e] for e in self.edges_T 
                    if e[0][0] == u and e[1][0] == v
                ),
                f"time_flow_consistency_{u}_{v}"
            )
        
        # Additional constraint to link time variables τ with time buckets
        for i, k, t_min, t_max in self.nodes_T:
            if i not in [self.depot_start, self.depot_end]:
                outgoing_edges = [e for e in self.edges_T if e[0] == (i,k)]
                if outgoing_edges:
                    M = max(tw[1] for tw in self.time_windows.values())
                    self.model += (
                        self.tau[i] >= t_min - M * (1 - pulp.lpSum(self.z_T[e] for e in outgoing_edges)),
                        f"time_bucket_lb_{i}_{k}"
                    )
                    self.model += (
                        self.tau[i] <= t_max + M * (1 - pulp.lpSum(self.z_T[e] for e in outgoing_edges)),
                        f"time_bucket_ub_{i}_{k}"
                    )

    def analyze_neighborhood_sizes(self):
        """Determine optimal neighborhood sizes using equation (9) from paper"""
        optimal_sizes = {}
        
        # Get dual variables from the solved model
        dual_vars = self._get_dual_variables()
        
        for u in self.customers:
            max_k = 1  # Start with smallest neighborhood
            
            # Get distances to all other customers
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            distances.sort(key=lambda x: x[1])
            max_possible_k = len(distances)
            
            # For each possible k, check if associated dual variables are positive
            for k in range(1, max_possible_k + 1):
                N_k_u = [j for j, _ in distances[:k]]
                N_k_plus_u = [u] + N_k_u
                
                # Sum dual variables according to equation (9)
                dual_sum = 0
                
                # Sum ϖ_uwk terms
                for w in N_k_u:
                    var_name = f"la_arc_final_{u}_{w}_{k}"
                    if var_name in dual_vars:
                        dual_sum += abs(dual_vars[var_name])
                
                # Sum ϖ_uwvk terms
                for w in N_k_plus_u:
                    for v in N_k_u:
                        if v != w:
                            var_name = f"la_arc_consistency_{u}_{w}_{v}_{k}"
                            if var_name in dual_vars:
                                dual_sum += abs(dual_vars[var_name])
                
                # If dual sum is positive, update max_k
                if dual_sum > 1e-6:  # Small threshold for numerical stability
                    max_k = k
            
            optimal_sizes[u] = max_k
        
        return optimal_sizes

    def _get_dual_variables(self):
        """Extract dual variables from solved model"""
        if self.model.status != pulp.LpStatusOptimal:
            return {}
            
        dual_vars = {}
        for name, constraint in self.model.constraints.items():
            if constraint.pi is not None:  # pi attribute holds dual value
                dual_vars[name] = constraint.pi
        
        return dual_vars

    def update_neighborhoods(self):
        """Update neighborhood sizes based on parsimony analysis"""
        # Analyze optimal neighborhood sizes
        optimal_sizes = self.analyze_neighborhood_sizes()
        
        # Update la_neighbors based on analysis
        new_neighbors = {}
        for u in self.customers:
            # Get sorted list of neighbors by distance
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            distances.sort(key=lambda x: x[1])
            
            # Take only the optimal number of neighbors
            k = optimal_sizes[u]
            new_neighbors[u] = [j for j, _ in distances[:k]]
        
        # Update la_neighbors
        self.la_neighbors = new_neighbors
        
        # Regenerate orderings with new neighborhoods
        self.R_u = self._generate_orderings()
        
        return optimal_sizes

    def solve(self, time_limit=None):
        """Solve the VRPTW instance with solution validation"""
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
            active_edges = []
            for (i,j) in self.E_star:
                val = pulp.value(self.x[i,j])
                if val is not None and val > 0.5:
                    print(f"x_{i}_{j} = {val}")
                    active_edges.append((i,j))
            
            print("\nActive edges:", active_edges)
            
            print("\nSolution Details:")
            tau_values = {}
            for i in self.customers + [self.depot_start, self.depot_end]:
                tau_val = pulp.value(self.tau[i])
                if tau_val is not None:
                    print(f"Node {i} - Time: {tau_val:.2f}")
                    tau_values[i] = tau_val
            
            solution['routes'] = self._extract_routes()
            solution['tau_values'] = tau_values
            solution['active_edges'] = active_edges
            
            # Validate solution
            if not self.validate_solution(solution):
                print("\nWARNING: Solution validation failed!")
                solution['status'] = 'Invalid'
        
        return solution

    def debug_model_feasibility(self):
        """Print detailed information about the model to debug infeasibility"""
        print("\nDEBUG INFO:")
        
        # Check capacity graph
        print("\nCapacity Graph:")
        print(f"Number of nodes: {len(self.nodes_D)}")
        print(f"Number of edges: {len(self.edges_D)}")
        print("\nExample edges:")
        for i, edge in enumerate(self.edges_D[:5]):
            (u1, k1), (u2, k2) = edge
            print(f"Edge {i}: ({u1},{k1}) -> ({u2},{k2})")
        
        # Check time windows feasibility
        print("\nTime Windows Feasibility:")
        for i, j in self.E_star:
            travel_time = self.costs[i,j] / 5
            earliest_i = self.time_windows[i][0]
            latest_i = self.time_windows[i][1]
            earliest_j = self.time_windows[j][0]
            latest_j = self.time_windows[j][1]
            
            earliest_possible_arrival = earliest_i + travel_time
            latest_possible_arrival = latest_i + travel_time
            
            if earliest_possible_arrival > latest_j:
                print(f"WARNING: Edge ({i},{j}) might be time-infeasible:")
                print(f"  Travel time: {travel_time}")
                print(f"  Node {i} window: [{earliest_i}, {latest_i}]")
                print(f"  Node {j} window: [{earliest_j}, {latest_j}]")
                print(f"  Earliest possible arrival: {earliest_possible_arrival}")
                print(f"  Latest possible arrival: {latest_possible_arrival}")
        
        # Check LA neighborhoods
        print("\nLA Neighborhoods:")
        for u in self.customers:
            print(f"Customer {u} neighbors: {self.la_neighbors[u]}")
        
        # Check orderings
        print("\nOrderings:")
        for u in self.customers:
            print(f"\nCustomer {u} orderings:")
            for i, r in enumerate(self.R_u[u]):
                print(f"  {i}: {r['sequence']}")

    def solve_with_debugging(self, time_limit=None):
        """Solve the model with debugging information"""
        self.debug_model_feasibility()
        return self.solve(time_limit)

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


def create_small_test_instance():
    """Create a small test instance with 3 customers and relaxed time windows"""
    instance = {
        'customers': [1, 2, 3],
        'depot_start': 0,
        'depot_end': 4,
        'costs': {
            # From depot start to customers and depot end
            (0,1): 10, (0,2): 15, (0,3): 20, (0,4): 0,
            
            # Between customers
            (1,2): 12, (1,3): 18, 
            (2,1): 12, (2,3): 14, 
            (3,1): 18, (3,2): 14,
            
            # From customers to depot end
            (1,4): 10,
            (2,4): 15,
            (3,4): 20,
            
            # Add reverse paths for completeness
            (4,0): 0, (4,1): 10, (4,2): 15, (4,3): 20,
            (2,0): 15, (3,0): 20, (1,0): 10
        },
        'time_windows': {
            0: (0, 200),    # Depot start - extended window
            1: (10, 100),   # Customer 1 - extended window
            2: (20, 150),   # Customer 2 - extended window
            3: (30, 180),   # Customer 3 - extended window
            4: (0, 200)     # Depot end - extended window
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


def test_with_debugging():
    """Run test with additional debugging information"""
    print("Creating test instance...")
    instance = create_small_test_instance()
    
    print("\nInitializing optimizer...")
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        K=2,  # Reduced neighborhood size
        capacity_granularity=2,  # Reduced granularity
        time_granularity=2  # Reduced granularity
    )
    
    print("\nSolving with debugging...")
    solution = optimizer.solve_with_debugging(time_limit=300)
    
    return optimizer, solution


if __name__ == "__main__":
    optimizer, solution = test_with_debugging()