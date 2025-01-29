import pulp
import numpy as np
import time
from collections import defaultdict

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, K=2, time_granularity=3, capacity_granularity=3, max_iterations=5):
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs
        self.time_windows = time_windows
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.K = K
        self.time_granularity = time_granularity
        self.capacity_granularity = capacity_granularity
        self.max_iterations = max_iterations
        
        # Algorithm parameters
        self.MIN_INC = 1  # Minimum improvement threshold
        self.sigma = 9    # Reset trigger threshold
        
        # Initialize model
        self.model = None
        self.create_initial_model()
    
    def create_initial_model(self):
        """Create initial model with all components"""
        self.model = pulp.LpProblem("VRPTW", pulp.LpMinimize)
        
        # Create valid edges
        self.E_star = [(i,j) for i in [self.depot_start] + self.customers 
                    for j in self.customers + [self.depot_end] if i != j]
        
        # Generate LA neighborhoods
        self.la_neighbors = self._generate_initial_la_neighbors()
        
        # Generate orderings
        self.R_u = self._generate_orderings()
        
        # Create time and capacity discretization
        self.T_u = self._create_time_buckets()
        self.D_u = self._create_capacity_buckets()
        
        # Create flow graphs first
        self.nodes_T, self.edges_T = self._create_time_graph()
        self.nodes_D, self.edges_D = self._create_capacity_graph()
        
        # Then create variables that depend on the graphs
        self._create_variables()
        
        # Finally add constraints
        self._add_constraints()

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
        
        # Time flow variables z_T
        self.z_T = pulp.LpVariable.dicts("z_T",
                                        self.edges_T,
                                        lowBound=0)

        # Capacity flow variables z_D
        self.z_D = pulp.LpVariable.dicts("z_D",
                                        self.edges_D,
                                        lowBound=0)
                                        
        # Load tracking variables
        self.load = pulp.LpVariable.dicts("load",
                                        self.customers,
                                        lowBound=0,
                                        upBound=self.vehicle_capacity)
 
    def _create_capacity_buckets(self):
        """Create initial capacity buckets for each customer"""
        buckets = {}
        
        # Create buckets for each customer
        for u in self.customers:
            demand = self.demands[u]
            remaining_capacity = self.vehicle_capacity - demand
            bucket_size = remaining_capacity / self.capacity_granularity
            
            customer_buckets = []
            current_capacity = demand
            
            # Create evenly spaced buckets
            for i in range(self.capacity_granularity):
                lower = current_capacity
                upper = min(current_capacity + bucket_size, self.vehicle_capacity)
                customer_buckets.append((lower, upper))
                current_capacity = upper
                
                if current_capacity >= self.vehicle_capacity:
                    break
            
            buckets[u] = customer_buckets
        
        # Add single bucket for depot (start and end)
        buckets[self.depot_start] = [(0, self.vehicle_capacity)]
        buckets[self.depot_end] = [(0, self.vehicle_capacity)]
        
        return buckets

    def _create_capacity_graph(self):
        """Create directed graph GD for capacity flow"""
        nodes_D = []  # (u, k, d⁻, d⁺)
        edges_D = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and their capacity buckets
        for u in self.customers:
            for k, (d_min, d_max) in enumerate(self.D_u[u]):
                nodes_D.append((u, k, d_min, d_max))
        
        # Add depot nodes
        depot_start_bucket = self.D_u[self.depot_start][0]
        depot_end_bucket = self.D_u[self.depot_end][0]
        nodes_D.append((self.depot_start, 0, depot_start_bucket[0], depot_start_bucket[1]))
        nodes_D.append((self.depot_end, 0, depot_end_bucket[0], depot_end_bucket[1]))
        
        # Create edges between nodes
        for i, k_i, d_min_i, d_max_i in nodes_D:
            for j, k_j, d_min_j, d_max_j in nodes_D:
                if i != j:
                    # Check if edge is feasible based on capacity
                    demand_j = self.demands[j]
                    remaining_i = d_max_i - self.demands[i]
                    
                    # Edge is feasible if remaining capacity after i can accommodate j
                    if (remaining_i >= demand_j and 
                        d_min_j >= demand_j and 
                        d_max_j <= self.vehicle_capacity):
                        edges_D.append(((i,k_i), (j,k_j)))
        
        return nodes_D, edges_D
        
    def _create_time_buckets(self):
        """Create time buckets for each customer"""
        buckets = {}
        
        # Create buckets for each customer
        for u in self.customers:
            earliest_time, latest_time = self.time_windows[u]
            time_span = latest_time - earliest_time
            bucket_size = time_span / self.time_granularity
            
            customer_buckets = []
            for i in range(self.time_granularity):
                lower = earliest_time + i * bucket_size
                upper = (earliest_time + (i + 1) * bucket_size 
                        if i < self.time_granularity - 1 else latest_time)
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
            for k, (t_min, t_max) in enumerate(self.T_u[u]):  # Changed from time_buckets to T_u
                nodes_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        depot_start_bucket = self.T_u[self.depot_start][0]  # Changed from time_buckets to T_u
        depot_end_bucket = self.T_u[self.depot_end][0]      # Changed from time_buckets to T_u
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

    def _merge_capacity_buckets(self, dual_vars):
        """Merge capacity buckets when their dual variables are equal"""
        print("\nMerging capacity buckets...")
        for u in self.customers:
            print(f"\nCustomer {u}:")
            print(f"  Before merge: {self.D_u[u]}")
            
            buckets_to_merge = []
            
            # Get all consecutive bucket pairs for this customer
            for k in range(len(self.D_u[u]) - 1):
                i = (u, k)    # First bucket node
                j = (u, k+1)  # Next bucket node
                
                # Get dual variables for these nodes
                dual_i = dual_vars.get(f"capacity_flow_conservation_{u}_{k}", 0)
                dual_j = dual_vars.get(f"capacity_flow_conservation_{u}_{k+1}", 0)
                
                print(f"  Comparing buckets {k} and {k+1}:")
                print(f"    Dual values: {dual_i:.6f} vs {dual_j:.6f}")
                
                # If duals are equal (within numerical tolerance), mark for merging
                if abs(dual_i - dual_j) < 1e-6:
                    print(f"    -> Will merge")
                    buckets_to_merge.append((k, k+1))
                else:
                    print(f"    -> Keep separate")
            
            # Merge marked buckets (work backwards to avoid index issues)
            for k1, k2 in reversed(buckets_to_merge):
                lower = self.D_u[u][k1][0]  # Lower bound of first bucket
                upper = self.D_u[u][k2][1]  # Upper bound of second bucket
                # Remove the two original buckets and insert merged bucket
                self.D_u[u].pop(k2)
                self.D_u[u].pop(k1)
                self.D_u[u].insert(k1, (lower, upper))
            
            print(f"  After merge: {self.D_u[u]}")

    def _merge_time_buckets(self, dual_vars):
        """Merge time buckets when their dual variables are equal"""
        print("\nMerging time buckets...")
        for u in self.customers:
            print(f"\nCustomer {u}:")
            print(f"  Before merge: {self.T_u[u]}")
            
            buckets_to_merge = []
            
            # Get all consecutive bucket pairs for this customer
            for k in range(len(self.T_u[u]) - 1):
                i = (u, k)    # First bucket node
                j = (u, k+1)  # Next bucket node
                
                # Get dual variables for these nodes
                dual_i = dual_vars.get(f"time_flow_conservation_{u}_{k}", 0)
                dual_j = dual_vars.get(f"time_flow_conservation_{u}_{k+1}", 0)
                
                print(f"  Comparing buckets {k} and {k+1}:")
                print(f"    Dual values: {dual_i:.6f} vs {dual_j:.6f}")
                
                # If duals are equal (within numerical tolerance), mark for merging
                if abs(dual_i - dual_j) < 1e-6:
                    print(f"    -> Will merge")
                    buckets_to_merge.append((k, k+1))
                else:
                    print(f"    -> Keep separate")
            
            # Merge marked buckets (work backwards to avoid index issues)
            for k1, k2 in reversed(buckets_to_merge):
                lower = self.T_u[u][k1][0]  # Lower bound of first bucket
                upper = self.T_u[u][k2][1]  # Upper bound of second bucket
                # Remove the two original buckets and insert merged bucket
                self.T_u[u].pop(k2)
                self.T_u[u].pop(k1)
                self.T_u[u].insert(k1, (lower, upper))
            
            print(f"  After merge: {self.T_u[u]}")

    def _is_significant_flow(self, flow, u_i, u_j):
        """Determine if a flow is significant enough to trigger bucket expansion"""
        # Flow should be significantly non-zero
        if flow < 1e-4:
            return False
            
        # Skip flows between customers that are too far apart in time
        # (these are less likely to be in optimal solution)
        travel_time = self.costs[u_i, u_j] / 5
        earliest_i = self.time_windows[u_i][0]
        latest_j = self.time_windows[u_j][1]
        if earliest_i + travel_time > latest_j - 10:  # 10 time units buffer
            return False
            
        # Skip flows between customers that would exceed capacity
        remaining_capacity = self.vehicle_capacity - self.demands[u_i]
        if self.demands[u_j] > remaining_capacity * 0.9:  # 90% threshold
            return False
            
        # Skip flows that would create very small buckets
        min_bucket_size = (self.vehicle_capacity - min(self.demands.values())) * 0.1  # 10% of max remaining
        if remaining_capacity < min_bucket_size:
            return False
            
        return True
    
    def _expand_capacity_buckets(self, z_D):
        """Add new capacity thresholds based on flow solution"""
        print("\nExpanding capacity buckets...")
        
        # First, create a mapping of flows to actual bucket indices
        flow_mapping = {}
        for (i, j), flow in z_D.items():
            u_i, k_i = i  # Source node customer and bucket index
            u_j, k_j = j  # Target node customer and bucket index
            
            # Skip depot nodes and insignificant flows
            if u_j in [self.depot_start, self.depot_end]:
                continue
                
            if not self._is_significant_flow(flow, u_i, u_j):
                print(f"Skipping insignificant flow ({u_i},{k_i}) -> ({u_j},{k_j}) with flow {flow}")
                continue
                
            # Find current bucket indices after merging
            source_bucket_idx = None
            target_bucket_idx = None
            
            # Find which bucket contains the original k_i index's value
            for idx, (lower, upper) in enumerate(self.D_u[u_i]):
                if k_i * ((self.vehicle_capacity - self.demands[u_i]) / 3) <= upper:
                    source_bucket_idx = idx
                    break
                    
            # Find which bucket contains the original k_j index's value
            for idx, (lower, upper) in enumerate(self.D_u[u_j]):
                if k_j * ((self.vehicle_capacity - self.demands[u_j]) / 3) <= upper:
                    target_bucket_idx = idx
                    break
                    
            if source_bucket_idx is not None and target_bucket_idx is not None:
                flow_mapping[(u_i, source_bucket_idx, u_j, target_bucket_idx)] = flow
        
        # Now process flows with correct bucket indices
        for (u_i, k_i, u_j, k_j), flow in flow_mapping.items():
            print(f"\nProcessing flow ({u_i},{k_i}) -> ({u_j},{k_j}) with flow {flow}")
            print(f"Source customer {u_i} buckets: {self.D_u[u_i]}")
            print(f"Target customer {u_j} buckets: {self.D_u[u_j]}")
            
            # Calculate new threshold
            d_plus_i = self.D_u[u_i][k_i][1]  # Upper bound of source bucket
            d_u_i = self.demands[u_i]         # Demand of source customer
            new_threshold = d_plus_i - d_u_i
            
            print(f"  New threshold calculated: {new_threshold}")
            print(f"  Current buckets for customer {u_j}: {self.D_u[u_j]}")
            
            # Add new threshold if it's not already present and it's meaningful
            found = False
            for bucket in self.D_u[u_j]:
                if abs(bucket[1] - new_threshold) < 1e-6:
                    found = True
                    print("  -> Threshold already exists")
                    break
                
            if not found and self.demands[u_j] < new_threshold < self.vehicle_capacity:
                print("  -> Adding new threshold")
                # Insert new bucket maintaining sorted order
                for k, bucket in enumerate(self.D_u[u_j]):
                    if new_threshold < bucket[1]:
                        # Split existing bucket at new threshold
                        if new_threshold > bucket[0]:
                            self.D_u[u_j].insert(k+1, (new_threshold, bucket[1]))
                            self.D_u[u_j][k] = (bucket[0], new_threshold)
                            print(f"  Updated buckets: {self.D_u[u_j]}")
                            break

    def _expand_time_buckets(self, z_T):
        """Add new time thresholds based on flow solution"""
        print("\nExpanding time buckets...")
        
        # First, create a mapping of flows to actual bucket indices
        flow_mapping = {}
        for (i, j), flow in z_T.items():
            u_i, k_i = i  # Source node customer and bucket index
            u_j, k_j = j  # Target node customer and bucket index
            
            # Skip depot nodes and insignificant flows
            if u_j in [self.depot_start, self.depot_end]:
                continue
                
            if not self._is_significant_flow(flow, u_i, u_j):
                print(f"Skipping insignificant flow ({u_i},{k_i}) -> ({u_j},{k_j}) with flow {flow}")
                continue
                
            # Find current bucket indices after merging
            source_bucket_idx = None
            target_bucket_idx = None
            
            # Find which bucket contains the original k_i index's value
            time_span_i = self.time_windows[u_i][1] - self.time_windows[u_i][0]
            for idx, (lower, upper) in enumerate(self.T_u[u_i]):
                if k_i * (time_span_i / 3) <= upper:
                    source_bucket_idx = idx
                    break
                    
            # Find which bucket contains the original k_j index's value
            time_span_j = self.time_windows[u_j][1] - self.time_windows[u_j][0]
            for idx, (lower, upper) in enumerate(self.T_u[u_j]):
                if k_j * (time_span_j / 3) <= upper:
                    target_bucket_idx = idx
                    break
                    
            if source_bucket_idx is not None and target_bucket_idx is not None:
                flow_mapping[(u_i, source_bucket_idx, u_j, target_bucket_idx)] = flow
        
        # Now process flows with correct bucket indices
        for (u_i, k_i, u_j, k_j), flow in flow_mapping.items():
            print(f"\nProcessing flow ({u_i},{k_i}) -> ({u_j},{k_j}) with flow {flow}")
            print(f"Source customer {u_i} buckets: {self.T_u[u_i]}")
            print(f"Target customer {u_j} buckets: {self.T_u[u_j]}")
            
            # Calculate new threshold
            t_plus_i = self.T_u[u_i][k_i][1]  # Upper bound of source bucket
            travel_time = self.costs[u_i,u_j] / 5  # Travel time between customers
            t_plus_j = self.T_u[u_j][k_j][1]  # Upper bound of target bucket
            
            new_threshold = min(t_plus_i - travel_time, t_plus_j)
            
            print(f"  New threshold calculated: {new_threshold}")
            print(f"  Current buckets for customer {u_j}: {self.T_u[u_j]}")
            
            # Add new threshold if it's not already present and it's meaningful
            found = False
            for bucket in self.T_u[u_j]:
                if abs(bucket[1] - new_threshold) < 1e-6:
                    found = True
                    print("  -> Threshold already exists")
                    break
                
            if not found and self.time_windows[u_j][0] < new_threshold < self.time_windows[u_j][1]:
                print("  -> Adding new threshold")
                # Insert new bucket maintaining sorted order
                for k, bucket in enumerate(self.T_u[u_j]):
                    if new_threshold < bucket[1]:
                        # Split existing bucket at new threshold
                        if new_threshold > bucket[0]:
                            self.T_u[u_j].insert(k+1, (new_threshold, bucket[1]))
                            self.T_u[u_j][k] = (bucket[0], new_threshold)
                            print(f"  Updated buckets: {self.T_u[u_j]}")
                            break

    def _update_bucket_graphs(self):
        """Update time and capacity graphs after bucket modifications"""
        self.nodes_T, self.edges_T = self._create_time_graph()
        self.nodes_D, self.edges_D = self._create_capacity_graph()

    def _validate_buckets(self):
        """Validate bucket structures after modifications"""
        # Check capacity buckets
        for u in self.customers:
            # Verify bucket continuity
            prev_upper = None
            for i, (lower, upper) in enumerate(self.D_u[u]):
                # Check bounds
                if lower >= upper:
                    raise ValueError(f"Invalid capacity bucket bounds for customer {u}: [{lower}, {upper}]")
                    
                # Check ordering
                if prev_upper is not None and abs(lower - prev_upper) > 1e-6:
                    raise ValueError(f"Gap in capacity buckets for customer {u} between {prev_upper} and {lower}")
                    
                # Check within vehicle capacity
                if upper > self.vehicle_capacity:
                    raise ValueError(f"Capacity bucket for customer {u} exceeds vehicle capacity: {upper}")
                    
                prev_upper = upper
            
        # Check time buckets
        for u in self.customers:
            # Verify bucket continuity
            prev_upper = None
            for i, (lower, upper) in enumerate(self.T_u[u]):
                # Check bounds
                if lower >= upper:
                    raise ValueError(f"Invalid time bucket bounds for customer {u}: [{lower}, {upper}]")
                    
                # Check ordering
                if prev_upper is not None and abs(lower - prev_upper) > 1e-6:
                    raise ValueError(f"Gap in time buckets for customer {u} between {prev_upper} and {lower}")
                    
                # Check within time windows
                if upper > self.time_windows[u][1] or lower < self.time_windows[u][0]:
                    raise ValueError(f"Time bucket for customer {u} outside time window: [{lower}, {upper}]")
                    
                prev_upper = upper
    
    def _generate_initial_la_neighbors(self):
        """Generate initial LA neighborhoods using K closest neighbors"""
        la_neighbors = {}
        for u in self.customers:
            # Get distances to all other customers
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            distances.sort(key=lambda x: x[1])
            
            # Take K closest neighbors that are reachable
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

    def _add_constraints(self):
        """Add all constraints"""
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

        # LA-arc constraints with parsimony
        self._add_la_arc_constraints_with_parsimony()
        
        # Capacity constraints (for load tracking)
        self._add_capacity_constraints()
        
        # Time flow constraints
        self._add_time_flow_constraints()
        
        # Capacity flow constraints - Add this new line
        self._add_capacity_flow_constraints()

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

    def _add_capacity_constraints(self):
        """Add capacity flow constraints"""
        # Initial load from depot
        for j in self.customers:
            self.model += (
                self.load[j] >= self.demands[j] * self.x[self.depot_start,j],
                f"initial_load_{j}"
            )
        
        # Load propagation between customers
        for i in self.customers:
            for j in self.customers:
                if i != j and (i,j) in self.E_star:
                    M = self.vehicle_capacity
                    self.model += (
                        self.load[j] >= self.load[i] + self.demands[j] - M * (1 - self.x[i,j]),
                        f"load_propagation_{i}_{j}"
                    )
        
        # Enforce capacity limit
        for i in self.customers:
            self.model += (
                self.load[i] <= self.vehicle_capacity,
                f"capacity_limit_{i}"
            )

    def _add_capacity_flow_constraints(self):
        """Add capacity flow constraints (equations 4a and 4b from paper)"""
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
        
        # Link time variables τ with time buckets
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
            max_k = 1
            
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
                if dual_sum > 1e-6:
                    max_k = k
            
            optimal_sizes[u] = max_k
        
        return optimal_sizes

    def _get_dual_variables(self):
        """Extract dual variables from solved model"""
        if self.model.status != pulp.LpStatusOptimal:
            return {}
            
        dual_vars = {}
        for name, constraint in self.model.constraints.items():
            if constraint.pi is not None:
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

    def solve_with_parsimony(self, time_limit=None):
        """Solve VRPTW with LA neighborhood parsimony and bucket management"""
        print("\nSolving VRPTW with LA neighborhood parsimony...")
        
        iteration = 1
        last_lp_val = float('-inf')
        no_improvement_count = 0
        
        while iteration <= self.max_iterations:
            print(f"\nIteration {iteration}")
            
            # Print initial bucket state
            print("\nInitial bucket state:")
            print("\nCapacity buckets:")
            for u in self.customers:
                print(f"Customer {u}: {self.D_u[u]}")
            print("\nTime buckets:")
            for u in self.customers:
                print(f"Customer {u}: {self.T_u[u]}")
            
            # Solve current model
            if time_limit:
                status = self.model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
            else:
                status = self.model.solve(pulp.PULP_CBC_CMD())
            
            if status != pulp.LpStatusOptimal:
                print(f"Failed to solve model: {pulp.LpStatus[status]}")
                break
                
            current_objective = pulp.value(self.model.objective)
            print(f"Current objective: {current_objective}")
            
            # Get solution components for bucket management
            dual_vars = self._get_dual_variables()
            
            # Safely get flow variables with non-zero values
            z_D = {}
            z_T = {}
            for (i,j), var in self.z_D.items():
                val = pulp.value(var)
                if val is not None and val > 1e-6:
                    z_D[(i,j)] = val
                    
            for (i,j), var in self.z_T.items():
                val = pulp.value(var)
                if val is not None and val > 1e-6:
                    z_T[(i,j)] = val
                    
            # Update bucket structure
            print("\nUpdating bucket structure...")
            self._merge_capacity_buckets(dual_vars)
            self._merge_time_buckets(dual_vars)
            
            if z_D:  # Only try to expand if we have non-zero flows
                self._expand_capacity_buckets(z_D)
            if z_T:  # Only try to expand if we have non-zero flows
                self._expand_time_buckets(z_T)
            
            # Print final bucket state
            print("\nFinal bucket state:")
            print("\nCapacity buckets:")
            for u in self.customers:
                print(f"Customer {u}: {self.D_u[u]}")
            print("\nTime buckets:")
            for u in self.customers:
                print(f"Customer {u}: {self.T_u[u]}")
            
            # Update neighborhood sizes based on dual variables
            optimal_sizes = self.analyze_neighborhood_sizes()
            print("\nOptimal neighborhood sizes:")
            for u in sorted(optimal_sizes.keys()):
                print(f"Customer {u}: {optimal_sizes[u]} neighbors")
            
            # Check for improvement
            if current_objective < last_lp_val - 1e-6:
                best_objective = current_objective
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Stop if no improvement for 2 consecutive iterations
            if no_improvement_count >= 2:
                print("\nNo improvement for 2 iterations. Stopping.")
                break
                
            # Update for next iteration
            self._update_bucket_graphs()
            last_lp_val = current_objective
            iteration += 1
        
        # Extract and validate final solution
        solution = self._extract_solution()
        if solution['status'] == 'Optimal':
            self.validate_solution(solution)
        
        return solution

    def solve(self, time_limit=None):
        """Legacy solve method - now calls solve_with_parsimony"""
        return self.solve_with_parsimony(time_limit)

    def _extract_solution(self):
        """Extract solution details"""
        if self.model.status != pulp.LpStatusOptimal:
            return {
                'status': pulp.LpStatus[self.model.status],
                'objective': None,
                'routes': None,
                'computation_time': self.model.solutionTime,
                'neighborhood_sizes': None
            }

        solution = {
            'status': pulp.LpStatus[self.model.status],
            'objective': pulp.value(self.model.objective),
            'routes': self._extract_routes(),
            'computation_time': self.model.solutionTime,
            'neighborhood_sizes': {u: len(neighbors) for u, neighbors in self.la_neighbors.items()}
        }
        
        return solution

    def _extract_routes(self):
        """Extract routes from solution"""
        if self.model.status != pulp.LpStatusOptimal:
            return None
            
        active_edges = [(i,j) for (i,j) in self.E_star 
                       if pulp.value(self.x[i,j]) is not None 
                       and pulp.value(self.x[i,j]) > 0.5]
        
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

    def validate_solution(self, solution):
        """Validate solution feasibility"""
        if solution['status'] != 'Optimal':
            return False

        routes = solution['routes']
        print("\nValidating solution:")
        
        for idx, route in enumerate(routes, 1):
            print(f"\nRoute {idx}: {' -> '.join(map(str, [0] + route + [self.depot_end]))}")
            
            # Check capacity
            route_load = sum(self.demands[i] for i in route)
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
    

def test_enhanced_solver():
    """Test the enhanced solver with parsimony"""
    # Create a test instance (using previously defined function)
    instance = create_test_instance()  # 5-customer instance
    
    # Initialize optimizer with parsimony
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        K=3,  # Initial neighborhood size
        max_iterations=5
    )
    
    # Solve with parsimony
    print("\nSolving with parsimony...")
    solution = optimizer.solve_with_parsimony(time_limit=300)
    
    # Print results
    print("\nSolution Results:")
    print(f"Status: {solution['status']}")
    print(f"Objective: {solution['objective']}")
    print(f"Computation time: {solution['computation_time']:.2f} seconds")
    print("\nFinal neighborhood sizes:")
    for u, size in sorted(solution['neighborhood_sizes'].items()):
        print(f"Customer {u}: {size} neighbors")
    
    print("\nRoutes:")
    for i, route in enumerate(solution['routes'], 1):
        print(f"Route {i}: {' -> '.join(str(x) for x in [0] + route + [optimizer.depot_end])}")
    
    # Validate solution
    is_valid = optimizer.validate_solution(solution)
    print(f"\nSolution is {'valid' if is_valid else 'invalid'}")
    
    return optimizer, solution

def test_large_instance():
    """Test with 25-customer instance"""
    print("Creating instance with 25 customers...")
    instance = create_large_test_instance()
    
    print("\nProblem characteristics:")
    total_demand = sum(instance['demands'][i] for i in instance['customers'])
    min_vehicles = (total_demand + instance['vehicle_capacity'] - 1) // instance['vehicle_capacity']
    
    print(f"Number of customers: {len(instance['customers'])}")
    print(f"Vehicle capacity: {instance['vehicle_capacity']}")
    print(f"Total demand: {total_demand}")
    print(f"Minimum vehicles needed: {min_vehicles}")
    
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        K=4,
        max_iterations=5
    )
    
    solution = optimizer.solve_with_parsimony(time_limit=600)
    
    # Validate solution
    is_valid = optimizer.validate_solution(solution)
    print(f"\nSolution is {'valid' if is_valid else 'invalid'}")
    
    return optimizer, solution

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
        1: (10, 40),    # Customer 1
        2: (30, 70),    # Customer 2
        3: (50, 80),    # Customer 3
        4: (20, 60),    # Customer 4
        5: (40, 90),    # Customer 5
        6: (0, 100)     # Depot end
    }
    
    demands = {
        0: 0,     # Depot
        1: 5,     # Customer 1
        2: 7,     # Customer 2
        3: 6,     # Customer 3
        4: 4,     # Customer 4
        5: 8,     # Customer 5
        6: 0      # Depot end
    }
    
    return {
        'customers': list(range(1, 6)),
        'depot_start': 0,
        'depot_end': 6,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 15
    }

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
    time_windows = {
        0: (0, 300),     # Depot start
        26: (0, 300)     # Depot end
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
    np.random.seed(42)  # For reproducibility
    demands = {0: 0, 26: 0}  # Zero demand for depots
    for i in range(1, 26):
        if i % 3 == 0:
            demands[i] = np.random.randint(9, 13)  # Large order
        elif i % 3 == 1:
            demands[i] = np.random.randint(6, 9)   # Medium order
        else:
            demands[i] = np.random.randint(3, 6)   # Small order
    
    # Vehicle capacity to handle about 3-4 customers per route
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

def create_small_test_instance():
    """Create a small test instance with 3 customers"""
    # Create locations (to generate costs consistently)
    locations = {
        0: (0, 0),    # Depot start
        1: (2, 4),    # Customer 1
        2: (-1, 3),   # Customer 2
        3: (4, 1),    # Customer 3
        4: (0, 0)     # Depot end (same as start)
    }
    
    # Calculate costs based on Euclidean distance * 5
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 5)
    
    instance = {
        'customers': [1, 2, 3],
        'depot_start': 0,
        'depot_end': 4,
        'costs': costs,
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

def run_instance_test(instance, instance_name, K=3, max_iterations=5, time_limit=300):
    """Run a single instance test with detailed output"""
    print(f"\nTesting {instance_name}...")
    
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
    
    solution = optimizer.solve_with_parsimony(time_limit=time_limit)
    
    # Print detailed results
    print("\nSolution Results:")
    print(f"Status: {solution['status']}")
    print(f"Objective: {solution['objective']}")
    print(f"Computation time: {solution['computation_time']:.2f} seconds")
    
    print("\nFinal neighborhood sizes:")
    for u, size in sorted(solution['neighborhood_sizes'].items()):
        print(f"Customer {u}: {size} neighbors")
    
    print("\nRoutes:")
    for i, route in enumerate(solution['routes'], 1):
        print(f"Route {i}: {' -> '.join(str(x) for x in [optimizer.depot_start] + route + [optimizer.depot_end])}")
    
    # Validate solution
    is_valid = optimizer.validate_solution(solution)
    print(f"\nSolution is {'valid' if is_valid else 'invalid'}")
    
    return optimizer, solution

def test_all_instances():
    """Test all instance sizes"""
    results = {}
    
    # Test 3-customer instance
    print("\n1. Testing with 3-customer instance...")
    instance_3 = create_small_test_instance()
    optimizer_3 = VRPTWOptimizer(
        customers=instance_3['customers'],
        depot_start=instance_3['depot_start'],
        depot_end=instance_3['depot_end'],
        costs=instance_3['costs'],
        time_windows=instance_3['time_windows'],
        demands=instance_3['demands'],
        vehicle_capacity=instance_3['vehicle_capacity'],
        K=2,
        time_granularity=3,
        capacity_granularity=3,
        max_iterations=3
    )
    solution_3 = optimizer_3.solve_with_parsimony(time_limit=300)
    results['3-customer'] = {'optimizer': optimizer_3, 'solution': solution_3}
    
    # Test 5-customer instance
    print("\n2. Testing with 5-customer instance...")
    instance_5 = create_test_instance()
    optimizer_5 = VRPTWOptimizer(
        customers=instance_5['customers'],
        depot_start=instance_5['depot_start'],
        depot_end=instance_5['depot_end'],
        costs=instance_5['costs'],
        time_windows=instance_5['time_windows'],
        demands=instance_5['demands'],
        vehicle_capacity=instance_5['vehicle_capacity'],
        K=3,
        time_granularity=3,
        capacity_granularity=3,
        max_iterations=4
    )
    solution_5 = optimizer_5.solve_with_parsimony(time_limit=300)
    results['5-customer'] = {'optimizer': optimizer_5, 'solution': solution_5}
    
    return results

if __name__ == "__main__":
    print("Testing VRPTW Solver with LA Neighborhood Parsimony")
    results = test_all_instances()
    print("\nAll tests completed!")
    
    # Compare results
    print("\nSummary:")
    for instance_name, result in results.items():
        solution = result['solution']
        print(f"\n{instance_name}:")
        print(f"  Status: {solution['status']}")
        print(f"  Objective: {solution['objective']}")
        print(f"  Computation time: {solution['computation_time']:.2f} seconds")
        print(f"  Number of routes: {len(solution['routes'])}")