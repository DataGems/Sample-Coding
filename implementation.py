import gurobipy as gp
from gurobipy import GRB
import numpy as np

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, local_area_neighbors, time_buckets, capacity_buckets):
        """
        Initialize VRPTW optimizer with problem parameters
        
        Parameters:
        customers: List of customer indices
        depot_start: Starting depot index (α)
        depot_end: Ending depot index (ᾱ)
        costs: Dictionary of travel costs between locations {(i,j): cost}
        time_windows: Dictionary of time windows {i: (earliest, latest)}
        demands: Dictionary of customer demands {i: demand}
        vehicle_capacity: Maximum vehicle capacity (d0)
        local_area_neighbors: Dictionary of LA neighbors for each customer {i: [neighbors]}
        time_buckets: Dictionary of time discretization {i: [(t_min, t_max)]}
        capacity_buckets: Dictionary of capacity discretization {i: [(d_min, d_max)]}
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
        self.model = gp.Model("VRPTW")
        
        # Create sets of valid edges
        self.E_star = self._create_valid_edges()
        
        # Create orderings for LA arcs
        self.R_u = self._generate_orderings()
        
        # Create capacity and time graphs
        self.G_D, self.E_D = self._create_capacity_graph()
        self.G_T, self.E_T = self._create_time_graph()
        
        # Initialize variables and constraints
        self._create_variables()
        self._add_constraints()
        
    def _create_valid_edges(self):
        """Create set of valid edges considering time windows and capacity"""
        E_star = []
        for i in self.customers + [self.depot_start]:
            for j in self.customers + [self.depot_end]:
                if i != j:
                    # Check if edge is feasible with respect to time windows
                    if self._is_time_feasible(i, j):
                        E_star.append((i, j))
        return E_star
    
    def _is_time_feasible(self, i, j):
        """Check if edge (i,j) is feasible with respect to time windows"""
        earliest_i, latest_i = self.time_windows[i]
        earliest_j, latest_j = self.time_windows[j]
        travel_time = self.costs[(i,j)]  # Assuming costs = travel times for simplicity
        
        if earliest_i + travel_time <= latest_j:
            return True
        return False
    
    def _generate_orderings(self):
        """Generate efficient orderings for LA arcs"""
        R_u = {}
        for u in self.customers:
            R_u[u] = self._generate_customer_orderings(u)
        return R_u
    
    def _generate_customer_orderings(self, u):
        """Generate efficient orderings for customer u"""
        # This is a simplified version - in practice, you'd use dynamic programming
        # as mentioned in Section 6 of the paper
        orderings = []
        neighbors = self.local_area_neighbors[u]
        
        # Generate simple orderings (just direct connections)
        for v in neighbors:
            if self._is_time_feasible(u, v):
                orderings.append({'sequence': [u, v], 'a_wv': {(u,v): 1}, 'a_star': {v: 1}})
        
        return orderings
    
    def _create_capacity_graph(self):
        """Create capacity graph G_D and edge set E_D"""
        G_D = []
        E_D = []
        
        # Create nodes
        for u in self.customers:
            for k, (d_min, d_max) in enumerate(self.capacity_buckets[u], 1):
                G_D.append((u, k, d_min, d_max))
        
        # Add depot nodes
        G_D.append((self.depot_start, 1, self.vehicle_capacity, self.vehicle_capacity))
        G_D.append((self.depot_end, 1, 0, self.vehicle_capacity))
        
        # Create edges (simplified version)
        for i, k_i, d_min_i, d_max_i in G_D:
            for j, k_j, d_min_j, d_max_j in G_D:
                if i != j and self._is_capacity_edge_valid(d_min_i, d_max_i, d_min_j, d_max_j, self.demands.get(i, 0)):
                    E_D.append(((i,k_i), (j,k_j)))
        
        return G_D, E_D
    
    def _create_time_graph(self):
        """Create time graph G_T and edge set E_T"""
        # Similar to _create_capacity_graph but for time buckets
        G_T = []
        E_T = []
        
        # Create nodes
        for u in self.customers:
            for k, (t_min, t_max) in enumerate(self.time_buckets[u], 1):
                G_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        earliest_start, latest_start = self.time_windows[self.depot_start]
        earliest_end, latest_end = self.time_windows[self.depot_end]
        G_T.append((self.depot_start, 1, earliest_start, latest_start))
        G_T.append((self.depot_end, 1, earliest_end, latest_end))
        
        # Create edges (simplified version)
        for i, k_i, t_min_i, t_max_i in G_T:
            for j, k_j, t_min_j, t_max_j in G_T:
                if i != j and self._is_time_edge_valid(t_min_i, t_max_i, t_min_j, t_max_j, self.costs.get((i,j), 0)):
                    E_T.append(((i,k_i), (j,k_j)))
        
        return G_T, E_T
    
    def _create_variables(self):
        """Create all decision variables for the model"""
        # Create x variables (binary route variables)
        self.x = {}
        for i, j in self.E_star:
            self.x[i,j] = self.model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        # Create y variables (ordering selection variables)
        self.y = {}
        for u in self.customers:
            for r, ordering in enumerate(self.R_u[u]):
                self.y[u,r] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'y_{u}_{r}')
        
        # Create z variables for capacity graph
        self.z_D = {}
        for edge in self.E_D:
            self.z_D[edge] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'z_D_{edge}')
        
        # Create z variables for time graph
        self.z_T = {}
        for edge in self.E_T:
            self.z_T[edge] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'z_T_{edge}')
        
        # Create δ and τ variables for original compact formulation
        self.delta = {}
        self.tau = {}
        for u in self.customers + [self.depot_start, self.depot_end]:
            self.delta[u] = self.model.addVar(lb=self.demands.get(u, 0), ub=self.vehicle_capacity, 
                                            name=f'delta_{u}')
            self.tau[u] = self.model.addVar(lb=self.time_windows[u][0], ub=self.time_windows[u][1], 
                                          name=f'tau_{u}')
    
    def _add_constraints(self):
        """Add all constraints from Equation 6"""
        # Original compact constraints (6b-6f)
        self._add_original_compact_constraints()
        
        # LA-arc movement consistency constraints (6g-6i)
        self._add_la_arc_constraints()
        
        # Flow graph capacity constraints (6j-6k)
        self._add_capacity_flow_constraints()
        
        # Flow graph time constraints (6l-6m)
        self._add_time_flow_constraints()
        
        # Set objective (6a)
        self._set_objective()
    
    def _add_original_compact_constraints(self):
        """Add constraints 6b-6f"""
        # Constraint 6b: Each customer must be visited exactly once
        for u in self.customers:
            self.model.addConstr(gp.quicksum(self.x[i,u] for i,j in self.E_star if j == u) == 1)
        
        # Constraint 6c: Each customer must be left exactly once
        for u in self.customers:
            self.model.addConstr(gp.quicksum(self.x[u,j] for i,j in self.E_star if i == u) == 1)
        
        # Constraint 6d: Capacity propagation
        for u in self.customers:
            for v,j in self.E_star:
                if v != u:
                    self.model.addConstr(
                        self.delta[j] - self.demands[j] >= 
                        self.delta[u] - (self.vehicle_capacity + self.demands[j]) * (1 - self.x[v,j])
                    )
        
        # Constraint 6e: Time window propagation
        for u in self.customers:
            for v,j in self.E_star:
                if v != u:
                    self.model.addConstr(
                        self.tau[j] - self.costs[v,j] >= 
                        self.tau[u] - (self.time_windows[u][1] + self.costs[v,j]) * (1 - self.x[v,j])
                    )
        
        # Constraint 6f: Minimum number of vehicles
        total_demand = sum(self.demands[u] for u in self.customers)
        min_vehicles = int(np.ceil(total_demand / self.vehicle_capacity))
        self.model.addConstr(
            gp.quicksum(self.x[self.depot_start,j] for i,j in self.E_star 
                       if i == self.depot_start) >= min_vehicles
        )
    
    def _add_la_arc_constraints(self):
        """Add constraints 6g-6i"""
        # Constraint 6g: Select exactly one ordering for each customer
        for u in self.customers:
            self.model.addConstr(
                gp.quicksum(self.y[u,r] for r in range(len(self.R_u[u]))) == 1
            )
        
        # Constraints 6h-6i: Consistency between x and y variables
        for u in self.customers:
            for w in self.local_area_neighbors[u] + [u]:
                for v in self.local_area_neighbors[u]:
                    if (w,v) in self.E_star:
                        # Sum up all orderings that use edge (w,v)
                        self.model.addConstr(
                            self.x[w,v] >= 
                            gp.quicksum(self.y[u,r] for r in range(len(self.R_u[u])) 
                                      if self.R_u[u][r]['a_wv'].get((w,v), 0) == 1)
                        )
                
                # Handle final customer in ordering
                if w in self.local_area_neighbors[u] + [u]:
                    self.model.addConstr(
                        gp.quicksum(self.x[w,v] for i,v in self.E_star if i == w and 
                                  v not in self.local_area_neighbors[u] + [u]) >= 
                        gp.quicksum(self.y[u,r] for r in range(len(self.R_u[u])) 
                                  if self.R_u[u][r]['a_star'].get(w, 0) == 1)
                    )
    
    def _add_capacity_flow_constraints(self):
        """Add constraints 6j-6k"""
        # Constraint 6j: Flow conservation for capacity
        for i, k, d_min, d_max in self.G_D:
            if i not in [self.depot_start, self.depot_end]:
                self.model.addConstr(
                    gp.quicksum(self.z_D[e] for e in self.E_D if e[0] == (i,k)) ==
                    gp.quicksum(self.z_D[e] for e in self.E_D if e[1] == (i,k))
                )
        
        # Constraint 6k: Consistency between x and z_D variables
        for i, j in self.E_star:
            self.model.addConstr(
                self.x[i,j] == 
                gp.quicksum(self.z_D[e] for e in self.E_D 
                           if e[0][0] == i and e[1][0] == j)
            )
    
    def _add_time_flow_constraints(self):
        """Add constraints 6l-6m"""
        # Constraint 6l: Flow conservation for time
        for i, k, t_min, t_max in self.G_T:
            if i not in [self.depot_start, self.depot_end]:
                self.model.addConstr(
                    gp.quicksum(self.z_T[e] for e in self.E_T if e[0] == (i,k)) ==
                    gp.quicksum(self.z_T[e] for e in self.E_T if e[1] == (i,k))
                )
        
        # Constraint 6m: Consistency between x and z_T variables
        for i, j in self.E_star:
            self.model.addConstr(
                self.x[i,j] == 
                gp.quicksum(self.z_T[e] for e in self.E_T 
                           if e[