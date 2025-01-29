import pandas as pd
import numpy as np
# from core_milp import CoreMILP  # Save previous artifact as core_milp.py
import gurobipy as gp
from gurobipy import GRB

class CoreMILP:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, 
                 service_times, demands, vehicle_capacity):
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs  # Travel costs/times
        self.time_windows = time_windows
        self.service_times = service_times
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        
        # Initialize model
        self.model = gp.Model("VRPTW_LA_Discretization")
        
        # Create valid edges - equation (1)
        self.E_star = [(i,j) for i in [self.depot_start] + self.customers 
                       for j in self.customers + [self.depot_end] if i != j]

    def create_variables(self):
        """Create all decision variables for equation (6)"""
        # Route variables x_{ij} from original compact formulation
        self.x = self.model.addVars(self.E_star, vtype=GRB.BINARY, name="x")
        
        # Time variables τ_i from original compact formulation 
        self.tau = self.model.addVars(self.customers + [self.depot_start, self.depot_end], 
                                    lb=0.0, name="tau")
                                    
        # Capacity tracking variables
        self.load = self.model.addVars(self.customers, lb=0.0, 
                                     ub=self.vehicle_capacity, name="load")
        
        self.model.update()

    # In CoreMILP class, update the add_compact_constraints method:
    def add_compact_constraints(self):
        """Add original compact formulation constraints (1a-f)"""
        # Objective (1a)
        self.model.setObjective(
            gp.quicksum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star),
            GRB.MINIMIZE)

        # Visit each customer once (1b, 1c)
        for u in self.customers:
            # Exactly one incoming edge
            self.model.addConstr(
                gp.quicksum(self.x[i,u] for i,j in self.E_star if j == u) == 1,
                name=f'visit_in_{u}')
            # Exactly one outgoing edge  
            self.model.addConstr(
                gp.quicksum(self.x[u,j] for i,j in self.E_star if i == u) == 1,
                name=f'visit_out_{u}')

        # Load propagation (1d) - only between customers
        M = self.vehicle_capacity
        for u in self.customers:
            for v,j in self.E_star:
                if v == u and j in self.customers:  # Only propagate between customers
                    self.model.addConstr(
                        self.load[j] >= self.load[u] + self.demands[j] - 
                        M * (1 - self.x[u,j]),
                        name=f'load_prop_{u}_{j}')

        # Initial load constraints from depot
        for j in self.customers:
            self.model.addConstr(
                self.load[j] >= self.demands[j] * self.x[self.depot_start,j],
                name=f'init_load_{j}')

        # Time propagation (1e)
        M = max(tw[1] for tw in self.time_windows.values())
        for (i,j) in self.E_star:
            if j != self.depot_end:
                self.model.addConstr(
                    self.tau[j] >= self.tau[i] + self.service_times[i] + 
                    self.costs[i,j] - M * (1 - self.x[i,j]),
                    name=f'time_prop_{i}_{j}')

        # Minimum vehicle usage (1f)
        min_vehicles = sum(self.demands[u] for u in self.customers) / self.vehicle_capacity
        self.model.addConstr(
            gp.quicksum(self.x[self.depot_start,j] for i,j in self.E_star 
                       if i == self.depot_start) >= min_vehicles,
            name='min_vehicles')

        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model.addConstr(self.tau[i] >= self.time_windows[i][0], 
                               name=f'tw_lb_{i}')
            self.model.addConstr(self.tau[i] <= self.time_windows[i][1], 
                               name=f'tw_ub_{i}')

    def create_discretization_variables(self, time_buckets, capacity_buckets):
        """Create flow variables for time/capacity discretization"""
        # Time flow variables z^T (equations 4, 5)
        self.nodes_T, self.edges_T = self._create_time_graph(time_buckets)
        self.z_T = self.model.addVars(self.edges_T, lb=0.0, name="z_T")
        
        # Capacity flow variables z^D
        self.nodes_D, self.edges_D = self._create_capacity_graph(capacity_buckets)
        self.z_D = self.model.addVars(self.edges_D, lb=0.0, name="z_D")
        
        self.model.update()

    def add_discretization_constraints(self):
        """Add flow conservation constraints (4a,b and 5a,b)"""
        # Capacity flow conservation (4a)
        for i, k, d_min, d_max in self.nodes_D:
            if i not in [self.depot_start, self.depot_end]:
                self.model.addConstr(
                    gp.quicksum(self.z_D[e] for e in self.edges_D if e[0] == (i,k)) ==
                    gp.quicksum(self.z_D[e] for e in self.edges_D if e[1] == (i,k)),
                    name=f'cap_flow_cons_{i}_{k}')
        
        # Capacity route consistency (4b)
        for u,v in self.E_star:
            self.model.addConstr(
                self.x[u,v] == gp.quicksum(
                    self.z_D[e] for e in self.edges_D 
                    if e[0][0] == u and e[1][0] == v),
                name=f'cap_flow_cons_route_{u}_{v}')

        # Time flow conservation (5a)
        for i, k, t_min, t_max in self.nodes_T:
            if i not in [self.depot_start, self.depot_end]:
                self.model.addConstr(
                    gp.quicksum(self.z_T[e] for e in self.edges_T if e[0] == (i,k)) ==
                    gp.quicksum(self.z_T[e] for e in self.edges_T if e[1] == (i,k)),
                    name=f'time_flow_cons_{i}_{k}')
        
        # Time route consistency (5b)
        for u,v in self.E_star:
            self.model.addConstr(
                self.x[u,v] == gp.quicksum(
                    self.z_T[e] for e in self.edges_T 
                    if e[0][0] == u and e[1][0] == v),
                name=f'time_flow_cons_route_{u}_{v}')

    def _create_time_graph(self, time_buckets):
        """Create time discretization graph GT"""
        nodes_T = []  # (u, k, t⁻, t⁺)
        edges_T = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and their time buckets
        for u in self.customers:
            for k, (t_min, t_max) in enumerate(time_buckets[u]):
                nodes_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        depot_start_bucket = time_buckets[self.depot_start][0]
        depot_end_bucket = time_buckets[self.depot_end][0]
        nodes_T.append((self.depot_start, 0, depot_start_bucket[0], depot_start_bucket[1]))
        nodes_T.append((self.depot_end, 0, depot_end_bucket[0], depot_end_bucket[1]))
        
        # Create edges between nodes where travel is time-feasible
        for i, k_i, t_min_i, t_max_i in nodes_T:
            for j, k_j, t_min_j, t_max_j in nodes_T:
                if i != j:
                    # Check time feasibility
                    earliest_arrival = t_min_i + self.service_times[i] + self.costs[i,j]
                    latest_arrival = t_max_i + self.service_times[i] + self.costs[i,j]
                    
                    if (earliest_arrival <= t_max_j and 
                        latest_arrival >= t_min_j):
                        edges_T.append(((i,k_i), (j,k_j)))
        
        return nodes_T, edges_T

def create_time_buckets(time_windows, num_buckets=3):
    """Create time buckets that respect time window ordering"""
    buckets = {}
    
    # Get overall time horizon
    t_min = min(tw[0] for tw in time_windows.values())
    t_max = max(tw[1] for tw in time_windows.values())
    t_width = (t_max - t_min) / num_buckets
    
    for u, (window_min, window_max) in time_windows.items():
        # For each node, create buckets that intersect with its time window
        node_buckets = []
        for i in range(num_buckets):
            bucket_min = t_min + i * t_width
            bucket_max = t_min + (i + 1) * t_width
            
            # Only include bucket if it intersects with node's time window
            if bucket_max > window_min and bucket_min < window_max:
                actual_min = max(bucket_min, window_min)
                actual_max = min(bucket_max, window_max)
                node_buckets.append((actual_min, actual_max))
        
        buckets[u] = node_buckets
    
    return buckets

    def _create_capacity_graph(self, capacity_buckets):
        """Create capacity discretization graph GD"""
        nodes_D = []  # (u, k, d⁻, d⁺)
        edges_D = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and their capacity buckets
        for u in self.customers:
            for k, (d_min, d_max) in enumerate(capacity_buckets[u]):
                nodes_D.append((u, k, d_min, d_max))
        
        # Add depot nodes
        depot_start_bucket = capacity_buckets[self.depot_start][0]
        depot_end_bucket = capacity_buckets[self.depot_end][0]
        nodes_D.append((self.depot_start, 0, depot_start_bucket[0], depot_start_bucket[1]))
        nodes_D.append((self.depot_end, 0, depot_end_bucket[0], depot_end_bucket[1]))
        
        # Create edges between nodes where capacity transition is feasible
        for i, k_i, d_min_i, d_max_i in nodes_D:
            for j, k_j, d_min_j, d_max_j in nodes_D:
                if i != j:
                    # Check capacity feasibility
                    remaining_i = d_max_i - self.demands[i]
                    if (remaining_i >= d_min_j and 
                        d_min_j >= self.demands[j] and 
                        d_max_j <= self.vehicle_capacity):
                        edges_D.append(((i,k_i), (j,k_j)))
        
        return nodes_D, edges_D

    def optimize(self, time_limit=None):
        """Solve the MILP"""
        if time_limit:
            self.model.setParam('TimeLimit', time_limit)
        
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            return {
                'status': 'optimal',
                'objective': self.model.ObjVal,
                'routes': self._extract_routes(),
                'runtime': self.model.Runtime
            }
        elif self.model.Status == GRB.TIME_LIMIT:
            return {
                'status': 'time_limit',
                'objective': self.model.ObjVal if self.model.SolCount > 0 else None,
                'routes': self._extract_routes() if self.model.SolCount > 0 else None,
                'runtime': self.model.Runtime
            }
        else:
            return {
                'status': 'error',
                'runtime': self.model.Runtime
            }

    def _extract_routes(self):
        """Extract routes from solution"""
        if self.model.SolCount == 0:
            return None
            
        routes = []
        active_edges = [(i,j) for i,j in self.E_star 
                       if self.x[i,j].X > 0.5]
        
        # Start from depot
        starts = [(i,j) for i,j in active_edges if i == self.depot_start]
        
        for start in starts:
            route = []
            current = start[1]
            route.append(current)
            
            while current != self.depot_end:
                next_edges = [(i,j) for i,j in active_edges if i == current]
                if not next_edges:
                    break
                current = next_edges[0][1]
                if current != self.depot_end:
                    route.append(current)
            
            routes.append(route)
            
        return routes

def load_test_instance(filename, num_customers=5):
    """Load first n customers from Solomon instance"""
    df = pd.read_csv(filename)
    
    # Convert 'Depot' to 0 in CUST_NUM
    df['CUST_NUM'] = df['CUST_NUM'].replace('Depot', '0')
    df['CUST_NUM'] = df['CUST_NUM'].astype(int)
    
    # Select depot and first n customers
    selected_ids = [0] + list(range(1, num_customers + 1))
    df = df[df['CUST_NUM'].isin(selected_ids)]
    
    # Extract coordinates
    coords = {row['CUST_NUM']: (row['XCOORD.'], row['YCOORD.']) 
             for _, row in df.iterrows()}
    
    # Create customer list (excluding depot)
    customers = sorted(list(set(df['CUST_NUM']) - {0}))
    
    # Add virtual end depot
    virtual_end = max(customers) + 1
    coords[virtual_end] = coords[0]
    
    # Calculate distances
    costs = {}
    all_nodes = [0] + customers + [virtual_end]
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                costs[i,j] = np.floor(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 10) / 10
    
    # Extract time windows and service times
    time_windows = {row['CUST_NUM']: (row['READY_TIME'], row['DUE_DATE'])
                   for _, row in df.iterrows()}
    time_windows[virtual_end] = time_windows[0]  # End depot same as start
    
    service_times = {row['CUST_NUM']: row['SERVICE_TIME'] 
                    for _, row in df.iterrows()}
    service_times[virtual_end] = 0
    
    demands = {row['CUST_NUM']: row['DEMAND']
              for _, row in df.iterrows()}
    demands[virtual_end] = 0
    
    # Initial simple buckets for testing
    time_buckets = create_time_buckets(time_windows)
    capacity_buckets = {}
    
    # Create 3 equal-width buckets for each customer
    for u in customers + [0, virtual_end]:
        t_min, t_max = time_windows[u]
        t_width = (t_max - t_min) / 3
        time_buckets[u] = [
            (t_min, t_min + t_width),
            (t_min + t_width, t_min + 2*t_width),
            (t_min + 2*t_width, t_max)
        ]
        
        if u == 0 or u == virtual_end:
            d_max = 200  # Vehicle capacity
        else:
            d_max = 200 - demands[u]
        d_width = d_max / 3
        capacity_buckets[u] = [
            (0, d_width),
            (d_width, 2*d_width),
            (2*d_width, d_max)
        ]
    
    return {
        'customers': customers,
        'depot_start': 0,
        'depot_end': virtual_end,
        'costs': costs,
        'time_windows': time_windows,
        'service_times': service_times,
        'demands': demands,
        'vehicle_capacity': 200,
        'time_buckets': time_buckets,
        'capacity_buckets': capacity_buckets
    }

def main():
    # Load small test instance
    data = load_test_instance('r101.csv', num_customers=5)
    
    print("\nProblem Data:")
    print(f"Customers: {data['customers']}")
    print("\nTime Windows:")
    for i in data['customers'] + [data['depot_start'], data['depot_end']]:
        print(f"Node {i}: {data['time_windows'][i]}")
        
    # Print time bucket information
    print("\nTime Buckets:")
    for u in data['customers']:
        print(f"\nCustomer {u}:")
        for i, (t_min, t_max) in enumerate(data['time_buckets'][u]):
            print(f"  Bucket {i}: [{t_min:.1f}, {t_max:.1f}]")
    
    # Create and solve MILP
    milp = CoreMILP(
        customers=data['customers'],
        depot_start=data['depot_start'],
        depot_end=data['depot_end'],
        costs=data['costs'],
        time_windows=data['time_windows'],
        service_times=data['service_times'],
        demands=data['demands'],
        vehicle_capacity=data['vehicle_capacity']
    )
    
    print("\nTesting base model (no discretization)...")
    milp.create_variables()
    milp.add_compact_constraints()
    
    # Enable Gurobi infeasibility analysis
    milp.model.setParam('DualReductions', 0)
    milp.model.computeIIS()
    
    print("\nInfeasible constraints:")
    for c in milp.model.getConstrs():
        if c.IISConstr:
            print(f"{c.ConstrName}: {c.Sense} {c.RHS}")
    
    # Now try with discretization
    print("\nTesting with discretization...")
    milp.create_discretization_variables(
        data['time_buckets'],
        data['capacity_buckets']
    )
    milp.add_discretization_constraints()
    
    result = milp.optimize(time_limit=300)
    
    print("\nResults:")
    print(f"Status: {result['status']}")
    print(f"Runtime: {result['runtime']:.2f} seconds")
    if result['status'] in ['optimal', 'time_limit'] and result['routes']:
        print(f"Objective: {result['objective']:.2f}")
        print("Routes:")
        for i, route in enumerate(result['routes'], 1):
            print(f"Route {i}: {' -> '.join(str(x) for x in [0] + route + [data['depot_end']])}")

if __name__ == "__main__":
    main()