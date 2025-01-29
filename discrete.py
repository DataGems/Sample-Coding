import pulp
import numpy as np
import time
from collections import defaultdict

class VRPTWOptimizer:

    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                vehicle_capacity, K=2, time_granularity=3):
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
        
        # Create time discretization
        self.time_buckets = self._create_time_buckets(time_granularity)
        
        # Create time graph
        self.nodes_T, self.edges_T = self._create_time_graph()
        
        print(f"\nModel initialized with:")
        print(f"Number of customers: {len(customers)}")
        print(f"Number of edges: {len(self.E_star)}")
        print(f"Number of time nodes: {len(self.nodes_T)}")
        print(f"Number of time edges: {len(self.edges_T)}")
        
        self._create_variables()
        self._add_constraints()

    def _create_time_buckets(self, granularity):
        """Create time buckets for each customer based on their time windows"""
        buckets = {}
        for u in self.customers:
            earliest, latest = self.time_windows[u]
            time_span = latest - earliest
            bucket_size = time_span / granularity
            
            customer_buckets = []
            for i in range(granularity):
                lower = earliest + i * bucket_size
                upper = earliest + (i + 1) * bucket_size if i < granularity - 1 else latest
                customer_buckets.append((lower, upper))
                
            buckets[u] = customer_buckets
        return buckets

    def _create_time_graph(self):
        """Create directed graph GT for time flow"""
        nodes_T = []  # (u, k, t⁻, t⁺)
        edges_T = []  # ((u1,k1), (u2,k2))
        
        # Create nodes for each customer and bucket
        for u in self.customers:
            for k, (t_min, t_max) in enumerate(self.time_buckets[u]):
                nodes_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        nodes_T.append((self.depot_start, 0, self.time_windows[self.depot_start][0], 
                    self.time_windows[self.depot_start][1]))
        nodes_T.append((self.depot_end, 0, self.time_windows[self.depot_end][0], 
                    self.time_windows[self.depot_end][1]))
        
        # Create edges
        for i, k_i, t_min_i, t_max_i in nodes_T:
            for j, k_j, t_min_j, t_max_j in nodes_T:
                if i != j:
                    travel_time = self.costs[i,j] / 5  # Convert cost to time
                    if t_max_i + travel_time <= t_max_j:
                        edges_T.append(((i,k_i), (j,k_j)))
        
        return nodes_T, edges_T

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

    def _create_variables(self):
            """Create all decision variables including time flow variables"""
            # Route variables x_{ij}
            self.x = pulp.LpVariable.dicts("x", 
                                        self.E_star,
                                        cat='Binary')
            
            # Time flow variables z_T
            self.z_T = pulp.LpVariable.dicts("z_T",
                                            self.edges_T,
                                            lowBound=0)
            
             # Time flow variables z_T
            self.z_T = pulp.LpVariable.dicts("z_T",
                                    self.edges_T,
                                    lowBound=0)

    def _add_routing_constraints(self):
        """Add basic routing constraints"""
        # Objective function
        self.model += pulp.lpSum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star)
        
        # Visit each customer once
        for u in self.customers:
            # Arrival at customer
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == 1
            # Departure from customer
            self.model += pulp.lpSum(self.x[u,j] for i,j in self.E_star if i == u) == 1

        # Capacity constraints
        # Initialize depot
        self.model += pulp.lpSum(self.demands[i] * self.x[self.depot_start,i] 
                                for i,j in self.E_star if i == self.depot_start) <= self.vehicle_capacity
        
        # Flow balance
        for u in self.customers:
            # Flow entering equals flow leaving
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == \
                        pulp.lpSum(self.x[u,j] for i,j in self.E_star if i == u)

        # Subtour elimination
        # Use time windows as MTZ constraints
        M = max(tw[1] for tw in self.time_windows.values())  # Big-M value
        for (i,j) in self.E_star:
            if j != self.depot_end:  # Don't need time constraints for trips to end depot
                travel_time = self.costs[i,j] / 5
                self.model += self.tau[j] >= self.tau[i] + travel_time - M * (1 - self.x[i,j])
        
        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model += self.tau[i] >= self.time_windows[i][0]
            self.model += self.tau[i] <= self.time_windows[i][1]

        # Minimum number of vehicles based on total demand
        total_demand = sum(self.demands[u] for u in self.customers)
        min_vehicles = int(np.ceil(total_demand / self.vehicle_capacity))
        self.model += pulp.lpSum(self.x[self.depot_start,j] 
            for i,j in self.E_star if i == self.depot_start) >= min_vehicles
    
    def _add_constraints(self):
        """Add routing and time flow constraints"""
        self._add_routing_constraints()
        self._add_time_flow_constraints()

    def _add_time_flow_constraints(self):
        """Add time flow constraints (5a-5b)"""
        # Flow conservation (5a)
        for i, k, t_min, t_max in self.nodes_T:
            if i not in [self.depot_start, self.depot_end]:
                self.model += (
                    pulp.lpSum(self.z_T[e] for e in self.edges_T if e[0] == (i,k)) ==
                    pulp.lpSum(self.z_T[e] for e in self.edges_T if e[1] == (i,k))
                )
        
        # Consistency with route variables (5b)
        for u,v in self.E_star:
            self.model += (
                self.x[u,v] == pulp.lpSum(
                    self.z_T[e] for e in self.edges_T 
                    if e[0][0] == u and e[1][0] == v
                )
            )

    def solve(self, time_limit=None):
        """Solve the VRPTW instance with time bucket information"""
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
            
            print("\nTime Flow Variables:")
            for e in self.edges_T:
                val = pulp.value(self.z_T[e])
                if val is not None and val > 0.001:
                    print(f"z_T_{e} = {val:.3f}")
            
            solution['routes'] = self._extract_routes()
            solution['time_buckets'] = self._extract_time_buckets()
            
        return solution

    def _extract_time_buckets(self):
        """Extract active time buckets for each customer"""
        time_buckets = {}
        for u in self.customers:
            for k, (t_min, t_max) in enumerate(self.time_buckets[u]):
                node = (u, k)
                incoming_flow = sum(pulp.value(self.z_T[e]) or 0 
                                for e in self.edges_T if e[1] == node)
                if incoming_flow > 0.5:
                    time_buckets[u] = (t_min, t_max)
                    break
        return time_buckets

    def _extract_routes(self):
        """Extract routes from solution"""
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

def create_small_test_instance():
    """Create a very simple test instance"""
    locations = {
        0: (0, 0),    # Depot start
        1: (1, 1),    # Customer 1
        2: (-1, 1),   # Customer 2
        3: (1, -1),   # Customer 3
        4: (-1, -1),  # Customer 4
        5: (0, 0)     # Depot end
    }
    
    # Calculate costs - more carefully
    costs = {}
    nodes = list(range(6))  # 0 through 5
    for i in nodes:
        for j in nodes:
            if i != j:  # Skip self-loops
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                costs[i,j] = int(dist * 5)
                
    # Print costs for debugging
    print("\nCost matrix:")
    for i in nodes:
        for j in nodes:
            if i != j:
                print(f"Cost {i}->{j}: {costs.get((i,j), 'N/A')}")
    
    time_windows = {
        0: (0, 1000),    # Very wide depot window
        1: (0, 100),     # Very wide customer windows
        2: (0, 100),
        3: (0, 100),
        4: (0, 100),
        5: (0, 1000)     # Very wide depot window
    }
    
    demands = {
        0: 0,    # Depot
        1: 1,    # Very small demands
        2: 1,
        3: 1,
        4: 1,
        5: 0     # Depot
    }
    
    return {
        'customers': [1, 2, 3, 4],
        'depot_start': 0,
        'depot_end': 5,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 10
    }

def main():
    print("Creating test instance...")
    instance = create_small_test_instance()
    
    print("\nProblem characteristics:")
    print(f"Number of customers: {len(instance['customers'])}")
    print(f"Vehicle capacity: {instance['vehicle_capacity']}")
    print(f"Total demand: {sum(instance['demands'][i] for i in instance['customers'])}")
    
    print("\nCustomer Details:")
    for i in sorted(instance['customers']):
        print(f"Customer {i}: Window {instance['time_windows'][i]}, Demand: {instance['demands'][i]}")

    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        time_granularity=3
    )

    solution = optimizer.solve(time_limit=300)

    if solution['status'] == 'Optimal':
        print(f"\nOptimal Solution Cost: {solution['objective']:.2f}")
        print("\nRoutes:")
        total_cost = 0
        for idx, route in enumerate(solution['routes'], 1):
            # Skip empty routes
            if not route or all(r == instance['depot_end'] for r in route):
                continue
                
            route_demand = sum(instance['demands'][c] for c in route)
            route_with_depots = [instance['depot_start']] + route + [instance['depot_end']]
            route_cost = 0
            for i in range(len(route_with_depots) - 1):
                start = route_with_depots[i]
                end = route_with_depots[i+1]
                if start != end:
                    route_cost += instance['costs'][start, end]
            total_cost += route_cost
            
            print(f"\nRoute {idx}: {' -> '.join([str(instance['depot_start'])] + [str(c) for c in route] + [str(instance['depot_end'])])}")
            print(f"  Total demand: {route_demand}")
            print(f"  Schedule:")
            current_time = 0
            current_loc = instance['depot_start']
            for stop in route:
                travel_time = instance['costs'][current_loc, stop] / 5
                arrival_time = max(current_time + travel_time, instance['time_windows'][stop][0])
                time_bucket = solution['time_buckets'].get(stop)
                print(f"    Customer {stop}: Arrive at {arrival_time:.1f}")
                print(f"      Window: {instance['time_windows'][stop]}")
                print(f"      Time bucket: {time_bucket}")
                print(f"      Demand: {instance['demands'][stop]}")
                current_time = arrival_time
                current_loc = stop
            
        print(f"\nTotal Cost: {total_cost}")
    else:
        print(f"\nNo optimal solution found. Status: {solution['status']}")

if __name__ == "__main__":
    main()
