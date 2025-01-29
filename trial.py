import pulp
import numpy as np
import time
from collections import defaultdict

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, K=2):
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
        
        print(f"\nNumber of edges: {len(self.E_star)}")
        print("Edges:", self.E_star)
        
        # Generate LA neighborhoods
        self.la_neighbors = self._generate_la_neighbors()
        print("\nLA Neighborhoods:")
        for u in self.customers:
            print(f"Customer {u}: {self.la_neighbors[u]}")
        
        # Generate orderings
        self.R_u = self._generate_orderings()
        print("\nOrderings for each customer:")
        for u in self.customers:
            print(f"Customer {u}: {len(self.R_u[u])} orderings")
            for idx, r in enumerate(self.R_u[u]):
                print(f"  {idx}: sequence={r['sequence']}")
        
        self._create_variables()
        self._add_constraints()

    def _generate_la_neighbors(self):
        """Generate K closest neighbors for each customer that are reachable"""
        la_neighbors = {}
        for u in self.customers:
            # Get distances to all other customers
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            # Take K closest that are reachable
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
        travel_time = self.costs[i,j] / 5  # Convert cost to time
        
        if earliest_i + travel_time > latest_j:
            return False
            
        if self.demands[i] + self.demands[j] > self.vehicle_capacity:
            return False
            
        return True

    def _generate_orderings(self):
        """Generate efficient orderings for each customer"""
        R_u = defaultdict(list)
        
        for u in self.customers:
            # Base ordering: just the customer itself
            R_u[u].append({
                'sequence': [u],
                'a_wv': {},
                'a_star': {u: 1}
            })
            
            # Add orderings with one neighbor
            for v in self.la_neighbors[u]:
                if self._is_reachable(u, v):
                    R_u[u].append({
                        'sequence': [u, v],
                        'a_wv': {(u,v): 1},
                        'a_star': {v: 1}
                    })
            
            # Add orderings with two neighbors (if feasible)
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
        
        # Capacity variables δ_i
        self.delta = pulp.LpVariable.dicts("delta",
                                         self.customers + [self.depot_start, self.depot_end],
                                         lowBound=0,
                                         upBound=self.vehicle_capacity)
        
        # LA-arc variables y_r
        self.y = {}
        for u in self.customers:
            for r in range(len(self.R_u[u])):
                self.y[u,r] = pulp.LpVariable(f"y_{u}_{r}", cat='Binary')

    def _add_constraints(self):
        """Add all constraints"""
        # Objective function
        self.model += pulp.lpSum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star)
        
        # Each customer must be visited exactly once
        for u in self.customers:
            self.model += pulp.lpSum(self.x[i,u] for i,j in self.E_star if j == u) == 1
            
        # Each customer must be left exactly once
        for u in self.customers:
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
            
        # Capacity constraints
        self.model += self.delta[self.depot_start] == self.vehicle_capacity
        self.model += self.delta[self.depot_end] >= 0
        
        for (i,j) in self.E_star:
            if j != self.depot_end:
                M = self.vehicle_capacity + max(self.demands.values())
                self.model += self.delta[j] <= self.delta[i] - self.demands.get(j, 0) + M * (1 - self.x[i,j])
                self.model += self.delta[j] >= self.demands.get(j, 0)
        
        # LA-arc constraints
        # Constraint (2a): Select exactly one ordering for each customer
        for u in self.customers:
            self.model += pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))) == 1
        
        # Constraints (2b): Consistency between x and y for LA-arc edges
        for u in self.customers:
            for w in [u] + self.la_neighbors[u]:
                for v in self.la_neighbors[u]:
                    if (w,v) in self.E_star:
                        self.model += self.x[w,v] >= pulp.lpSum(
                            self.y[u,r] for r in range(len(self.R_u[u]))
                            if self.R_u[u][r]['a_wv'].get((w,v), 0) == 1
                        )
        
        # Constraints (2c): Consistency for final customer in ordering
        for u in self.customers:
            for w in [u] + self.la_neighbors[u]:
                outside_neighbors = [j for j in self.customers + [self.depot_end] 
                                   if j not in self.la_neighbors[u] + [u]]
                if outside_neighbors:  # Only add constraint if there are outside neighbors
                    self.model += pulp.lpSum(self.x[w,j] for j in outside_neighbors 
                                           if (w,j) in self.E_star) >= \
                                pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))
                                         if self.R_u[u][r]['a_star'].get(w, 0) == 1)

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
                if i in self.delta:
                    delta_val = pulp.value(self.delta[i])
                    tau_val = pulp.value(self.tau[i])
                    if delta_val is not None and tau_val is not None:
                        print(f"Node {i} - Capacity: {delta_val:.2f}, Time: {tau_val:.2f}")
        
        return solution

    def _extract_routes(self):
        """Extract routes from solution"""
        print("\nDebug: Route Extraction")
        active_edges = [(i,j) for (i,j) in self.E_star 
                       if pulp.value(self.x[i,j]) is not None and pulp.value(self.x[i,j]) > 0.5]
        print("Active edges:", active_edges)
        
        routes = []
        current_route = []
        current = self.depot_start
        visited = set()
        
        while len(visited) < len(self.customers):
            found_next = False
            for j in self.customers + [self.depot_end]:
                if (current, j) in self.E_star and (current, j) in active_edges:
                    found_next = True
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
            
            if not found_next:
                if current_route:
                    routes.append(current_route)
                if len(visited) < len(self.customers):
                    current = self.depot_start
                    current_route = []
                else:
                    break
        
        if current_route:
            routes.append(current_route)
        
        return routes

def create_eight_customer_instance():
    """Create instance with 8 customers"""
    # Locations in a 6x6 grid
    locations = {
        0: (3, 3),     # Depot centrally located
        1: (1, 5),     # Customer 1 - northwest
        2: (5, 5),     # Customer 2 - northeast
        3: (0, 3),     # Customer 3 - west
        4: (6, 3),     # Customer 4 - east
        5: (2, 1),     # Customer 5 - southwest
        6: (4, 1),     # Customer 6 - southeast
        7: (3, 0),     # Customer 7 - south
        8: (3, 6),     # Customer 8 - north
        9: (3, 3)      # Depot end (same as start)
    }
    
    # Calculate costs based on Euclidean distance * 5
    costs = {}
    for i in locations:
        for j in locations:
            if i != j:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                costs[i,j] = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 5)
    
    # Time windows - morning and afternoon shifts
    time_windows = {
        0: (0, 200),     # Depot start
        1: (0, 80),      # Customer 1 - morning
        2: (90, 160),    # Customer 2 - afternoon
        3: (20, 90),     # Customer 3 - morning
        4: (70, 150),    # Customer 4 - mid-day
        5: (0, 70),      # Customer 5 - early morning
        6: (80, 140),    # Customer 6 - afternoon
        7: (40, 110),    # Customer 7 - mid-morning
        8: (110, 180),   # Customer 8 - late afternoon
        9: (0, 200)      # Depot end
    }
    
    # Demands - mix of small and large orders
    demands = {
        0: 0,     # Depot start
        1: 4,     # Customer 1 - small
        2: 8,     # Customer 2 - large
        3: 3,     # Customer 3 - small
        4: 7,     # Customer 4 - medium
        5: 5,     # Customer 5 - medium
        6: 6,     # Customer 6 - medium
        7: 4,     # Customer 7 - small
        8: 7,     # Customer 8 - medium
        9: 0      # Depot end
    }
    
    return {
        'customers': list(range(1, 9)),
        'depot_start': 0,
        'depot_end': 9,
        'costs': costs,
        'time_windows': time_windows,
        'demands': demands,
        'vehicle_capacity': 18,
        'locations': locations
    }

def main():
    
    if solution['status'] == 'Optimal':
        print(f"Optimal Solution Cost: {solution['objective']:.2f}")
        
        # Extract routes from active edges more robustly
        active_edges = [(i,j) for (i,j) in optimizer.E_star 
                       if pulp.value(optimizer.x[i,j]) is not None 
                       and pulp.value(optimizer.x[i,j]) > 0.5]
        print("\nActive edges:", active_edges)
        
        # Find routes by starting from depot (0)
        routes = []
        depot_starts = [(i,j) for (i,j) in active_edges if i == optimizer.depot_start]
        
        for start_edge in depot_starts:
            route = []
            current = start_edge[1]  # First customer after depot
            route.append(current)
            
            while current != optimizer.depot_end:
                # Find next edge
                next_edges = [(i,j) for (i,j) in active_edges if i == current]
                if not next_edges:
                    break
                current = next_edges[0][1]
                if current != optimizer.depot_end:
                    route.append(current)
            
            routes.append(route)
        
        print("\nExtracted Routes:")
        total_cost = 0
        for idx, route in enumerate(routes, 1):
            route_demand = sum(instance['demands'][c] for c in route)
            route_cost = sum(instance['costs'][i,j] for i, j in zip([0] + route, route + [9]))
            total_cost += route_cost
            
            print(f"\nRoute {idx}: {' -> '.join(['0'] + [str(c) for c in route] + ['9'])}")
            print(f"  Total demand: {route_demand}")
            print(f"  Route cost: {route_cost}")
            print(f"  Schedule:")
            
            current_time = 0
            current_loc = optimizer.depot_start
            for stop in route:
                travel_time = instance['costs'][current_loc, stop] / 5  # Convert cost to time
                arrival_time = max(current_time + travel_time, instance['time_windows'][stop][0])
                print(f"    Customer {stop}: Arrive at {arrival_time:.1f} "
                      f"(Window: {instance['time_windows'][stop]}, "
                      f"Demand: {instance['demands'][stop]})")
                current_time = arrival_time
                current_loc = stop
        
        print(f"\nTotal Solution Cost: {total_cost}")
        print(f"Number of Routes: {len(routes)}")

if __name__ == "__main__":
    main()
    