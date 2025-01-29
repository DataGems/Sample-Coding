!pip3 install pulp


import pulp
import numpy as np
import time
from collections import defaultdict

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, demands, 
                 vehicle_capacity, K=2):  # K is number of closest neighbors
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
        # Time window feasibility
        earliest_i, latest_i = self.time_windows[i]
        earliest_j, latest_j = self.time_windows[j]
        travel_time = self.costs[i,j] / 5  # Convert cost to time
        
        if earliest_i + travel_time > latest_j:
            return False
            
        # Capacity feasibility
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
        # Check capacity
        total_demand = sum(self.demands[i] for i in sequence)
        if total_demand > self.vehicle_capacity:
            return False
            
        # Check time windows
        current_time = 0
        for i in range(len(sequence)-1):
            current = sequence[i]
            next_customer = sequence[i+1]
            
            # Add travel time
            current_time = max(current_time + self.costs[current,next_customer]/5,
                             self.time_windows[next_customer][0])
            
            if current_time > self.time_windows[next_customer][1]:
                return False
        
        return True

    def _create_variables(self):
        """Create variables including y variables for LA-arcs"""
        # Original variables
        self.x = pulp.LpVariable.dicts("x", 
                                     self.E_star,
                                     cat='Binary')
        
        self.tau = pulp.LpVariable.dicts("tau",
                                        self.customers + [self.depot_start, self.depot_end],
                                        lowBound=0)
        
        self.delta = pulp.LpVariable.dicts("delta",
                                         self.customers + [self.depot_start, self.depot_end],
                                         lowBound=0,
                                         upBound=self.vehicle_capacity)
        
        # LA-arc variables
        self.y = {}
        for u in self.customers:
            for r in range(len(self.R_u[u])):
                self.y[u,r] = pulp.LpVariable(f"y_{u}_{r}", cat='Binary')

    def _add_constraints(self):
        """Add all constraints including LA-arc constraints"""
        # Original constraints
        self._add_original_constraints()
        
        # LA-arc constraints (2a-2c from page 6)
        self._add_la_arc_constraints()

    def _add_original_constraints(self):
        """Add original VRPTW constraints"""
        # [Previous constraints remain the same]
        # ... [Previous constraint code here]

    def _add_la_arc_constraints(self):
        """Add LA-arc movement consistency constraints (2a-2c)"""
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
                self.model += pulp.lpSum(self.x[w,j] for j in outside_neighbors 
                                       if (w,j) in self.E_star) >= \
                            pulp.lpSum(self.y[u,r] for r in range(len(self.R_u[u]))
                                     if self.R_u[u][r]['a_star'].get(w, 0) == 1)

    def solve(self, time_limit=None):
        """Solve the VRPTW instance"""
        # [Previous solve method remains the same]
        # ... [Previous solve code here]

# [Rest of the implementation remains the same]
```

Key additions:
1. LA neighborhood generation based on K closest reachable customers
2. Efficient ordering generation (R_u) for each customer
3. New y variables for LA-arc selection
4. LA-arc movement consistency constraints (2a-2c)

We can test this with a smaller K value first (like K=2) to see the impact on the solution. Would you like me to provide the complete implementation including the test instance and main function?