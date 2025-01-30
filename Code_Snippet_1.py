def _merge_time_buckets(self, dual_vars):
    """Merge time buckets when their dual variables are equal (line 10 of Algorithm 1)"""
    changes_made = False
    for u in self.customers:
        print(f"\nTime buckets for customer {u} before merge:")
        print(self.T_u[u])
        
        buckets_to_merge = []
        # Get consecutive bucket pairs with equal duals
        for k in range(len(self.T_u[u]) - 1):
            dual_i = dual_vars.get(f'time_flow_cons_{u}_{k}', 0)
            dual_j = dual_vars.get(f'time_flow_cons_{u}_{k+1}', 0)
            
            if abs(dual_i - dual_j) < 1e-6:  # Numerical tolerance
                buckets_to_merge.append((k, k+1))
                changes_made = True
        
        # Merge buckets (work backwards to avoid index issues)
        for k1, k2 in reversed(buckets_to_merge):
            lower = self.T_u[u][k1][0]  # Lower bound of first bucket
            upper = self.T_u[u][k2][1]  # Upper bound of second bucket
            self.T_u[u].pop(k2)  # Remove second bucket
            self.T_u[u][k1] = (lower, upper)  # Update first bucket
            
        print("After merge:")
        print(self.T_u[u])
        
    return changes_made

def _merge_capacity_buckets(self, dual_vars):
    """Merge capacity buckets when their dual variables are equal (line 11 of Algorithm 1)"""
    changes_made = False
    for u in self.customers:
        print(f"\nCapacity buckets for customer {u} before merge:")
        print(self.D_u[u])
        
        buckets_to_merge = []
        # Get consecutive bucket pairs with equal duals 
        for k in range(len(self.D_u[u]) - 1):
            dual_i = dual_vars.get(f'cap_flow_cons_{u}_{k}', 0)
            dual_j = dual_vars.get(f'cap_flow_cons_{u}_{k+1}', 0)
            
            if abs(dual_i - dual_j) < 1e-6:  # Numerical tolerance
                buckets_to_merge.append((k, k+1))
                changes_made = True
                
        # Merge buckets (work backwards to avoid index issues)
        for k1, k2 in reversed(buckets_to_merge):
            lower = self.D_u[u][k1][0]  # Lower bound of first bucket
            upper = self.D_u[u][k2][1]  # Upper bound of second bucket
            self.D_u[u].pop(k2)  # Remove second bucket 
            self.D_u[u][k1] = (lower, upper)  # Update first bucket
            
        print("After merge:")
        print(self.D_u[u])
        
    return changes_made

def _contract_la_neighbors(self, dual_vars):
    """Contract LA-neighborhoods based on dual values (line 12 of Algorithm 1)"""
    changes_made = False
    for u in self.customers:
        # Get largest k with positive dual value per equation (9)
        max_k = 0
        for k in range(1, len(self.la_neighbors[u]) + 1):
            dual_sum = sum(dual_vars.get(f'la_arc_cons_{u}_w_v_{k}', 0) 
                          for w in self.la_neighbors[u] 
                          for v in self.la_neighbors[u] if v != w)
            if dual_sum > 0:
                max_k = k
        
        # Contract neighborhood if possible
        if max_k < len(self.la_neighbors[u]):
            print(f"Contracting neighborhood for customer {u} from {len(self.la_neighbors[u])} to {max_k}")
            self.la_neighbors[u] = self.la_neighbors[u][:max_k]
            changes_made = True
            
    return changes_made