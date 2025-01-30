def solve_with_parsimony(self, time_limit=None):
    """Implement Algorithm 1 from paper"""
    if time_limit:
        self.model.setParam('TimeLimit', time_limit)
    
    self.model.setParam('MIPGap', 0.01)
    iter_since_reset = 0
    last_lp_val = float('-inf')
    
    while True:
        # Reset neighborhoods if needed
        if iter_since_reset >= self.sigma:
            print("\nResetting neighborhoods to maximum size...")
            self._reset_neighborhoods()
            iter_since_reset = 0
        
        # Solve LP relaxation 
        print("\nSolving LP relaxation...")
        self.model.optimize()
        current_obj = self.model.objVal
        print(f"LP objective: {current_obj}")
        
        # Get dual variables
        dual_vars = self._get_dual_variables()
        
        # If improved enough, do contraction operations
        if current_obj > last_lp_val + self.MIN_INC:
            print("\nSolution improved - attempting contractions...")
            changes = False
            changes |= self._merge_time_buckets(dual_vars)
            changes |= self._merge_capacity_buckets(dual_vars)
            changes |= self._contract_la_neighbors(dual_vars)
            
            if changes:
                last_lp_val = current_obj
                iter_since_reset = 0
                continue
        
        # Add new thresholds based on flow solution
        print("\nAdding new thresholds...")
        changes = self._add_flow_thresholds()
        if not changes:
            break
            
        iter_since_reset += 1
    
    # Final contraction
    self._merge_time_buckets(dual_vars)
    self._merge_capacity_buckets(dual_vars)
    self._contract_la_neighbors(dual_vars)
    
    # Solve final MILP
    print("\nSolving final MILP...")
    for v in self.model.getVars():
        if 'x_' in v.VarName:
            v.VType = GRB.BINARY
    self.model.optimize()
    
    return self._extract_solution()