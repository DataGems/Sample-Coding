import gurobipy as gp
from gurobipy import GRB

def test_dual_values():
    """
    Create a simple transportation problem where we know what the dual values should be
    """
    print("Testing dual value access with simple model...")
    
    # Create model
    m = gp.Model("test")
    
    # Disable presolve etc like in our main code
    m.setParam('Method', 1)        # Use dual simplex
    m.setParam('Presolve', 0)      # Disable presolve
    m.setParam('PreDual', 0)       # Preserve dual structure
    m.setParam('DualReductions', 0) # Prevent dual reductions
    
    # Create variables
    # Flow from 2 sources to 2 sinks
    x = {}
    for i in range(2):
        for j in range(2):
            x[i,j] = m.addVar(name=f'x_{i}_{j}')
    
    # Flow conservation constraints
    # Source 1 has supply of 10, Source 2 has supply of 20
    c1 = m.addConstr(x[0,0] + x[0,1] == 10, name='supply_0')
    c2 = m.addConstr(x[1,0] + x[1,1] == 20, name='supply_1')
    
    # Sink 1 has demand of 15, Sink 2 has demand of 15
    c3 = m.addConstr(x[0,0] + x[1,0] == 15, name='demand_0')
    c4 = m.addConstr(x[0,1] + x[1,1] == 15, name='demand_1')
    
    # Costs: prefer sending to matching index
    # i.e. source 0 prefers sink 0, source 1 prefers sink 1
    m.setObjective(x[0,0] + 2*x[0,1] + 2*x[1,0] + x[1,1], GRB.MINIMIZE)
    
    # Solve
    m.optimize()
    
    print("\nSolution:")
    for i in range(2):
        for j in range(2):
            print(f"x_{i}_{j} = {x[i,j].X}")
            
    print("\nDual values:")
    print(f"supply_0: {c1.Pi}")
    print(f"supply_1: {c2.Pi}")
    print(f"demand_0: {c3.Pi}")
    print(f"demand_1: {c4.Pi}")

if __name__ == "__main__":
    test_dual_values()