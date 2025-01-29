import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import defaultdict
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import itertools
import logging

# Configure logging
def setup_logger():
    # Create logger
    logger = logging.getLogger('la_arc_generator')
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler with formatter
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)  # Allow all messages through handler
    logger.addHandler(ch)
    
    return logger

# Create logger instance
logger = setup_logger()

# Add this at the beginning of Better_LA_Routes.py, after the imports
# but before the LASequence class

class TimeWindowHandler:
    def __init__(self, time_windows, service_times, travel_times):
        """
        Initialize with problem data
        
        Args:
            time_windows: Dict[customer_id, Tuple[earliest, latest]]
            service_times: Dict[customer_id, float]
            travel_times: Dict[Tuple[from_id, to_id], float]
        """
        self.time_windows = time_windows
        self.service_times = service_times
        self.travel_times = travel_times

    def phi_r(self, sequence):
        """
        Calculate earliest possible departure time from first customer without waiting
        Implements equation (13a,13c) from paper
        
        Args:
            sequence: List[int] of customer IDs in visit order
            
        Returns:
            float: Earliest feasible departure time or float('-inf') if infeasible
        """
        if not sequence:
            return float('-inf')
            
        # Base case: single customer
        if len(sequence) == 1:
            u = sequence[0]
            return self.time_windows[u][0]
            
        # Base case: two customers u,v 
        if len(sequence) == 2:
            u, v = sequence
            earliest_u = self.time_windows[u][0]
            earliest_v = self.time_windows[v][0]
            latest_v = self.time_windows[v][1]
            
            # Calculate arrival at v if departing u at earliest_u
            departure_u = earliest_u 
            arrival_v = (departure_u + 
                        self.service_times[u] + 
                        self.travel_times[(u,v)])
            
            # Check feasibility
            if arrival_v > latest_v:
                return float('-inf')
            
            # If we arrive before v's earliest time, need earlier departure
            if arrival_v < earliest_v:
                # Work backwards to find needed departure time
                needed_departure = (earliest_v - 
                                  self.service_times[u] - 
                                  self.travel_times[(u,v)])
                return max(earliest_u, needed_departure)
                
            return earliest_u

        # Recursive case: Check φ_r for subpath
        subpath_phi = self.phi_r(sequence[1:])
        if subpath_phi == float('-inf'):
            return float('-inf')
            
        # Calculate earliest arrival at second customer
        u = sequence[0]
        v = sequence[1]
        earliest_u = self.time_windows[u][0]
        latest_v = self.time_windows[v][1]
        
        departure_u = earliest_u
        arrival_v = (departure_u + 
                    self.service_times[u] + 
                    self.travel_times[(u,v)])
        
        # Check feasibility
        if arrival_v > latest_v:
            return float('-inf')
            
        # If arrival too early for subpath, need to delay
        if arrival_v < subpath_phi:
            needed_departure = (subpath_phi - 
                              self.service_times[u] - 
                              self.travel_times[(u,v)])
            return max(earliest_u, needed_departure)
            
        return earliest_u

    def phi_hat_r(self, sequence):
        """
        Calculate latest possible departure time from first customer
        Implements equation (13b,13d) from paper
        
        Args:
            sequence: List[int] of customer IDs in visit order
            
        Returns:
            float: Latest feasible departure time or float('-inf') if infeasible
        """
        if not sequence:
            return float('-inf')
            
        # Base case: single customer
        if len(sequence) == 1:
            u = sequence[0]
            return self.time_windows[u][1]
            
        # Base case: two customers u,v
        if len(sequence) == 2:
            u, v = sequence
            earliest_u = self.time_windows[u][0]
            latest_u = self.time_windows[u][1]
            latest_v = self.time_windows[v][1]
            
            # Latest possible departure from u to reach v by deadline
            latest_departure = (latest_v - 
                              self.service_times[u] - 
                              self.travel_times[(u,v)])
            
            # Check feasibility
            if latest_departure < earliest_u:
                return float('-inf')
                
            return min(latest_u, latest_departure)
            
        # Recursive case: Check φ̂_r for subpath
        subpath_phi_hat = self.phi_hat_r(sequence[1:])
        if subpath_phi_hat == float('-inf'):
            return float('-inf')
            
        # Calculate latest possible departure
        u = sequence[0]
        earliest_u = self.time_windows[u][0]
        latest_u = self.time_windows[u][1]
        
        # Need to arrive at second customer by its φ̂_r time
        v = sequence[1]
        latest_departure = (subpath_phi_hat - 
                          self.service_times[u] - 
                          self.travel_times[(u,v)])
        
        # Check feasibility
        if latest_departure < earliest_u:
            return float('-inf')
            
        return min(latest_u, latest_departure)

    def is_time_feasible(self, sequence):
        """
        Check if sequence satisfies time window constraints
        
        Args:
            sequence: List[int] of customer IDs in visit order
            
        Returns:
            bool: True if sequence is time feasible
        """
        # Empty sequence is feasible
        if not sequence:
            return True
            
        # Calculate φ_r and φ̂_r
        phi = self.phi_r(sequence)
        phi_hat = self.phi_hat_r(sequence)
        
        # Sequence is feasible if and only if φ_r ≤ φ̂_r
        if phi == float('-inf') or phi_hat == float('-inf'):
            return False
            
        return phi <= phi_hat

    def get_time_bounds(self, sequence):
        """
        Get feasible departure time bounds for sequence
        
        Args:
            sequence: List[int] of customer IDs in visit order
            
        Returns:
            Tuple[float, float]: (earliest, latest) feasible departure times
            Returns (None, None) if sequence is infeasible
        """
        phi = self.phi_r(sequence)
        phi_hat = self.phi_hat_r(sequence)
        
        if phi == float('-inf') or phi_hat == float('-inf') or phi > phi_hat:
            return None, None
            
        return phi, phi_hat

    def validate_timing(self, sequence, start_time):
        """
        Validate exact timing of sequence given start time
        
        Args:
            sequence: List[int] of customer IDs
            start_time: float time to start sequence
            
        Returns:
            List[Dict]: Timing details for each stop including:
                - arrival_time: When vehicle arrives
                - service_start: When service begins
                - departure_time: When vehicle departs
            Returns None if timing is infeasible
        """
        timing = []
        current_time = start_time
        
        for i in range(len(sequence)):
            current = sequence[i]
            arrival_time = current_time
            
            # Check arrival vs time window
            if arrival_time > self.time_windows[current][1]:
                return None
                
            # Service starts at max of arrival and earliest allowed time
            service_start = max(arrival_time, self.time_windows[current][0])
            departure_time = service_start + self.service_times[current]
            
            timing.append({
                'customer': current,
                'arrival_time': arrival_time,
                'service_start': service_start, 
                'departure_time': departure_time
            })
            
            # Set up for next customer if any
            if i < len(sequence)-1:
                next_customer = sequence[i+1]
                current_time = departure_time + self.travel_times[(current,next_customer)]
                
        return timing

class LASequence:
    def __init__(self, start_customer, intermediate_stops, end_customer, costs, time_windows, service_times):
        self.start = start_customer
        self.intermediate = intermediate_stops
        self.end = end_customer
        self.sequence = [start_customer] + intermediate_stops + [end_customer]
        
        self.costs = costs
        self.time_windows = time_windows
        self.service_times = service_times
        self.time_handler = TimeWindowHandler(time_windows, service_times, costs)
        
        self.total_distance = self._calculate_distance()

        bounds = self.time_handler.get_time_bounds(self.sequence)
        if bounds[0] is not None and bounds[0] <= bounds[1]:  # Add this check
            self.earliest_completion = self._get_completion_time(bounds[0])
            self.latest_start = bounds[1]
        
        # Calculate time bounds at initialization
        self.earliest_completion = None
        self.latest_start = None
        bounds = self.time_handler.get_time_bounds(self.sequence)
        if bounds[0] is not None:
            self.earliest_completion = self._get_completion_time(bounds[0])
            self.latest_start = bounds[1]
        
        logger.debug(f"Created sequence {self.sequence} with distance {self.total_distance:.2f}")
        
    def __str__(self):
        return (f"Sequence {self.sequence}: "
                f"distance={self.total_distance:.2f}, "
                f"start_window={self.time_windows[self.start]}")
    
    def _calculate_distance(self):
        total = 0
        for i in range(len(self.sequence)-1):
            total += self.costs[self.sequence[i], self.sequence[i+1]]
        return total
        
    def _get_completion_time(self, start_time):
        """Calculate completion time given a start time"""
        timing = self.time_handler.validate_timing(self.sequence, start_time)
        if timing:
            return timing[-1]['departure_time']
        return None
        
    def is_elementary(self):
        """Check if sequence uses each customer at most once"""
        is_elem = len(self.sequence) == len(set(self.sequence))
        logger.debug(f"Sequence {self.sequence} elementary check: {is_elem}")
        return is_elem
        
    def is_capacity_feasible(self, demands, vehicle_capacity):
        """Check if sequence satisfies capacity constraints"""
        total_demand = sum(demands[i] for i in self.sequence[:-1])
        is_feasible = total_demand <= vehicle_capacity
        logger.debug(f"Sequence {self.sequence} capacity check: {is_feasible} "
                    f"(total={total_demand}, capacity={vehicle_capacity})")
        return is_feasible
        
    def is_time_feasible(self):
        phi = self.time_handler.phi_r(self.sequence)
        phi_hat = self.time_handler.phi_hat_r(self.sequence)
        return phi != float('-inf') and phi_hat != float('-inf') and phi <= phi_hat

def generate_la_arcs(customers, neighbor_sets, costs, time_windows, service_times, demands, vehicle_capacity):
    """
    Generate LA-arcs using efficient frontier filtering
    
    Args:
        customers: List of customer IDs
        neighbor_sets: Dict mapping customer -> list of neighbor IDs
        costs: Dict mapping (i,j) -> travel cost/time
        time_windows: Dict mapping customer -> (earliest, latest) time
        service_times: Dict mapping customer -> service duration
        demands: Dict mapping customer -> demand
        vehicle_capacity: Vehicle capacity
        
    Returns:
        Dict mapping customer -> list of efficient LASequence objects
    """
    def is_dominated(seq1, seq2):
        """
        Check if seq1 is dominated by seq2.
        A sequence is dominated if another sequence is:
        - Not worse in any dimension (cost, earliest completion, latest start)
        - Strictly better in at least one dimension
        """
        EPSILON = 1e-6  # Numerical tolerance
        
        # If either sequence is time infeasible, can't establish dominance
        if seq1.earliest_completion is None or seq2.earliest_completion is None:
            return False
        
        # Cost comparison (lower is better)
        cost_better = seq2.total_distance < seq1.total_distance - EPSILON
        cost_equal = abs(seq2.total_distance - seq1.total_distance) <= EPSILON
        
        # Earliest completion comparison (earlier is better)
        completion_better = seq2.earliest_completion < seq1.earliest_completion - EPSILON
        completion_equal = abs(seq2.earliest_completion - seq1.earliest_completion) <= EPSILON
        
        # Latest start comparison (later is better)
        start_better = seq2.latest_start > seq1.latest_start + EPSILON
        start_equal = abs(seq2.latest_start - seq1.latest_start) <= EPSILON
        
        # Must be strictly better in at least one dimension
        has_improvement = (cost_better or completion_better or start_better)
        
        # Must not be worse in any dimension
        not_worse = (cost_better or cost_equal) and \
                   (completion_better or completion_equal) and \
                   (start_better or start_equal)
                    
        return has_improvement and not_worse

    def is_capacity_feasible(sequence):
        """Check if sequence satisfies capacity constraints"""
        total_demand = sum(demands[i] for i in sequence.sequence[:-1])
        return total_demand <= vehicle_capacity

    def generate_sequences(start_customer, available_neighbors, end_customer):
        """Generate all possible sequences from start to end using neighbors"""
        if not available_neighbors:
            # Base case - direct path to end customer
            sequence = LASequence(start_customer, [], end_customer,
                                costs, time_windows, service_times)
            if (is_capacity_feasible(sequence) and sequence.is_time_feasible()):
                return [sequence]
            return []
            
        sequences = []
        
        # Try direct path to end
        direct = LASequence(start_customer, [], end_customer,
                          costs, time_windows, service_times)
        if (is_capacity_feasible(direct) and direct.is_time_feasible()):
            sequences.append(direct)
            
        # Try paths through intermediate neighbors
        for length in range(1, len(available_neighbors) + 1):
            for intermediate_set in itertools.combinations(available_neighbors, length):
                for intermediate_perm in itertools.permutations(intermediate_set):
                    sequence = LASequence(start_customer, list(intermediate_perm), 
                                        end_customer, costs, time_windows, service_times)
                    if (is_capacity_feasible(sequence) and sequence.is_time_feasible()):
                        sequences.append(sequence)
                        
        return sequences

    # Main algorithm
    la_arcs = {}
    
    for u in customers:
        logger.info(f"\nGenerating LA-arcs for customer {u}")
        neighbors = neighbor_sets[u]
        non_neighbors = [c for c in customers if c not in neighbors and c != u]
        
        efficient_sequences = []
        
        # Generate sequences to each non-neighbor
        for v in non_neighbors:
            sequences = generate_sequences(u, neighbors, v)
            
            # Filter sequences using efficient frontier
            for seq in sequences:
                logger.debug(f"\nChecking sequence: {seq.sequence}")
                logger.debug(f"Distance: {seq.total_distance:.2f}")
                if seq.earliest_completion is not None:
                    logger.debug(f"Earliest completion: {seq.earliest_completion:.2f}")
                    logger.debug(f"Latest start: {seq.latest_start:.2f}")
                
                # Check if new sequence is dominated by any existing sequence
                is_dominated_by_existing = False
                for existing in efficient_sequences:
                    if is_dominated(seq, existing):
                        logger.debug(f"Dominated by {existing.sequence}")
                        logger.debug(f"  Existing - dist:{existing.total_distance:.2f}, " + \
                                   f"compl:{existing.earliest_completion:.2f}, " + \
                                   f"start:{existing.latest_start:.2f}")
                        is_dominated_by_existing = True
                        break
                        
                if not is_dominated_by_existing:
                    # Remove any existing sequences dominated by new sequence
                    efficient_sequences = [s for s in efficient_sequences 
                                        if not is_dominated(s, seq)]
                    # Add new sequence
                    efficient_sequences.append(seq)
                    logger.debug("Added to efficient sequences")
        
        logger.info(f"Found {len(efficient_sequences)} efficient sequences")
        la_arcs[u] = efficient_sequences
        
    return la_arcs

class TimeCalculator:
    def __init__(self, time_windows, service_times, travel_times):
        self.time_windows = time_windows  # Dict customer -> (earliest, latest)
        self.service_times = service_times  # Dict customer -> service_time
        self.travel_times = travel_times  # Dict (from,to) -> travel_time
        
    def phi_r(self, sequence: List[int]) -> float:
        """
        Calculate earliest possible departure time from first customer without waiting.
        This implements equation (13a,13c) from the paper.
        
        Returns:
            Earliest possible departure time from first customer
            or float('-inf') if sequence is time-infeasible
        """
        if not sequence:
            return float('-inf')
            
        # Base case: single customer
        if len(sequence) == 1:
            return self.time_windows[sequence[0]][0]  # Return earliest time
            
        # Base case: two customers i,j
        if len(sequence) == 2:
            i, j = sequence
            earliest_i = self.time_windows[i][0]
            earliest_j = self.time_windows[j][0]
            
            # Time arriving at j if we start at i's earliest time
            arrival_j = earliest_i + self.service_times[i] + self.travel_times[i,j]
            
            # If we can't arrive by j's deadline, sequence is infeasible
            if arrival_j > self.time_windows[j][1]:
                return float('-inf')
                
            # If we arrive before j's earliest time, sequence is feasible but needs waiting
            if arrival_j < earliest_j:
                needed_start = earliest_j - (self.service_times[i] + self.travel_times[i,j])
                return max(earliest_i, needed_start)
                
            # Otherwise we can proceed directly
            return earliest_i
            
        # Recursive case
        # First check if subpath is feasible
        subpath_phi = self.phi_r(sequence[1:])
        if subpath_phi == float('-inf'):
            return float('-inf')
            
        # Calculate earliest we could arrive at second customer
        i = sequence[0]
        j = sequence[1]
        earliest_i = self.time_windows[i][0]
        arrival_j = earliest_i + self.service_times[i] + self.travel_times[i,j]
        
        # If we can't arrive by j's deadline, sequence is infeasible
        if arrival_j > self.time_windows[j][1]:
            return float('-inf')
            
        # If we arrive before when subpath needs to start, we'll need waiting
        if arrival_j < subpath_phi:
            needed_start = subpath_phi - (self.service_times[i] + self.travel_times[i,j])
            return max(earliest_i, needed_start)
            
        # Otherwise we can proceed directly
        return earliest_i

    def phi_hat_r(self, sequence: List[int]) -> float:
        """
        Calculate latest possible departure time from first customer.
        This implements equation (13b,13d) from the paper.
        
        Returns:
            Latest possible departure time from first customer
            or float('-inf') if sequence is time-infeasible
        """
        if not sequence:
            return float('-inf')
            
        # Base case: single customer    
        if len(sequence) == 1:
            return self.time_windows[sequence[0]][1]  # Return latest time
            
        # Base case: two customers i,j
        if len(sequence) == 2:
            i, j = sequence
            latest_i = self.time_windows[i][1]
            latest_j = self.time_windows[j][1]
            
            # Latest possible departure from i to reach j by its deadline
            latest_departure = latest_j - (self.service_times[i] + self.travel_times[i,j])
            
            # If we can't start after i's earliest time, sequence is infeasible
            if latest_departure < self.time_windows[i][0]:
                return float('-inf')
                
            # Return min of latest_departure and i's deadline
            return min(latest_i, latest_departure)
            
        # Recursive case
        # First check if subpath is feasible
        subpath_phi_hat = self.phi_hat_r(sequence[1:])
        if subpath_phi_hat == float('-inf'):
            return float('-inf')
            
        # Calculate latest we could start at first customer
        i = sequence[0]
        latest_i = self.time_windows[i][1]
        
        # We need to arrive at second customer by its phi_hat time
        j = sequence[1]
        latest_departure = subpath_phi_hat - (self.service_times[i] + self.travel_times[i,j])
        
        # If we can't start after i's earliest time, sequence is infeasible
        if latest_departure < self.time_windows[i][0]:
            return float('-inf')
            
        # Return min of latest_departure and i's deadline
        return min(latest_i, latest_departure)

def test_calculator():
    """Test the time calculator with example data"""
    # Test data
    time_windows = {
        1: (0, 10),   # Customer 1: must start between 0-10
        2: (5, 15),   # Customer 2: must start between 5-15
        3: (10, 20),  # Customer 3: must start between 10-20
        4: (15, 25),  # Customer 4: must start between 15-25
        5: (20, 30)   # Customer 5: must start between 20-30
    }
    
    service_times = {i: 2 for i in range(1,6)}  # 2 time units per customer
    
    # Define travel times for our grid
    travel_times = {
        (1,2): 1.0, (1,3): 1.41, (1,4): 2.24, (1,5): 2.0,
        (2,1): 1.0, (2,3): 1.0,  (2,4): 2.0,  (2,5): 2.24,
        (3,1): 1.41, (3,2): 1.0,  (3,4): 1.0,  (3,5): 1.41,
        (4,1): 2.24, (4,2): 2.0,  (4,3): 1.0,  (4,5): 1.0,
        (5,1): 2.0,  (5,2): 2.24, (5,3): 1.41, (5,4): 1.0
    }
    
    calculator = TimeCalculator(time_windows, service_times, travel_times)
    
    # Test sequences we know should work from our previous tests
    test_sequences = [
        [1],      # Single customer
        [1, 4],   # Direct to non-neighbor
        [1, 2, 4] # Path through neighbor
    ]
    
    print("\nTesting Time Calculator:")
    print("-" * 50)
    
    for sequence in test_sequences:
        print(f"\nTesting sequence: {sequence}")
        phi = calculator.phi_r(sequence)
        phi_hat = calculator.phi_hat_r(sequence)
        
        print(f"φ_r   = {phi:.1f}")
        print(f"φ̂_r   = {phi_hat:.1f}")
        
        if phi <= phi_hat:
            print("✓ φ_r ≤ φ̂_r verified")
        else:
            print("❌ Error: φ_r > φ̂_r")
            
        # Verify sequence is actually feasible by checking each step
        current_time = phi
        is_feasible = True
        for i in range(len(sequence)-1):
            current = sequence[i]
            next_customer = sequence[i+1]
            
            # Add service time
            current_time += service_times[current]
            # Add travel time
            current_time += travel_times[(current,next_customer)]
            
            # Check if we arrive in time
            if current_time > time_windows[next_customer][1]:
                print(f"❌ Infeasible: Arrive at {next_customer} at {current_time:.1f}, deadline is {time_windows[next_customer][1]}")
                is_feasible = False
                break
                
            # Update to start of service
            current_time = max(current_time, time_windows[next_customer][0])
            
        if is_feasible:
            print("✓ Time feasibility verified")
            
@dataclass
class Customer:
    id: int
    earliest: float  # t⁺
    latest: float    # t⁻
    service_time: float

@dataclass
class Route:
    sequence: List[int]  # Customer IDs in order
    travel_times: Dict[Tuple[int, int], float]  # (i,j) -> travel_time

class PhiCalculator:
    def __init__(self, customers: Dict[int, Customer], travel_times: Dict[Tuple[int, int], float]):
        self.customers = customers
        self.travel_times = travel_times
    
    def phi_r(self, route: Route) -> float:
        """
        Calculate earliest possible departure time from first customer without waiting at any customer.
        This implements equation (13a,13c) from the paper.
        """
        if len(route.sequence) == 1:
            return self.customers[route.sequence[0]].earliest
        
        # Base case: route with 2 customers
        if len(route.sequence) == 2:
            u, v = route.sequence
            travel_time = self.travel_times.get((u,v))
            if travel_time is None:
                raise ValueError(f"No travel time defined for {u}->{v}")
            
            # Equation (13a): min(t⁺_u, t⁺_v + t_uv)
            return min(
                self.customers[u].earliest,
                self.customers[v].earliest + travel_time + self.customers[u].service_time
            )
        
        # Recursive case: longer route
        # Get recursive result for route minus first customer
        subroute = Route(
            sequence=route.sequence[1:],
            travel_times=route.travel_times
        )
        phi_minus = self.phi_r(subroute)
        
        # Equation (13c): min(t⁺_u, φ_r⁻ + t_uw)
        u = route.sequence[0]
        w = route.sequence[1]
        travel_time = self.travel_times.get((u,w))
        if travel_time is None:
            raise ValueError(f"No travel time defined for {u}->{w}")
            
        return min(
            self.customers[u].earliest,
            phi_minus + travel_time + self.customers[u].service_time
        )
    
    def phi_hat_r(self, route: Route) -> float:
        """
        Calculate latest possible departure time from first customer.
        This implements equation (13b,13d) from the paper.
        """
        if len(route.sequence) == 1:
            return self.customers[route.sequence[0]].latest
            
        # Base case: route with 2 customers
        if len(route.sequence) == 2:
            u, v = route.sequence
            travel_time = self.travel_times.get((u,v))
            if travel_time is None:
                raise ValueError(f"No travel time defined for {u}->{v}")
            
            # Equation (13b): max(t⁻_u, t⁻_v + t_uv)
            return max(
                self.customers[u].latest,
                self.customers[v].latest + travel_time + self.customers[u].service_time
            )
        
        # Recursive case: longer route
        subroute = Route(
            sequence=route.sequence[1:],
            travel_times=route.travel_times
        )
        phi_hat_minus = self.phi_hat_r(subroute)
        
        # Equation (13d): max(t⁻_u, φ̂_r⁻ + t_uw)
        u = route.sequence[0]
        w = route.sequence[1]
        travel_time = self.travel_times.get((u,w))
        if travel_time is None:
            raise ValueError(f"No travel time defined for {u}->{w}")
            
        return max(
            self.customers[u].latest,
            phi_hat_minus + travel_time + self.customers[u].service_time
        )

class VRPTWOptimizer:
    def __init__(self, customers, depot_start, depot_end, costs, time_windows, service_times, demands, 
                 vehicle_capacity, K=3, time_granularity=3, capacity_granularity=3, max_iterations=5):
        self.customers = customers
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.costs = costs  # Just distances, no service times
        self.time_windows = time_windows
        self.service_times = service_times  # Added service times
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.K = K
        self.time_granularity = time_granularity
        self.capacity_granularity = capacity_granularity
        self.max_iterations = max_iterations
        
        self.model = None
        self.create_initial_model()
    
    def create_initial_model(self):
        """Create initial model with all components"""
        self.model = gp.Model("VRPTW")
        
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
        self.x = {}
        for i, j in self.E_star:
            self.x[i,j] = self.model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        # Time variables τ_i
        self.tau = {}
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.tau[i] = self.model.addVar(lb=0, name=f'tau_{i}')
        
        # LA-arc variables y_r
        self.y = {}
        for u in self.customers:
            for r in range(len(self.R_u[u])):
                self.y[u,r] = self.model.addVar(vtype=GRB.BINARY, name=f'y_{u}_{r}')
        
        # Time flow variables z_T
        self.z_T = {}
        for edge in self.edges_T:
            self.z_T[edge] = self.model.addVar(lb=0, name=f'z_T_{edge}')

        # Capacity flow variables z_D
        self.z_D = {}
        for edge in self.edges_D:
            self.z_D[edge] = self.model.addVar(lb=0, name=f'z_D_{edge}')
                                        
        # Load tracking variables
        self.load = {}
        for i in self.customers:
            self.load[i] = self.model.addVar(lb=0, ub=self.vehicle_capacity, name=f'load_{i}')
            
        self.model.update()

    def _add_constraints(self):
        """Add all constraints"""
        # Objective function
        self.model.setObjective(
            gp.quicksum(self.costs[i,j] * self.x[i,j] for i,j in self.E_star),
            GRB.MINIMIZE
        )
        
        # Visit each customer once
        for u in self.customers:
            self.model.addConstr(
                gp.quicksum(self.x[i,u] for i,j in self.E_star if j == u) == 1,
                name=f'visit_in_{u}'
            )
            self.model.addConstr(
                gp.quicksum(self.x[u,j] for i,j in self.E_star if i == u) == 1,
                name=f'visit_out_{u}'
            )
        
        # Time window constraints
        M = max(tw[1] for tw in self.time_windows.values())
        for (i,j) in self.E_star:
            if j != self.depot_end:
                self.model.addConstr(
                    self.tau[j] >= self.tau[i] + self.service_times[i] + self.costs[i,j] 
                    - M * (1 - self.x[i,j]),
                    name=f'time_prop_{i}_{j}'
                )
        
        # Time window bounds
        for i in self.customers + [self.depot_start, self.depot_end]:
            self.model.addConstr(
                self.tau[i] >= self.time_windows[i][0],
                name=f'tw_lb_{i}'
            )
            self.model.addConstr(
                self.tau[i] <= self.time_windows[i][1],
                name=f'tw_ub_{i}'
            )

        # LA-arc constraints with parsimony
        self._add_la_arc_constraints_with_parsimony()
        
        # Capacity constraints
        self._add_capacity_constraints()
        
        # Time flow constraints
        self._add_time_flow_constraints()
        
        # Capacity flow constraints
        self._add_capacity_flow_constraints()

        self.model.update()

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
            self.model.addConstr(
                gp.quicksum(self.y[u,r] for r in range(len(self.R_u[u]))) == 1,
                name=f'one_ordering_{u}'
            )
            
            # Add k-indexed constraints from equation (8a)
            for k in range(1, len(distances) + 1):
                for w in N_k_plus_u[k]:
                    for v in N_k_u[k]:
                        if (w,v) in self.E_star:
                            self.model.addConstr(
                                rho * k + self.x[w,v] >= gp.quicksum(
                                    self.y[u,r] for r in range(len(self.R_u[u]))
                                    if self._is_in_k_neighborhood_ordering(r, w, v, k, N_k_u[k])
                                ),
                                name=f'la_arc_cons_{u}_{w}_{v}_{k}'
                            )
            
            # Add k-indexed constraints from equation (8b)
            for k in range(1, len(distances) + 1):
                for w in N_k_plus_u[k]:
                    outside_neighbors = [j for j in self.customers + [self.depot_end] 
                                      if j not in N_k_plus_u[k]]
                    if outside_neighbors:
                        self.model.addConstr(
                            rho * k + gp.quicksum(
                                self.x[w,j] for j in outside_neighbors 
                                if (w,j) in self.E_star
                            ) >= gp.quicksum(
                                self.y[u,r] for r in range(len(self.R_u[u]))
                                if self._is_final_in_k_neighborhood(r, w, k, N_k_plus_u[k])
                            ),
                            name=f'la_arc_final_{u}_{w}_{k}'
                        )

    def _add_capacity_constraints(self):
        """Add capacity flow constraints"""
        # Initial load from depot
        for j in self.customers:
            self.model.addConstr(
                self.load[j] >= self.demands[j] * self.x[self.depot_start,j],
                name=f'init_load_{j}'
            )
        
        # Load propagation between customers
        M = self.vehicle_capacity
        for i in self.customers:
            for j in self.customers:
                if i != j and (i,j) in self.E_star:
                    self.model.addConstr(
                        self.load[j] >= self.load[i] + self.demands[j] - M * (1 - self.x[i,j]),
                        name=f'load_prop_{i}_{j}'
                    )
        
        # Enforce capacity limit
        for i in self.customers:
            self.model.addConstr(
                self.load[i] <= self.vehicle_capacity,
                name=f'cap_limit_{i}'
            )

    def _add_capacity_flow_constraints(self):
        """Add capacity flow constraints (equations 4a and 4b from paper)"""
        # Flow conservation (4a)
        for i, k, d_min, d_max in self.nodes_D:
            if i not in [self.depot_start, self.depot_end]:
                self.model.addConstr(
                    gp.quicksum(self.z_D[e] for e in self.edges_D if e[0] == (i,k)) ==
                    gp.quicksum(self.z_D[e] for e in self.edges_D if e[1] == (i,k)),
                    name=f'cap_flow_cons_{i}_{k}'
                )
        
        # Consistency with route variables (4b)
        for u,v in self.E_star:
            self.model.addConstr(
                self.x[u,v] == gp.quicksum(
                    self.z_D[e] for e in self.edges_D 
                    if e[0][0] == u and e[1][0] == v
                ),
                name=f'cap_flow_cons_route_{u}_{v}'
            )

    def _add_time_flow_constraints(self):
        """Add time flow constraints"""
        # Flow conservation (5a)
        for i, k, t_min, t_max in self.nodes_T:
            if i not in [self.depot_start, self.depot_end]:
                self.model.addConstr(
                    gp.quicksum(self.z_T[e] for e in self.edges_T if e[0] == (i,k)) ==
                    gp.quicksum(self.z_T[e] for e in self.edges_T if e[1] == (i,k)),
                    name=f'time_flow_cons_{i}_{k}'
                )
        
        # Consistency with route variables (5b)
        for u,v in self.E_star:
            self.model.addConstr(
                self.x[u,v] == gp.quicksum(
                    self.z_T[e] for e in self.edges_T 
                    if e[0][0] == u and e[1][0] == v
                ),
                name=f'time_flow_cons_route_{u}_{v}'
            )
        
        # Link time variables τ with time buckets
        M = max(tw[1] for tw in self.time_windows.values())
        for i, k, t_min, t_max in self.nodes_T:
            if i not in [self.depot_start, self.depot_end]:
                outgoing_edges = [e for e in self.edges_T if e[0] == (i,k)]
                if outgoing_edges:
                    self.model.addConstr(
                        self.tau[i] >= t_min - M * (1 - gp.quicksum(self.z_T[e] for e in outgoing_edges)),
                        name=f'time_bucket_lb_{i}_{k}'
                    )
                    self.model.addConstr(
                        self.tau[i] <= t_max + M * (1 - gp.quicksum(self.z_T[e] for e in outgoing_edges)),
                        name=f'time_bucket_ub_{i}_{k}'
                    )

    def solve_with_parsimony(self, time_limit=None):
        """Solve VRPTW with LA neighborhood parsimony"""
        print("\n=== Initial LA-Neighborhood Analysis ===")
        self._print_neighborhood_analysis()
        
        if time_limit:
            self.model.setParam('TimeLimit', time_limit)
        
        # Set other Gurobi parameters
        self.model.setParam('MIPGap', 0.01)  # 1% optimality gap
        self.model.setParam('Threads', 4)     # Use 4 threads
        
        iteration = 1
        last_lp_val = float('-inf')
        iter_since_reset = 0
        
        while iteration <= self.max_iterations:
            print(f"\n=== Iteration {iteration} ===")
            
            # Solve current iteration
            self.model.optimize()
            
            current_obj = self.model.objVal if self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
            
            print(f"\nIteration Objective: {current_obj}")
            
            if current_obj is not None and current_obj > last_lp_val + self.MIN_INC:
                print("Solution improved")
                last_lp_val = current_obj
                iter_since_reset = 0
                
                try:
                    # Get dual variables and analyze buckets
                    print("\n--- Getting LP Relaxation Information ---")
                    dual_vars = self.get_dual_variables()
                    
                    if dual_vars:  # Only analyze if we got dual variables
                        print("\n--- LA-Neighborhood Analysis After Improvement ---")
                        self._print_neighborhood_analysis()
                        print("\n--- Bucket Analysis ---")
                        self._print_bucket_analysis()
                except Exception as e:
                    print(f"Warning: Could not complete analysis due to: {e}")
                    
            else:
                print("No significant improvement")
                iter_since_reset += 1
            
            if iter_since_reset >= self.sigma:
                print("\nResetting neighborhoods to maximum size...")
                self._reset_neighborhoods()
                iter_since_reset = 0
            
            iteration += 1
        
        # Extract final solution
        solution = self._extract_solution()
        if solution['status'] == 'Optimal':
            self.validate_solution(solution)
        
        return solution

    def _print_neighborhood_analysis(self):
        """Print detailed analysis of LA neighborhoods"""
        print("\nLA-Neighborhood Details:")
        
        # Print sizes and members
        for u in self.customers:
            neighbors = self.la_neighbors[u]
            print(f"\nCustomer {u}:")
            print(f"  Neighborhood size: {len(neighbors)}")
            print(f"  Neighbors: {neighbors}")
            
            # Print distances to neighbors
            distances = [(j, self.costs[u,j]) for j in neighbors]
            distances.sort(key=lambda x: x[1])
            print("  Distances to neighbors:")
            for j, dist in distances:
                print(f"    -> Customer {j}: {dist/5:.1f}")
            
            # Print time window compatibility
            print("  Time window compatibility:")
            for j in neighbors:
                u_early, u_late = self.time_windows[u]
                j_early, j_late = self.time_windows[j]
                travel_time = self.costs[u,j]/5
                print(f"    -> Customer {j}: Window [{j_early}, {j_late}]")
                print(f"       Earliest possible arrival: {u_early + travel_time:.1f}")
                print(f"       Latest possible arrival: {u_late + travel_time:.1f}")

    def _print_bucket_analysis(self):
        """Print detailed analysis of time and capacity buckets"""
        print("\nTime Bucket Analysis:")
        for u in self.customers:
            print(f"\nCustomer {u}:")
            print(f"  Time window: {self.time_windows[u]}")
            print(f"  Number of buckets: {len(self.T_u[u])}")
            print("  Bucket ranges:")
            for i, (t_min, t_max) in enumerate(self.T_u[u]):
                print(f"    Bucket {i}: [{t_min:.1f}, {t_max:.1f}]")
        
        print("\nCapacity Bucket Analysis:")
        for u in self.customers:
            print(f"\nCustomer {u}:")
            print(f"  Demand: {self.demands[u]}")
            print(f"  Number of buckets: {len(self.D_u[u])}")
            print("  Bucket ranges:")
            for i, (d_min, d_max) in enumerate(self.D_u[u]):
                print(f"    Bucket {i}: [{d_min:.1f}, {d_max:.1f}]")

    def _extract_solution(self):
        """Extract solution details"""
        if self.model.Status == GRB.OPTIMAL:
            status = 'Optimal'
        elif self.model.Status == GRB.TIME_LIMIT:
            status = 'TimeLimit'
        else:
            status = 'Other'
        
        solution = {
            'status': status,
            'objective': self.model.ObjVal if status in ['Optimal', 'TimeLimit'] else None,
            'routes': self._extract_routes() if status in ['Optimal', 'TimeLimit'] else None,
            'computation_time': self.model.Runtime,
            'neighborhood_sizes': {u: len(neighbors) for u, neighbors in self.la_neighbors.items()}
        }
        
        return solution
    
    def _extract_routes(self):
            """Extract routes from solution"""
            if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                return None
                
            active_edges = [(i,j) for (i,j) in self.E_star 
                        if self.x[i,j].X > 0.5]
            
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
    
    def get_dual_variables(self):
        """Get dual variables from the LP relaxation"""
        # For Gurobi, we need to solve the LP relaxation first
        relaxed = self.model.copy()
        for v in relaxed.getVars():
            if v.vType != GRB.CONTINUOUS:
                v.vType = GRB.CONTINUOUS
        
        relaxed.optimize()
        
        # Get dual values using Pi attribute
        dual_vars = {}
        if relaxed.Status == GRB.OPTIMAL:
            for c in relaxed.getConstrs():
                try:
                    dual_vars[c.ConstrName] = relaxed.getConstrByName(c.ConstrName).Pi
                except Exception:
                    # If we can't get the dual for a constraint, skip it
                    continue
                    
        return dual_vars
        
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
        """Create initial time buckets for each customer"""
        buckets = {}
        
        # Create buckets for each customer
        for u in self.customers:
            earliest_time, latest_time = self.time_windows[u]
            time_span = latest_time - earliest_time
            bucket_size = time_span / self.time_granularity
            
            customer_buckets = []
            current_time = earliest_time
            
            # Create evenly spaced buckets
            for i in range(self.time_granularity):
                lower = current_time
                upper = min(current_time + bucket_size, latest_time)
                customer_buckets.append((lower, upper))
                current_time = upper
                
                if current_time >= latest_time:
                    break
            
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
            for k, (t_min, t_max) in enumerate(self.T_u[u]):
                nodes_T.append((u, k, t_min, t_max))
        
        # Add depot nodes
        depot_start_bucket = self.T_u[self.depot_start][0]
        depot_end_bucket = self.T_u[self.depot_end][0]
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
        
        print("\nUpdated graphs:")
        print(f"Time graph: {len(self.nodes_T)} nodes, {len(self.edges_T)} edges")
        print(f"Capacity graph: {len(self.nodes_D)} nodes, {len(self.edges_D)} edges")

    def _reset_neighborhoods(self):
        """Reset LA neighborhoods to maximum size"""
        for u in self.customers:
            distances = [(j, self.costs[u,j]) for j in self.customers if j != u]
            distances.sort(key=lambda x: x[1])
            self.la_neighbors[u] = [j for j, _ in distances[:self.K]]
            print(f"Reset neighbors for customer {u}: {self.la_neighbors[u]}")

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

def load_solomon_instance(filename, customer_ids=None):
    """
    Load Solomon VRPTW instance from CSV file
    
    Args:
        filename: Path to CSV file
        customer_ids: List of specific customer IDs to include
    """
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Convert 'Depot' to 0 in CUST_NUM
    df['CUST_NUM'] = df['CUST_NUM'].replace('Depot', '0')
    df['CUST_NUM'] = df['CUST_NUM'].astype(int)
    
    # Filter customers if specific IDs provided
    if customer_ids is not None:
        selected_ids = [0] + sorted(customer_ids)
        df = df[df['CUST_NUM'].isin(selected_ids)]
        
        if len(df) != len(selected_ids):
            missing = set(selected_ids) - set(df['CUST_NUM'])
            raise ValueError(f"Customer IDs not found: {missing}")
    
    # Extract coordinates
    coords = {row['CUST_NUM']: (row['XCOORD.'], row['YCOORD.']) 
             for _, row in df.iterrows()}
    
    # Create customer list (excluding depot)
    customers = sorted(list(set(df['CUST_NUM']) - {0}))
    
    # Add virtual end depot with same coordinates as start depot
    virtual_end = max(customers) + 1
    coords[virtual_end] = coords[0]
    
    # Calculate distances (rounded down to 1 decimal)
    # Important: costs/distances do NOT include service times
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
    time_windows[virtual_end] = time_windows[0]  # End depot has same time window as start
    
    # Extract service times (0 for depots)
    service_times = {row['CUST_NUM']: row['SERVICE_TIME'] 
                    for _, row in df.iterrows()}
    service_times[virtual_end] = 0
    
    demands = {row['CUST_NUM']: row['DEMAND']
              for _, row in df.iterrows()}
    demands[virtual_end] = 0
    
    # Create instance dictionary
    instance = {
        'customers': customers,
        'depot_start': 0,
        'depot_end': virtual_end,
        'costs': costs,  # Just distances between customers
        'time_windows': time_windows,
        'service_times': service_times,  # Added service times
        'demands': demands,
        'vehicle_capacity': 200
    }
    
    print("\nSelected Customer Details:")
    print("ID  Ready  Due    Service  Demand  Location")
    print("-" * 50)
    for c in customers:
        x, y = coords[c]
        tw = time_windows[c]
        print(f"{c:<3} {tw[0]:<6} {tw[1]:<6} {service_times[c]:<8} {demands[c]:<7} ({x},{y})")
    
    return instance

def run_solomon_instance(filename, customer_ids, K=3, time_granularity=3, capacity_granularity=3, max_iterations=5, time_limit=300):
    """
    Load and solve a Solomon instance with specific customers
    
    Args:
        filename: Path to Solomon instance file
        customer_ids: List of customer IDs to include
        K: Initial neighborhood size
        time_granularity: Number of time buckets per customer
        capacity_granularity: Number of capacity buckets per customer
        max_iterations: Maximum iterations for LA-Discretization
        time_limit: Time limit in seconds
    """
    print(f"Loading Solomon instance with {len(customer_ids)} selected customers...")
    instance = load_solomon_instance(filename, customer_ids)
    
    optimizer = VRPTWOptimizer(
        customers=instance['customers'],
        depot_start=instance['depot_start'],
        depot_end=instance['depot_end'],
        costs=instance['costs'],
        time_windows=instance['time_windows'],
        demands=instance['demands'],
        vehicle_capacity=instance['vehicle_capacity'],
        K=K,
        time_granularity=time_granularity,
        capacity_granularity=capacity_granularity,
        max_iterations=max_iterations
    )
    
    print("\nSolving...")
    solution = optimizer.solve_with_parsimony(time_limit=time_limit)
    
    return optimizer, solution

def test_phi_functions():
    """Test the phi functions with more comprehensive test cases"""
    print("\nTesting phi functions:")
    print("-" * 70)
    
    # Test Case 1: Overlapping time windows
    print("\nTest Case 1: Overlapping time windows")
    customers = {
        1: Customer(1, 0, 20, 2),    # id, earliest, latest, service_time
        2: Customer(2, 5, 25, 3),    # overlaps with 1 and 3
        3: Customer(3, 15, 35, 2)    # overlaps with 2
    }
    travel_times = {
        (1,2): 4,
        (2,3): 3,
        (1,3): 8
    }
    _run_test_case(customers, travel_times)
    
    # Test Case 2: Tight time windows
    print("\nTest Case 2: Tight time windows")
    customers = {
        1: Customer(1, 0, 10, 2),     # Must start early
        2: Customer(2, 15, 18, 1),    # Very tight window
        3: Customer(3, 25, 30, 2)     # Also tight window
    }
    travel_times = {
        (1,2): 5,
        (2,3): 5,
        (1,3): 12
    }
    _run_test_case(customers, travel_times)
    
    # Test Case 3: Long service times
    print("\nTest Case 3: Long service times")
    customers = {
        1: Customer(1, 0, 30, 10),    # Long service time
        2: Customer(2, 20, 50, 15),   # Even longer service
        3: Customer(3, 40, 80, 5)     # Normal service
    }
    travel_times = {
        (1,2): 5,
        (2,3): 5,
        (1,3): 12
    }
    _run_test_case(customers, travel_times)

def _run_test_case(customers: Dict[int, Customer], travel_times: Dict[Tuple[int, int], float]):
    """Run tests for a specific set of customers and travel times"""
    calculator = PhiCalculator(customers, travel_times)
    
    # Test single customers first
    print("\nSingle customer tests:")
    for cust_id in customers:
        route = Route([cust_id], travel_times)
        _test_route(calculator, route, customers)
    
    # Test pairs of customers
    print("\nTwo customer tests:")
    for i in customers:
        for j in customers:
            if i != j and (i,j) in travel_times:
                route = Route([i,j], travel_times)
                _test_route(calculator, route, customers)
    
    # Test three customer routes
    print("\nThree customer tests:")
    for i in customers:
        for j in customers:
            for k in customers:
                if (i != j and j != k and i != k and 
                    (i,j) in travel_times and (j,k) in travel_times):
                    route = Route([i,j,k], travel_times)
                    _test_route(calculator, route, customers)

def _test_route(calculator: PhiCalculator, route: Route, customers: Dict[int, Customer]):
    """Test calculations for a specific route"""
    try:
        print(f"\nTesting route: {route.sequence}")
        print("Time windows:")
        for i in route.sequence:
            c = customers[i]
            print(f"  Customer {i}: [{c.earliest}, {c.latest}] (service: {c.service_time})")
        
        # Calculate phi values
        phi = calculator.phi_r(route)
        phi_hat = calculator.phi_hat_r(route)
        print(f"φ_r   = {phi:.1f}")
        print(f"φ̂_r   = {phi_hat:.1f}")
        
        # Verify phi_r ≤ phi_hat_r
        assert phi <= phi_hat, f"Error: φ_r > φ̂_r for route {route.sequence}"
        print("✓ φ_r ≤ φ̂_r verified")
        
        # For multi-customer routes, verify time feasibility
        if len(route.sequence) > 1:
            # Check earliest possible completion
            time = phi
            for i in range(len(route.sequence)-1):
                u = route.sequence[i]
                v = route.sequence[i+1]
                # Add service time for current customer
                time += customers[u].service_time
                # Add travel time to next customer
                time += route.travel_times[(u,v)]
                # Check if we can still reach next customer
                assert time <= customers[v].latest, \
                    f"Time window violated at customer {v} (arrival: {time:.1f})"
                # Update time to start of service at next customer
                time = max(time, customers[v].earliest)
            print("✓ Earliest path time feasibility verified")
            
            # Check latest possible completion
            time = phi_hat
            for i in range(len(route.sequence)-1):
                u = route.sequence[i]
                v = route.sequence[i+1]
                time += customers[u].service_time + route.travel_times[(u,v)]
                assert time <= customers[v].latest, \
                    f"Latest path time window violated at customer {v}"
            print("✓ Latest path time feasibility verified")
        
    except AssertionError as e:
        print(f"❌ Test failed: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_la_arcs():
    """Test LA-arc generation with a small example"""
    # Test data
    customers = [1, 2, 3, 4, 5]
    demands = {i: 30 for i in customers}  # Each customer demands 30
    vehicle_capacity = 100                # Vehicle can carry 100
    
    # Customer 1's neighbors are 2 and 3
    neighbor_sets = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2],
        4: [3, 5],
        5: [3, 4]
    }
    
    # Simple grid layout: each unit is distance 1
    coordinates = {
        1: (0, 0),
        2: (0, 1),
        3: (1, 1),
        4: (2, 1),
        5: (2, 0)
    }
    
    # Calculate actual distances
    costs = {}
    for i, j in itertools.product(customers, customers):
        if i != j:
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            costs[i,j] = ((x2-x1)**2 + (y2-y1)**2)**0.5

    # Time windows - relatively loose to start
    time_windows = {
        1: (0, 20),
        2: (0, 20),
        3: (0, 20),
        4: (0, 20),
        5: (0, 20)
    }
    
    # Service times
    service_times = {i: 2 for i in customers}  # 2 time units each
    
    print("\nTest 1: Basic LA-arc generation")
    print("--------------------------------")
    la_arcs = generate_la_arcs(customers, neighbor_sets, costs, time_windows, 
                              service_times, demands, vehicle_capacity)
    
    # Check results for customer 1
    print("\nAnalyzing sequences for customer 1:")
    sequences = la_arcs[1]
    print(f"Total sequences: {len(sequences)}")
    
    # Group by length
    by_length = {}
    for seq in sequences:
        length = len(seq.sequence)
        by_length[length] = by_length.get(length, 0) + 1
    
    print("\nSequences by length:")
    for length, count in sorted(by_length.items()):
        print(f"Length {length}: {count} sequences")
    
    print("\nExample sequences:")
    for seq in sequences[:5]:  # Show first 5
        print(f"\nSequence: {seq.sequence}")
        print(f"Distance: {seq.total_distance:.2f}")
        print(f"Elementary: {seq.is_elementary()}")
        print(f"Capacity feasible: {seq.is_capacity_feasible(demands, vehicle_capacity)}")
        print(f"Time feasible: {seq.is_time_feasible()}")
    
    print("\nTest 2: Tight capacity constraints")
    print("----------------------------------")
    # Make demands larger so only short sequences are feasible
    tight_demands = {i: 60 for i in customers}
    la_arcs_tight = generate_la_arcs(customers, neighbor_sets, costs, time_windows, 
                                    service_times, tight_demands, vehicle_capacity)
    
    print(f"\nSequences for customer 1 with tight capacity:")
    sequences = la_arcs_tight[1]
    print(f"Total sequences: {len(sequences)}")
    
    print("\nTest 3: Tight time windows")
    print("---------------------------")
    # Make time windows tighter
    tight_windows = {
        1: (0, 10),
        2: (5, 15),
        3: (10, 20),
        4: (15, 25),
        5: (20, 30)
    }
    la_arcs_tight_time = generate_la_arcs(customers, neighbor_sets, costs, tight_windows, 
                                         service_times, demands, vehicle_capacity)
    
    print(f"\nSequences for customer 1 with tight time windows:")
    sequences = la_arcs_tight_time[1]
    print(f"Total sequences: {len(sequences)}")

def test_la_arcs_with_comparison(debug=True):
    """Test LA-arc generation and compare with TimeCalculator results"""
    # Original test data
    customers = [1, 2, 3, 4, 5]
    
    # Customer 1's neighbors are 2 and 3
    neighbor_sets = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2],
        4: [3, 5],
        5: [3, 4]
    }
    
    # Simple grid layout: each unit is distance 1
    coordinates = {
        1: (0, 0),
        2: (0, 1),
        3: (1, 1),
        4: (2, 1),
        5: (2, 0)
    }
    
    # Calculate actual distances
    costs = {}
    for i, j in itertools.product(customers, customers):
        if i != j:
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            costs[i,j] = ((x2-x1)**2 + (y2-y1)**2)**0.5
    
    # Time windows from tight time windows test
    time_windows = {
        1: (0, 10),   # Customer 1: must start between 0-10
        2: (5, 15),   # Customer 2: must start between 5-15
        3: (10, 20),  # Customer 3: must start between 10-20
        4: (15, 25),  # Customer 4: must start between 15-25
        5: (20, 30)   # Customer 5: must start between 20-30
    }
    
    service_times = {i: 2 for i in customers}  # 2 time units per customer
    
    # Create calculator
    calculator = TimeCalculator(time_windows, service_times, costs)
    
    # First run original LA-arc test
    print("\nRunning original LA-arc tests...")
    test_la_arcs()
    
    # Now check some specific sequences with TimeCalculator
    print("\nAnalyzing sequences with TimeCalculator:")
    print("-" * 50)
    
    test_sequences = [
        [1],        # Single customer
        [1, 4],     # Direct to non-neighbor
        [1, 2, 4],  # One intermediate stop
        [1, 2, 3, 4] # Two intermediate stops
    ]
    
    for sequence in test_sequences:
        print(f"\nAnalyzing sequence: {sequence}")
        
        # Get time windows for context
        windows = [f"[{time_windows[i][0]}, {time_windows[i][1]}]" for i in sequence]
        print(f"Time windows: {', '.join(windows)}")
        
        # Calculate φ_r and φ̂_r
        phi = calculator.phi_r(sequence)
        phi_hat = calculator.phi_hat_r(sequence)
        
        print(f"φ_r   = {phi:.1f}")
        print(f"φ̂_r   = {phi_hat:.1f}")
        
        if phi != float('-inf'):
            print(f"Sequence is time-feasible")
            print(f"Can depart first customer between {phi:.1f} and {phi_hat:.1f}")
            
            # Validate the sequence
            current_time = phi
            print("\nValidating sequence:")
            for i in range(len(sequence)-1):
                u = sequence[i]
                v = sequence[i+1]
                
                print(f"  At {u}: time = {current_time:.1f}")
                current_time += service_times[u]
                print(f"  After service: time = {current_time:.1f}")
                current_time += costs[u,v]
                print(f"  Arrive at {v}: time = {current_time:.1f}")
                current_time = max(current_time, time_windows[v][0])
                print(f"  Start service: time = {current_time:.1f}")
                
        else:
            print("Sequence is time-infeasible")

def analyze_sequences(sequences, time_windows, service_times):
    """Analyze patterns in feasible sequences"""
    stats = {
        'count': len(sequences),
        'by_length': defaultdict(int),
        'earliest_departures': [],  # For φ_r analysis
        'latest_departures': [],    # For φ̂_r analysis
        'time_slack': []            # Time window flexibility
    }
    
    for seq in sequences:
        # Count by length
        stats['by_length'][len(seq.sequence)] += 1
        
        # Calculate earliest possible departure time (φ_r related)
        earliest_time = time_windows[seq.sequence[0]][0]
        total_time = 0
        for i in range(len(seq.sequence)-1):
            current = seq.sequence[i]
            next_customer = seq.sequence[i+1]
            total_time += service_times[current] + seq.travel_times[(current, next_customer)]
        stats['earliest_departures'].append(earliest_time)
        
        # Calculate latest possible departure time (φ̂_r related)
        latest_deadline = float('inf')
        current_time = 0
        for i in range(len(seq.sequence)):
            current = seq.sequence[i]
            if i < len(seq.sequence)-1:
                next_customer = seq.sequence[i+1]
                latest_possible = time_windows[next_customer][1] - current_time - service_times[current] - seq.travel_times[(current, next_customer)]
                latest_deadline = min(latest_deadline, latest_possible)
            current_time += service_times[current]
            if i < len(seq.sequence)-1:
                current_time += seq.travel_times[(current, next_customer)]
        stats['latest_departures'].append(latest_deadline)
        
        # Calculate time window flexibility
        slack = time_windows[seq.sequence[0]][1] - time_windows[seq.sequence[0]][0]
        stats['time_slack'].append(slack)
    
    return stats

def print_sequence_analysis(customer_id, sequences, time_windows, service_times):
    """Print detailed analysis of sequences for a customer"""
    stats = analyze_sequences(sequences, time_windows, service_times)
    
    print(f"\nSequence Analysis for Customer {customer_id}")
    print("=" * 50)
    print(f"Total feasible sequences: {stats['count']}")
    
    print("\nSequence Lengths:")
    for length, count in sorted(stats['by_length'].items()):
        print(f"  Length {length}: {count} sequences")
    
    if stats['earliest_departures']:
        print("\nDeparture Time Analysis:")
        print(f"  Earliest possible departure: {min(stats['earliest_departures']):.1f}")
        print(f"  Latest allowed departure: {min(stats['latest_departures']):.1f}")
        print(f"  Average time slack: {sum(stats['time_slack'])/len(stats['time_slack']):.1f}")
        
        print("\nSequence Details:")
        for seq in sequences:
            print(f"\n  Sequence: {seq.sequence}")
            print(f"    Distance: {seq.total_distance:.2f}")
            print(f"    Time window at start: {time_windows[seq.sequence[0]]}")
            if len(seq.sequence) > 1:
                print(f"    Time window at end: {time_windows[seq.sequence[-1]]}")
    else:
        print("\nNo feasible sequences found!")


if __name__ == "__main__":
    print("Testing LA-Arc generation with TimeCalculator...")
    test_la_arcs_with_comparison(debug=True)