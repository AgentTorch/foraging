import torch
import torch.nn.functional as F

def create_transition_matrix(n, m, inaccessible_states, food_states, food_bias=2.0, move_radius=1):
    """
    Create a transition matrix for a grid world environment.
    
    Args:
    n, m: Grid dimensions
    inaccessible_states: List of (x, y) tuples representing inaccessible states
    food_states: List of (x, y) tuples representing states with food
    food_bias: Factor to increase probability of moving towards food states
    move_radius: Maximum distance a bird can move in one step
    
    Returns:
    transition_matrix: Tensor of shape (NM, NM) representing transition probabilities
    """
    NM = n * m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a mask for accessible states
    accessible_mask = torch.ones(n, m, device=device)
    for x, y in inaccessible_states:
        accessible_mask[x, y] = 0
    
    # Create a mask for food states
    food_mask = torch.ones(n, m, device=device)
    for x, y in food_states:
        food_mask[x, y] = food_bias
    
    # Initialize transition matrix
    transition_matrix = torch.zeros((NM, NM), device=device)
    
    # Define possible moves within the move_radius
    moves = [(dx, dy) for dx in range(-move_radius, move_radius+1) 
                      for dy in range(-move_radius, move_radius+1) 
                      if 0 < dx*dx + dy*dy <= move_radius*move_radius]
    
    # Populate transition matrix
    for x in range(n):
        for y in range(m):
            if accessible_mask[x, y] == 0:
                continue  # Skip inaccessible states
            
            current_state = x * m + y
            valid_moves = []
            
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and accessible_mask[nx, ny] == 1:
                    next_state = nx * m + ny
                    probability = food_mask[nx, ny]
                    valid_moves.append((next_state, probability))
            
            # Normalize probabilities
            total_probability = sum(prob for _, prob in valid_moves)
            for next_state, probability in valid_moves:
                transition_matrix[current_state, next_state] = probability / total_probability
    
    return transition_matrix

def compute_sr_and_values(transition_matrix, reward_vector, gamma=0.9):
    """
    Compute the successor representations and value of each state in the grid.
    
    Args:
    transition_matrix: Tensor of shape (NM, NM) representing transition probabilities
    reward_vector: Tensor of shape (NM, 1) representing rewards for each state
    gamma: Discount factor
    
    Returns:
    SR: Successor Representation matrix of shape (NM, NM)
    values: Value of each state, shape (NM, 1)
    """
    NM = transition_matrix.shape[0]
    
    # Compute Successor Representation
    I = torch.eye(NM, device=transition_matrix.device)
    SR = torch.inverse(I - gamma * transition_matrix)
    
    # Compute values
    values = SR @ reward_vector
    
    return SR, values

# Example usage:
n, m = 10, 10
NM = n * m

# Create a sample transition matrix (you would normally compute this based on your environment)
transition_matrix = torch.rand((NM, NM))
transition_matrix /= transition_matrix.sum(dim=1, keepdim=True)

# Create a sample reward vector
reward_vector = torch.zeros((NM, 1))
reward_vector[55] = 1  # Food at (5, 5)
reward_vector[88] = 1  # Food at (8, 8)

# Compute SR and values
SR, values = compute_sr_and_values(transition_matrix, reward_vector)

# Sample bird positions
num_birds = 5
bird_positions = torch.randint(0, 10, (num_birds, 2))

# Compute memory steering
radius = 2
memory_steering = compute_memory_steering(bird_positions, values, n, m, radius)

print("Bird positions:")
print(bird_positions)
print("\nMemory steering vectors:")
print(memory_steering)

if __name__ == '__main__':
    # Example usage:
    n, m = 10, 10
    inaccessible_states = [(2, 2), (2, 3), (3, 2), (3, 3)]  # Example of inaccessible states
    food_states = [(5, 5), (8, 8)]  # States with food

    transition_matrix = create_transition_matrix(n, m, inaccessible_states, food_states)

    print("Transition matrix shape:", transition_matrix.shape)
    print("Sum of probabilities for first state:", transition_matrix[0].sum().item())