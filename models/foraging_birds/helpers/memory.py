import torch
import torch.nn.functional as F
from agent_torch.core import Registry
import os
import pandas as pd

def create_transition_matrix_vectorized(n, m, inaccessible_states, food_states, food_bias=2.0, move_radius=1):
    """
    Create a transition matrix for a grid world environment using vectorized operations.
    
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
    
    # Create masks for accessible and food states
    accessible_mask = torch.ones(n, m)
    accessible_mask[torch.tensor(inaccessible_states).T.tolist()] = 0
    
    food_mask = torch.ones(n, m)
    food_mask[torch.tensor(food_states).T.tolist()] = food_bias
    
    # Create all possible state pairs
    all_states = torch.arange(NM)
    x = all_states // m
    y = all_states % m
    
    # Compute distances between all state pairs
    dx = x.unsqueeze(1) - x.unsqueeze(0)
    dy = y.unsqueeze(1) - y.unsqueeze(0)
    distances = dx.pow(2) + dy.pow(2)
    
    # Create transition matrix based on move_radius
    transition_matrix = ((distances > 0) & (distances <= move_radius**2)).float()
    
    # Apply accessibility mask
    transition_matrix *= accessible_mask.view(-1).unsqueeze(1)
    transition_matrix *= accessible_mask.view(-1).unsqueeze(0)
    
    # Apply food bias
    transition_matrix *= food_mask.view(-1).unsqueeze(0)
    
    # Normalize probabilities
    transition_matrix /= transition_matrix.sum(dim=1, keepdim=True).clamp(min=1e-10)
    
    return transition_matrix

def indices_to_binary_matrix(indices: list, n: int, m: int) -> torch.Tensor:
    """
    Create a binary matrix from a list of indices.

    Args:
    indices (list): A list of tuples (x, y) representing locations to be marked as 1.
    n (int): Number of rows in the matrix.
    m (int): Number of columns in the matrix.

    Returns:
    torch.Tensor: A binary tensor of shape (n, m) with 1s at the specified indices and 0s elsewhere.
    """
    # Create a zero tensor of shape (n, m)
    binary_matrix = torch.zeros((n, m), dtype=torch.float32)
    
    # Set 1s at the specified indices
    for x, y in indices:
        if 0 <= x < m and 0 <= y < n:  # Check if indices are within bounds
            binary_matrix[y, x] = 1
        else:
            print(f"Warning: Index ({x}, {y}) is out of bounds and will be ignored.")
    
    return binary_matrix

@Registry.register_helper('compute_sr_values', 'initialization')
def compute_sr_and_values(shape, params):
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
    n, m = shape
    gamma = params['gamma']
    inaccessible_states = []

    food_locations_path = os.path.join(os.getcwd(), params['reward_vector'])
    food_states = pd.read_csv(food_locations_path).values

    reward_matrix = indices_to_binary_matrix(food_states, n, m)
    reward_vector = reward_matrix.view(-1)

    transition_matrix = create_transition_matrix_vectorized(n, m, inaccessible_states, food_states)
    NM = transition_matrix.shape[0]
    
    # Compute Successor Representation
    I = torch.eye(NM)
    SR = torch.inverse(I - gamma * transition_matrix)
    
    # Compute values
    state_values = SR @ reward_vector
    state_values = state_values.view(n, m)

    assert state_values.shape == torch.Size(shape)

    return state_values

if __name__ == '__main__':
    # Example usage:
    n, m = 10, 10
    NM = n * m
    inaccessible_states = [(2, 2), (2, 3), (3, 2), (3, 3)]  # Example of inaccessible states
    food_states = [(5, 5), (8, 8)]  # States with food

    transition_matrix = create_transition_matrix(n, m, inaccessible_states, food_states)

    print("Transition matrix shape:", transition_matrix.shape)
    print("Sum of probabilities for first state:", transition_matrix[0].sum().item())

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