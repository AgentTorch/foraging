from agent_torch import Registry
from agent_torch.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.helpers import get_by_path
import torch
import re

def read_var(state, path):
    return get_by_path(state, re.split('/', path))

def limit_magnitude(vectors, max_magnitude):
    magnitudes = torch.norm(vectors, dim=1, keepdim=True)
    return torch.where(
        magnitudes > max_magnitude,
        vectors / magnitudes * max_magnitude,
        vectors
    )

@Registry.register_substep("observe_neighbors", "observation")
class ObserveNeighbors(SubstepObservation):
    '''find set of neighbors in observation radius'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.device = torch.device(self.config['simulation_metadata']['device'])

    def find_neighbors(self, positions, perception_radius):
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # BxBx2
        distance = torch.norm(diff, dim=2)  # BxB
        neighbor_mask = (distance < perception_radius) & (distance > 0)  # Exclude self

        return neighbor_mask

    def forward(self, state):
        input_variables = self.input_variables
        agent_locations = read_var(state, input_variables['location']).float()
        neighborhood = read_var(state, input_variables['neighbor_radius'])

        neighbor_mask = self.find_neighbors(agent_locations, neighborhood) # BxB mask

        return {self.output_variables[0]: neighbor_mask}

@Registry.register_substep("social_steering", "policy")
class SocialSteering(SubstepAction):
    '''compute steering based on cohesion, alignment and seperation weights'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.max_force = self.config['simulation_metadata']['max_force']

        self.cohesion_weight = self.args['cohesion_weight']
        self.alignment_weight = self.args['alignment_weight']
        self.separation_weight = self.args['separation_weight']
    
    def _separation(self, positions, neighbor_mask, separation_distance):
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # BxBx2
        distance = torch.norm(diff, dim=2)  # BxB
        close_mask = (distance < separation_distance) & neighbor_mask
        steering = torch.where(
            close_mask.unsqueeze(2),
            diff / (distance.unsqueeze(2) + 1e-8),
            torch.zeros_like(diff)
        ).sum(dim=1)  # Bx2

        return steering

    def _alignment(self, positions, velocities, neighbor_mask):
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True)  # Bx1
        avg_velocity = (velocities.unsqueeze(1) * neighbor_mask.unsqueeze(2)).sum(dim=1) / (neighbor_count + 1e-8)  # Bx2
        steering = avg_velocity - velocities

        return steering

    def _cohesion(self, positions, neighbor_mask):
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True) # Bx1
        center_of_mass = (positions.unsqueeze(1) * neighbor_mask.unsqueeze(2)).sum(dim=1) / (neighbor_count + 1e-8)  # Bx2
        steering = center_of_mass - positions

        return steering

    def forward(self, state, observation):
        input_variables = self.input_variables
        neighbor_mask = observation['neighbor_mask']

        location = read_var(state, input_variables['location']).float()
        velocity = read_var(state, input_variables['velocity'])
        separation_distance = read_var(state, input_variables['separation_distance'])
        
        separation = self._separation(location, neighbor_mask, separation_distance)
        alignment = self._alignment(location, velocity, neighbor_mask)
        cohesion = self._cohesion(location, neighbor_mask)

        steering = separation + self.alignment_weight * alignment + self.cohesion_weight * cohesion
        steering = limit_magnitude(steering, self.max_force)

        return {self.output_variables[0]: steering}

@Registry.register_substep("memory_steering", "policy")
class MemorySteering(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_memory_steering(bird_positions, values, n, m, radius):
        """
        Compute memory-based steering for all birds.
        
        Args:
        bird_positions: Tensor of shape (num_birds, 2) representing (x, y) positions
        values: Tensor of shape (NM, 1) representing value of each state
        n, m: Grid dimensions
        radius: Neighborhood radius for considering next states
        
        Returns:
        memory_steering: Tensor of shape (num_birds, 2) representing steering vectors
        """
        num_birds = bird_positions.shape[0]
        device = bird_positions.device
        
        # Generate all possible offsets within the radius
        y, x = torch.meshgrid(torch.arange(-radius, radius+1, device=device),
                            torch.arange(-radius, radius+1, device=device))
        offsets = torch.stack((x.flatten(), y.flatten()), dim=1)
        valid_offsets = torch.sum(offsets**2, dim=1) <= radius**2
        offsets = offsets[valid_offsets]
        
        # Compute neighbor positions for all birds
        neighbors = bird_positions.unsqueeze(1) + offsets.unsqueeze(0)
        
        # Clip neighbors to valid grid positions
        neighbors = torch.clamp(neighbors, min=0, max=torch.tensor([n-1, m-1], device=device))
        
        # Compute state indices for birds and their neighbors
        bird_indices = bird_positions[:, 0] * m + bird_positions[:, 1]
        neighbor_indices = neighbors[:, :, 0] * m + neighbors[:, :, 1]
        
        # Get values for neighbors
        neighbor_values = values[neighbor_indices]
        
        # Compute softmax probabilities
        probs = F.softmax(neighbor_values, dim=1)
        
        # Compute steering vectors
        steering_vectors = neighbors - bird_positions.unsqueeze(1)
        memory_steering = torch.sum(probs * steering_vectors, dim=1)
        
        return memory_steering

    def forward(self, state, observation):
        pass

@Registry.register_substep("update_location_velocity", "transition")
class UpdateLocationVelocity(SubstepTransition):
    '''assign new location based on decided steering'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.max_speed = self.config['simulation_metadata']['max_speed']

    def _update_velocity(self, curr_velocity, target_steering):
        new_velocity = curr_velocity + target_steering
        new_velocity = limit_magnitude(new_velocity, self.max_speed)

        return new_velocity

    def _update_location(self, curr_location, new_velocity):
        new_location = curr_location + new_velocity

        return new_location

    def forward(self, state, action):
        input_variables = self.input_variables
        target_steering = action['bird']['steering']
        location = read_var(state, input_variables['location'])
        velocity = read_var(state, input_variables['velocity'])

        new_velocity = self._update_velocity(velocity, target_steering)
        new_location = self._update_location(location, new_velocity)

        return {self.output_variables[0]: new_location,
                self.output_variables[1]: new_velocity}