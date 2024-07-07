from agent_torch.core import Registry
from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.core.helpers import get_by_path
import torch
import re
import torch.nn.functional as F

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

        social_steering = separation + self.alignment_weight * alignment + self.cohesion_weight * cohesion
        social_steering = limit_magnitude(social_steering, self.max_force)

        return {self.output_variables[0]: social_steering}

@Registry.register_substep("memory_steering", "policy")
class MemorySteering(SubstepAction):
    """
        Compute memory-based steering for all birds.
        
        Returns:
        memory_steering: Tensor of shape (num_birds, 2)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device(self.config['simulation_metadata']['device'])

        self.w = self.config['simulation_metadata']['max_x']
        self.h = self.config['simulation_metadata']['max_y']
        self.num_birds = self.config['simulation_metadata']['num_birds']

    def forward(self, state, observation):
        input_variables = self.input_variables
        neighbor_mask = observation['neighbor_mask']

        bird_positions = read_var(state, input_variables['location']).float()
        state_values = read_var(state, input_variables['state_value'])

        y_grid, x_grid = torch.meshgrid(torch.arange(self.h), torch.arange(self.w), indexing='ij')
        grid_positions = torch.stack((y_grid, x_grid), dim=-1).to(bird_positions.device)

        # Compute soft indexing weights
        distances = torch.sum((bird_positions.unsqueeze(1).unsqueeze(1) - grid_positions.unsqueeze(0)) ** 2, dim=-1)
        indexing_weights = F.softmax(-distances.view(self.num_birds, -1), dim=-1).view(self.num_birds, self.h, self.w)

        # Soft indexing of state values
        bird_state_values = torch.sum(indexing_weights * state_values.unsqueeze(0), dim=(1, 2))

        # Expand bird positions and values for neighbor comparisons
        expanded_positions = bird_positions.unsqueeze(1).expand(-1, self.num_birds, -1)
        expanded_values = bird_state_values.unsqueeze(1).expand(-1, self.num_birds)

        # Apply neighbor mask
        valid_neighbors = neighbor_mask.float()
        neighbor_positions = expanded_positions * valid_neighbors.unsqueeze(-1)
        neighbor_values = expanded_values * valid_neighbors

        # Compute softmax probabilities
        probs = F.softmax(neighbor_values + 1e-10, dim=1)

        # Compute steering vectors
        steering_vectors = neighbor_positions - bird_positions.unsqueeze(1)

        # Compute memory steering
        memory_steering = torch.sum(probs.unsqueeze(-1) * steering_vectors, dim=1)

        return {self.output_variables[0]: memory_steering}

@Registry.register_substep("update_location_velocity", "transition")
class UpdateLocationVelocity(SubstepTransition):
    '''assign new location based on decided steering'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.max_speed = self.config['simulation_metadata']['max_speed']
        self.cognitive_weight = self.args['cognitive_weight']

    def _update_velocity(self, curr_velocity, target_steering):
        new_velocity = curr_velocity + target_steering
        new_velocity = limit_magnitude(new_velocity, self.max_speed)

        return new_velocity

    def _update_location(self, curr_location, new_velocity):
        new_location = curr_location + new_velocity

        return new_location

    def forward(self, state, action):
        input_variables = self.input_variables
        social_steering = action['bird']['social_steering']
        memory_steering = action['bird']['memory_steering']

        location = read_var(state, input_variables['location'])
        velocity = read_var(state, input_variables['velocity'])

        target_steering = self.cognitive_weight*memory_steering + (1 - self.cognitive_weight)*social_steering

        new_velocity = self._update_velocity(velocity, target_steering)
        new_location = self._update_location(location, new_velocity)

        return {self.output_variables[0]: new_location,
                self.output_variables[1]: new_velocity}