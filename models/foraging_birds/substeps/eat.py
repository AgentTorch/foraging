from agent_torch.core.substep import SubstepAction, SubstepTransition
from agent_torch.core.registry import Registry
from agent_torch.core.helpers import get_by_path
import torch
import re
import torch.nn.functional as F

def read_var(state, path):
    return get_by_path(state, re.split('/', path))

@Registry.register_substep("decide_eat_or_move", "policy")
class DecideEatOrMove(SubstepAction):
    '''agents willigness to eat based on marginal value theory'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.depletion_rate = self.config['simulation_metadata']['depletion_rate']
        self.avg_patch_quality = self.config['simulation_metadata']['average_patch_quality']
        self.avg_travel_time = self.config['simulation_metadata']['average_travel_time'] # computed based on distribution on rewards

        self.num_birds = self.config['simulation_metadata']['num_birds']

        self.END_FORAGING_VAR = 0
        self.START_FORAGING_VAR = 1

        self.INCREMENT_VAR = 1

    def _start(self, bird_positions, nutritional_value):
        """
        For birds not currently foraging that may chose to start
        """
        m, n = nutritional_value.shape
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(torch.arange(n, device=nutritional_value.device), 
                                        torch.arange(m, device=nutritional_value.device),
                                        indexing='ij')
        
        # Reshape grid coordinates and nutritional values
        grid_coords = torch.stack([y_grid, x_grid], dim=-1).view(-1, 2).float()
        nutr_values = nutritional_value.view(-1)
        
        # Compute distances between bird positions and all grid positions
        distances = torch.sum((bird_positions.unsqueeze(1) - grid_coords.unsqueeze(0)) ** 2, dim=-1)
        
        # Create a soft selection mask using softmax with a large negative factor for sharpness
        selection_mask = F.softmax(-1000 * distances, dim=1)
        
        # Use the mask to get a weighted sum of nutritional values for each bird
        bird_patch_resources = torch.sum(selection_mask * nutr_values.unsqueeze(0), dim=1)

        start_foraging_decision = (bird_patch_resources > 0)

        return start_foraging_decision.unsqueeze(1)

    def _end_or_resume(self, time_in_patch, bird_positions):
        """
        For birds currently foraging that may chose to end or resume
        """
        # Compute instantaneous gain rate for foraging birds
        instantaneous_gain_rate = self.depletion_rate * torch.exp(-self.depletion_rate * time_in_patch)

        # Compute average rate of gain for entire environment
        avg_gain_rate = self.avg_patch_quality / (self.avg_travel_time + self.avg_patch_quality / self.depletion_rate)

        # Decision for birds currently foraging
        resume_foraging_decision = (instantaneous_gain_rate > avg_gain_rate)

        return resume_foraging_decision

    def forward(self, state, observation):
        """
        Decide whether birds should start, continue, or stop foraging based on MVT.
        
        Args:
        bird_positions: tensor of shape (num_birds, 2) with (x, y) positions
        current_patch_resources: tensor of shape (num_birds,) representing remaining resources in each bird's patch
        time_in_patch: tensor of shape (num_birds,) representing time spent in current patch for each bird
        is_foraging: tensor of shape (num_birds,) boolean indicating if each bird is currently foraging
        
        Returns:
        decisions: tensor of shape (num_birds,) with values:
                   0 - for birds not currently foraging
                   1 - for birds currently foraging.
        """

        input_variables = self.input_variables

        time_in_patch = read_var(state, input_variables['time_in_patch'])
        is_foraging = read_var(state, input_variables['is_foraging'])
        bird_positions = read_var(state, input_variables['location'])
        current_patch_resources = read_var(state, input_variables['nutritional_value'])

        start_foraging_decision = self._start(bird_positions, current_patch_resources)
        resume_foraging_decision = self._end_or_resume(time_in_patch, bird_positions)

        eating_decision = torch.zeros(start_foraging_decision.shape, dtype=torch.float, device=start_foraging_decision.device)
        print("Eating decision: ", eating_decision.shape)
        eating_decision = eating_decision + self.INCREMENT_VAR*(is_foraging*resume_foraging_decision + (1. - is_foraging)*start_foraging_decision)

        return {self.output_variables[0]: eating_decision}

@Registry.register_substep("update_food_and_action", "transition")
class UpdateFoodAndAction(SubstepTransition):
    '''Change food consumed by the bird and nutrition of the patch, at each time step'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.depletion_rate = self.config['simulation_metadata']['depletion_rate']
        self.INCREMENT_VAR = 1
        self.RESET_VAR = -1

        self.NOT_FORAGING_STATE = 0 # decision for agents not currently foraging: will move in next step
        self.IS_FORAGING_STATE = 1 # decision for agents currrently foraging: will not move

    def _update_time(self, time_in_patch, decisions):
        curr_time_in_patch = time_in_patch.clone()
        is_foraging_delta = (decisions == self.IS_FORAGING_STATE)*self.INCREMENT_VAR # increase by 1
        not_foraging_delta = (decisions == self.NOT_FORAGING_STATE)*self.RESET_VAR*curr_time_in_patch # reset to 0

        time_delta = is_foraging_delta + not_foraging_delta
        new_time_in_patch = time_in_patch + time_delta

        return new_time_in_patch
    
    def _update_foraging_status(self, is_foraging, decisions):
        updated_is_foraging = decisions.clone() # decision == 1: continue foraging; elif decision == 0: stop foraging.

        return updated_is_foraging

    def _update_nutrition(self, nutritional_value, decisions, bird_positions):
        '''resets the nutritional value of each patch location based on birds consuming food'''
        foraging_mask = decisions.float()
        depletion = torch.zeros(nutritional_value.shape)
        depletion.index_put_((bird_positions[:, 0].long(), bird_positions[:, 1].long()), 
                            foraging_mask * self.depletion_rate, 
                            accumulate=True)
        
        '''Check differentiability of this function'''
        updated_patch_nutritional_value = torch.clamp(patch_nutritional_value - depletion, min=0)
        
        return updated_patch_nutritional_value
    
    def forward(self, state, action):
        eating_decision = action['bird']['eating_decision']

        time_in_patch = read_var(state, self.input_variables['time_in_patch'])
        is_foraging = read_var(state, self.input_variables['is_foraging'])
        nutritional_value = read_var(state, self.input_variables['nutritional_value'])
        bird_positions = read_var(state, self.input_variables['location'])

        new_is_foraging = self._update_foraging_status(is_foraging, eating_decision)
        new_time_in_patch = self._update_time(time_in_patch, eating_decision)
        new_nutritional_value = self._update_nutrition(nutritional_value, eating_decision, bird_positions)
    
        return {self.output_variables[0]: new_nutritional_value,
                self.output_variables[1]: new_time_in_patch,
                self.output_variables[2]: new_is_foraging}