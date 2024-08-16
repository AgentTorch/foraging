# Bird Foraging Model Card for AgentTorch

## Model Details
- **Name:** Bird Foraging and Flocking Model
- **Version:** 0.4.0
- **Type:** Agent-based model (ABM) environment
- **Framework:** AgentTorch v0.4.0
- **Execution Mode:** Heuristic

## Intended Use
- **Primary Use:** Simulate bird foraging behavior with flocking dynamics and optimal foraging theory
- **Intended Users:** Ecologists, behavioral biologists, and researchers studying animal movement and foraging strategies

## Model Architecture
- **Environment:** 10x10 grid
- **Simulation Duration:** 2 episodes, 20 steps per episode, 2 substeps per step
- **Agent Population:** 40 birds
- **Food Sources:** 5 food patches

## Components

### Agents (Birds)
- **Properties:** 
  - Location (2D coordinates)
  - Velocity
  - Foraging status
  - Time spent in patch

- **Behaviors:** 
  - Move (based on boids algorithm)
  - Forage (based on Marginal Value Theorem)

### Environment
- **Properties:**
  - Grid bounds
  - Neighbor radius (4.0 units)
  - Separation distance (1.0 units)
  - State value (based on Successor Representation)

### Objects (Food)
- **Properties:**
  - Location
  - Nutritional value

## Simulation Substeps
1. **Move:** Birds move based on boids steering algorithm
   - Social steering (cohesion, alignment, separation)
   - Memory-based steering
2. **Eat:** Birds decide to eat or move based on Marginal Value Theorem

## Input Data
- Initial bird locations (from file)
- Food patch locations (from file)

## Model Parameters
- **Movement:**
  - Max speed: 0.5
  - Min speed: -0.5
  - Max force: 3.0
  - Cohesion weight: 1.0
  - Alignment weight: 0.1
  - Cognitive weight: 0.7

- **Foraging:**
  - Depletion rate: 0.3
  - Average patch quality: 0.05
  - Average travel time: 20

- **Environment:**
  - Memory decay parameter (gamma): 0.9

## Key Features
- Integration of flocking behavior (boids algorithm) with foraging decisions
- Use of Successor Representation for environmental state valuation
- Implementation of Marginal Value Theorem for foraging decisions
- Dynamic food depletion and patch-leaving decisions

## Output Data
- Bird positions and velocities over time
- Foraging status and time spent in patches
- Food patch depletion levels

## Technical Specifications
- **Programming Language:** Python
- **Dependencies:** AgentTorch v0.4.0 framework, PyTorch
- **Compute Requirements:** CPU (as specified in config)

## Limitations
- Simplified 2D grid environment
- Fixed number of agents and food patches
- Does not account for factors like predation, reproduction, or seasonal changes
- Simplified representation of food patch regeneration

## Ethical Considerations
- Model simplifications may not capture all nuances of real bird behavior
- Results should be interpreted cautiously when applied to wildlife management or conservation policies

## References
- AgentTorch GitHub repository: [github.com/AgentTorch/foraging](https://github.com/AgentTorch/foraging/new/master/models/foraging_birds)
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. SIGGRAPH Comput. Graph., 21(4), 25–34.
- Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. Theoretical Population Biology, 9(2), 129-136.
- Dayan, P. (1993). Improving generalization for temporal difference learning: The successor representation. Neural Computation, 5(4), 613-624.
