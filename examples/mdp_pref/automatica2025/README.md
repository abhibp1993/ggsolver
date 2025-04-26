# How to run?

Three steps:
1. Generate flattened MDP graph: `model_gen.py`
   - Change CONFIG as you wish.
   - Run the file to generate flattened product MDP graph, 
       stochastic ordering vector, and multi-objective function 
       over product MDP states.
   > You may skip this part if `model.pkl` is already available.

2. Compute policy: `run_solver.py`
   - The solutions are computed and stored on `solutions.pkl`.
   
3. Run simulator: `sim.py`
   - Simulate `n_runs`-many runs to generate probability distribution by 
    running the synthesized policy. The output prints theoretically 
    expected and observed probability distributions. 