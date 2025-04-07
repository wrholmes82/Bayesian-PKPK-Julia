# Bayesian PBPK 

The project is for educational purposes only. The intent is to learn to work with Julia's SciML ecosystem.

### Purpose
1. Construct a model using ModelingToolkit.jl. 
    - This was taken directly from https://github.com/metrumresearchgroup/CTSI2025-PBPK-in-Julia .

2. Fit a subset of the models parameters using NelderMeade, PSO, and Differential Evolution.

3. Use Turing.jl to perform parameter inference on the same problem.

### Notes on 'remake' in DifferentialEquations.jl
Appropriately using 'remake' is critical to efficient and correct optimization. This project provides two loss functions.

- One uses a convenient form of remake. This however is very slow in optimization loops. Further, it leads to unexpected behaviors in Turing.jl, particularly with multi-threading.

- The second uses a remake approach recommended by the SciML docs. It is 25x more efficient than the former in NelderMeade optimization and behaves well in Turing.jl.
    - https://docs.sciml.ai/ModelingToolkit/stable/examples/remake/
    - https://docs.sciml.ai/ModelingToolkit/stable/basics/FAQ/

### A few notes on compute time

- These results are for optimizing a subset of five parameters.
- All tests are on an M1 Macbook Pro plugged in.
- NelderMeade optimization requires ~10-15 seconds. PSO and DE ~20-30 seconds. All single core.
- Bayesian inferrence requires ~ 3-4 minutes to generate 5000 samples / thread with 6 threads.
