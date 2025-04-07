#=
April 7, 2025

Tests and benchmarks NelderMeade optimization for the PBPK model using naive 'remake' and 
a more optimized 'remake' approach.

The naive remake approach is from https://github.com/metrumresearchgroup/CTSI2025-PBPK-in-Julia

The updated remake approach is from https://docs.sciml.ai/ModelingToolkit/stable/examples/remake/
=#

using Pkg; 
Pkg.activate("./PBPK_MTK_Env/");

using DifferentialEquations, ModelingToolkit, Unitful
using OrdinaryDiffEq # Don't use this directly, but it is required to load a saved ODEProblem.
using Plots 
using CSV, Tidier
using ComponentArrays, Optimization, OptimizationOptimJL
using OptimizationBBO, OptimizationOptimisers
using Random, Distributions
using ForwardDiff
using JLD2
using BenchmarkTools

using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values, state_values
using SciMLStructures: Tunable, canonicalize, replace, replace!
using PreallocationTools

# Pulls in the MTK model, loss functions, and a utility function to build the ODEProblem
# and supporting structures / functions for optimization.
include("Model_def.jl");

dat = CSV.read("shi2010-fig1.csv", DataFrame);

p_fixed, θ = build_PBPK_problem(dat);
θ_vec = getdata(θ);  # Extracting vector of parameters from the component-array.

##### Set up optimization problems
# This is the standard optimization problem.
optfn = OptimizationFunction(loss_opt, Optimization.AutoForwardDiff());
optprob = OptimizationProblem(optfn, θ_vec, p_fixed, 
                                lb = p_fixed[6], 
                                ub = p_fixed[7]);


##### Nelder Mead optimization 
# (~10 seconds on M1 Macbook - Single core, 1 second after compilation)
p_optim_NM = solve(optprob, Optim.NelderMead());
p_optim_NM.stats;
best_parameters = ComponentArray(p_optim_NM.u, p_fixed[5]); # convert to component-array
loss_opt(best_parameters, p_fixed) # check the loss function
# jldsave("InitialFit_NM_040425.jld2", p_optim_NM = p_optim_NM, p_fixed = p_fixed);


##### Set up old optimization problems
# (215 seconds on M1 Macbook - Single core, same after compilation)
optfn_old = OptimizationFunction(loss, Optimization.AutoForwardDiff());
optprob_old = OptimizationProblem(optfn_old, θ, p_fixed, 
                                lb = p_fixed[6], 
                                ub = p_fixed[7]);

p_optim_NM_old = solve(optprob_old, Optim.NelderMead())
best_parameters_old = p_optim_NM_old.u; # convert to component-array
