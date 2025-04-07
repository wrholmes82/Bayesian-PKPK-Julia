#=
April 7, 2025

Tests and benchmarks loss functions for the PBPK model using naive 'remake' and 
a more optimized 'remake' approach. No optimization here, just testing the loss function.

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


##### Test the optimized loss function
f(x) = loss_opt(x , p_fixed);
f(θ_vec) # test the loss function
ForwardDiff.gradient(f, θ_vec) # test the gradient

@benchmark f(θ_vec) # test the loss function
@benchmark ForwardDiff.gradient(f, θ_vec) # test the gradient

##### Test the old loss function.
f_old(x) = loss(x , p_fixed);
f_old(θ) # test the loss function

@benchmark f_old(θ) # test the loss function
@benchmark ForwardDiff.gradient(f_old, θ) # test the gradient

