#=
April 7, 2025

Optimization of PBPK model using Nelder-Mead, Particle Swarm, and Differential Evolution
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
base_sol = loss_opt(θ_vec, p_fixed, true); # get the solution
lower_bound_vec = getdata(p_fixed[6]);
upper_bound_vec = getdata(p_fixed[7]);

##### Set up optimization problems
# This is the standard optimization problem.
optfn = OptimizationFunction(loss_opt, Optimization.AutoForwardDiff());
optprob = OptimizationProblem(optfn, θ_vec, p_fixed, 
                                lb = lower_bound_vec, 
                                ub = upper_bound_vec);


##### Nelder Mead optimization 
# (~10 seconds on M1 Macbook - Single core, 1 second after compilation)
p_optim_NM = solve(optprob, Optim.NelderMead());
p_optim_NM.stats;
best_parameters = ComponentArray(p_optim_NM.u, p_fixed[5]); # convert to component-array
loss_opt(best_parameters, p_fixed) # check the loss function
NM_sol = loss_opt(best_parameters, p_fixed, true); # get the solution
# jldsave("InitialFit_NM_040525.jld2", p_optim_NM = p_optim_NM, p_fixed = p_fixed);


##### PSO optimization. Does not dispatch to component-array.
# ~ 28 seconds on M1 Macbook - Single core.
pso = Optimization.solve(optprob,ParticleSwarm(n_particles = 100),
                        maxtime = 120)

# PSO doesn't usually find the local minimum, so refine with LBFGS.
optprob_pso_extend = Optimization.OptimizationProblem(optfn, pso.u, p_fixed,
                                            lb = p_fixed[6], 
                                            ub = p_fixed[7]);

pso_lbfgs = Optimization.solve(optprob_pso_extend, LBFGS(), maxtime = 60);
best_parameters_pso = ComponentArray(pso_lbfgs.u, p_fixed[5]); # convert to component-array
loss(best_parameters_pso, p_fixed) # check the loss function
pso_lbfgs_sol = loss(best_parameters_pso, p_fixed, true); # get the solution


##### Using BBO Differential Evolution
# ~ 18 seconds on M1 Macbook - Single core.
DE = solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 100_000)
best_parameters_de = ComponentArray(DE.u, p_fixed[5]); # convert to component-array
loss(best_parameters_de, p_fixed) # check the loss function
DE_sol = loss(best_parameters_de, p_fixed, true); # get the solution


scatter(dat.time, dat.conc, label = "data");
plot!(NM_sol, idxs = :CP, label = "NM", linestyle = :dash)
plot!(pso_lbfgs_sol, idxs = :CP, label = "PSO-LBFGS")
plot!(DE_sol, idxs = :CP, label = "DE")
plot!(base_sol, idxs = :CP, label = "Base Params", linestyle = :dot)