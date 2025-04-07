#=
April 7, 2025

Using Turing to perform Bayesian inference on the PBPK model. Takes 3-4 minutes to sample.

Note that I prefer to use Turing.@addlogprob! to provide Turing the likelihood.
I find this is more flexible and easier to debug than using the ~ operator.

Bootstrapped with Nelder-Mead.

Checked with a posterior predictive check.
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
using Turing, StatsPlots

using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values, state_values
using SciMLStructures: Tunable, canonicalize, replace, replace!
using PreallocationTools

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
# Just used to get initial parameters. I center the priors at this.
# Though note this is bad practice. Don't do this. Just for testing.
p_optim_NM = solve(optprob, Optim.NelderMead());
p_optim_NM.stats;
best_parameters = ComponentArray(p_optim_NM.u, p_fixed[5]); # convert to component-array
loss_opt(best_parameters, p_fixed) # check the loss function

##### Build turing model
@model function Bayes_PBPK_DistBased(p_const, p_optim)
    # odeprob = p_const[1];
    dat = p_const[2];
    lower_bound = p_const[6];
    upper_bound = p_const[7];
    label_order = p_const[8];

    ka ~ truncated(LogNormal(log(p_optim.ka)), lower_bound.ka, upper_bound.ka)
    Kpmu ~ truncated(LogNormal(log(p_optim.Kpmu)), lower_bound.Kpmu, upper_bound.Kpmu)
    Kpli ~ truncated(LogNormal(log(p_optim.Kpli)), lower_bound.Kpli, upper_bound.Kpli)
    Kpad ~ truncated(LogNormal(log(p_optim.Kpad)), lower_bound.Kpad, upper_bound.Kpad)
    BP ~ Uniform(lower_bound.BP, upper_bound.BP)
    σ ~ truncated(Normal(1.0,0.5), 0.0, Inf)

    p = ComponentArray(ka = ka, Kpmu = Kpmu, Kpli = Kpli, Kpad = Kpad, BP = BP);

    # Check are constructing the component-array with the correct order.
    if ~(ComponentArrays.labels(p) == label_order)
        error("Parameter array ordering does not match build_PBPK_problem .");
    end

    p_vec = getdata(p);

    sol = loss_opt(p_vec, p_const, true);
    sol_interp = sol(dat.time)
    conc_comp = sol_interp[:CP]
    conc_data = dat.conc

    # A catch in case exploring bad parameter regions that generate negative concentrations.
    if minimum(conc_comp) < 0.0
        Turing.@addlogprob! -Inf
        return nothing
    end

    # Construct 1D normal distribution for the log likelihood.
    normal_dist = Normal(0.0,σ)

    # Calculate the log likelihood.
    # I prefer to control my own distributions and logpdf rather than use Turings ~ assignment.
    Turing.@addlogprob! 1.0*sum(logpdf.(normal_dist , conc_data .- conc_comp))

    return nothing
end

model = Bayes_PBPK_DistBased(p_fixed, best_parameters);

# Sampling takes about 3-4 minutes. Be sure at least N threads are available.
chain_NUTS = sample(model, NUTS(), MCMCThreads() , 5000 , 6 , discard_initial = 1000);

# Approcimately 20-25 minutes. Doesn't multithread well. Doesn't work well either. Bad posterior.
# chain_PG1 = sample(model, PG(20), MCMCThreads(), 5000, 4, discard_initial = 1000);

chain = chain_NUTS;

##### Posterior predictive check
plot(; legend=false);
posterior_samples = sample(chain[[:ka, :Kpmu, :Kpli, :Kpad, :BP]], 300; replace=false);
loss_vals = [];
for p in eachrow(Array(posterior_samples))

    sol_p = loss_opt(p, p_fixed, true);
    # sol_p = loss_for_Turing_vec(p,p_const,true);

    push!(loss_vals, loss_opt(p, p_fixed, false));
    plot!(sol_p, idxs = :CP; alpha=0.1, color="#BBBBBB");
end

plot!(loss_opt(best_parameters,p_fixed,true), idxs = :CP, 
    label = "Max Likelihood Nelder Mead", color = "black", linewidth=2);
scatter!(dat.time, dat.conc, label = "data")

histogram(loss_vals)
