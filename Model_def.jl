# using Pkg; 
# Pkg.activate("./PBPK_MTK_Env/");

# using ModelingToolkit

# using ModelingToolkit, Unitful, DifferentialEquations


# Taken directly from https://github.com/metrumresearchgroup/CTSI2025-PBPK-in-Julia
PBPK = function(; name)
	@independent_variables t, [description = "time"]
	Dt = Differential(t)

	pars = @parameters begin
		# volumes (L); source: https://www.ncbi.nlm.nih.gov/pubmed/14506981
		Vad = 18.2
  		Vbo = 10.5 
  		Vbr = 1.45
  		Vgu = 0.65
  		Vhe = 0.33
  		Vki = 0.31
  		Vli = 1.8
  		Vlu = 0.5
  		Vmu = 29
  		Vsp = 0.15
  		Vbl = 5.6

		# blood flows (L/h); Cardiac output = 6.5 (L/min); source: https://www.ncbi.nlm.nih.gov/pubmed/14506981 
		Qad = 0.05*6.5*60
  		Qbo = 0.05*6.5*60
  		Qbr = 0.12*6.5*60
  		Qgu = 0.15*6.5*60 
  		Qhe = 0.04*6.5*60
  		Qki = 0.19*6.5*60
  		Qmu = 0.17*6.5*60
  		Qsp = 0.03*6.5*60
  		Qha = 0.065*6.5*60  
  		Qlu = 6.5*60 

		# partition coefficients estimated by Poulin and Theil method https://jpharmsci.org/article/S0022-3549(16)30889-9/fulltext
  		Kpad = 9.89   # adipose:plasma
  		Kpbo = 7.91   # bone:plasma
  		Kpbr = 7.35   # brain:plasma
  		Kpgu = 5.82   # gut:plasma
  		Kphe = 1.95   # heart:plasma
  		Kpki = 2.9    # kidney:plasma
  		Kpli = 4.66   # liver:plasma
  		Kplu = 0.83   # lungs:plasma
  		Kpmu = 2.94   # muscle:plasma; optimized
  		Kpsp = 2.96   # spleen:plasma
  		Kpre = 4      # calculated as average of non adipose Kps
  		BP = 1.0      # blood:plasma ratio

		# other parameters
  		WEIGHT = 73
  		ka = 0.849  #2.0  #0.849   # absorption rate constant (/hr) 
  		fup = 0.42   # fraction of unbound drug in plasma
  
  		# in vitro hepatic clearance parameters http://dmd.aspetjournals.org/content/38/1/25.long
  		fumic = 0.711 # fraction of unbound drug in microsomes
  		MPPGL = 30.3  # adult mg microsomal protein per g liver (mg/g)
  		VmaxH = 40    # adult hepatic Vmax (pmol/min/mg)
  		KmH = 9.3     # adult hepatic Km (uM)
  
  		# renal clearance  https://link.springer.com/article/10.1007%2Fs40262-014-0181-y
  		CL_Ki = 0.096 # (L/hr) renal clearance
	end

	vars = @variables begin
		GUTLUMEN(t) = 200.0 
		GUT(t) = 0.0 
		ADIPOSE(t) = 0.0
		BRAIN(t) = 0.0
		HEART(t) = 0.0
		BONE(t) = 0.0
  		KIDNEY(t) = 0.0
		LIVER(t) = 0.0
		LUNG(t) = 0.0
		MUSCLE(t) = 0.0
		SPLEEN(t) = 0.0
		REST(t) = 0.0
  		ART(t) = 0.0
		VEN(t) = 0.0
		CP(t)
	end

	# additional volume derivations
  	Vve = 0.705*Vbl  # venous blood
    Var = 0.295*Vbl  # arterial blood
    Vre = WEIGHT - (Vli+Vki+Vsp+Vhe+Vlu+Vbo+Vbr+Vmu+Vad+Vgu+Vbl)  # volume of rest of the body compartment
  
    # additional blood flow derivation
    Qli = Qgu + Qsp + Qha
    Qre = Qlu - (Qli + Qki + Qbo + Qhe + Qmu + Qad + Qbr)
  
    # intrinsic hepatic clearance calculation
  	CL_Li = ((VmaxH/KmH)*MPPGL*Vli*1000*60*1e-6) / fumic  # (L/hr) hepatic clearance

	# Calculation of tissue drug concentrations (mg/L)
  	Cadipose = ADIPOSE/Vad
    Cbone = BONE/Vbo
    Cbrain = BRAIN/Vbr 
    Cheart = HEART/Vhe 
    Ckidney = KIDNEY/Vki
    Cliver = LIVER/Vli 
    Clung = LUNG/Vlu 
    Cmuscle = MUSCLE/Vmu
    Cspleen = SPLEEN/Vsp
    Crest = REST/Vre
    Carterial = ART/Var
    Cvenous = VEN/Vve
    Cgut = GUT/Vgu

	observed = [
		CP ~ Cvenous/BP
	]

	eqs = [
  		Dt(GUTLUMEN) ~ -ka*GUTLUMEN,
  		Dt(GUT) ~ ka*GUTLUMEN + Qgu*(Carterial - Cgut/(Kpgu/BP)),
  		Dt(ADIPOSE) ~ Qad*(Carterial - Cadipose/(Kpad/BP)),
  		Dt(BRAIN) ~ Qbr*(Carterial - Cbrain/(Kpbr/BP)),
  		Dt(HEART) ~ Qhe*(Carterial - Cheart/(Kphe/BP)),
  		Dt(KIDNEY) ~ Qki*(Carterial - Ckidney/(Kpki/BP)) - CL_Ki*(fup*Ckidney/(Kpki/BP)),
  		Dt(LIVER) ~ Qgu*(Cgut/(Kpgu/BP)) + Qsp*(Cspleen/(Kpsp/BP)) + Qha*(Carterial) - Qli*(Cliver/(Kpli/BP)) - CL_Li*(fup*Cliver/(Kpli/BP)),
  		Dt(LUNG) ~ Qlu*(Cvenous - Clung/(Kplu/BP)),
  		Dt(MUSCLE) ~ Qmu*(Carterial - Cmuscle/(Kpmu/BP)),
  		Dt(SPLEEN) ~ Qsp*(Carterial - Cspleen/(Kpsp/BP)),
  		Dt(BONE) ~ Qbo*(Carterial - Cbone/(Kpbo/BP)),
  		Dt(REST) ~ Qre*(Carterial - Crest/(Kpre/BP)),
  		Dt(VEN) ~ Qad*(Cadipose/(Kpad/BP)) + Qbr*(Cbrain/(Kpbr/BP)) +
    Qhe*(Cheart/(Kphe/BP)) + Qki*(Ckidney/(Kpki/BP)) + Qli*(Cliver/(Kpli/BP)) + 
    Qmu*(Cmuscle/(Kpmu/BP)) + Qbo*(Cbone/(Kpbo/BP)) + Qre*(Crest/(Kpre/BP)) - Qlu*Cvenous,
  		Dt(ART) ~ Qlu*(Clung/(Kplu/BP) - Carterial)
	]

	ODESystem(eqs, t, vars, pars; name=name, observed=observed)
end

"""
    loss_opt(x, p , pred = false)

x = parameter vector for evaluation

p = (odeprob , dat , reset_param_fun! , diffcache , tunable_recipe, lower_bound, upper_bound, label_order);

# Note that 'p' needs specific ordeing determined by 'build_PBPK_problem'.

# pred = true will return the solution instead of the loss. Useful for Bayes and plotting.
"""
function loss_opt(x, p , pred = false)

	odeprob = p[1]; # ODEProblem stored as parameters to avoid using global variables

	dat = p[2];
	timesteps = dat.time;
	truth = dat.conc;

	setter = p[3];
	diffcache = p[4];

    ps = parameter_values(odeprob) # obtain the parameter object from the problem

    # get an appropriately typed preallocated buffer to store the `x` values in
    buffer = get_tmp(diffcache, x)
    # copy the current values to this buffer
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    # create a copy of the parameter object with the buffer
    ps = replace(Tunable(), ps, buffer)

	# remake the problem, passing in our new parameter object
    setter(ps, x)
    newprob = remake(odeprob; p = ps)

    sol = solve(newprob, AutoTsit5(Rosenbrock23()); saveat = timesteps)

	if pred; return sol; end

    predicted = sol[:CP]
    # return sum((truth .- predicted) .^ 2) / length(truth)
	return sum(abs2, truth .- predicted)
end

"""
    loss(x, p , pred = false)

x = parameter vector for evaluation

p = (odeprob , dat , reset_param_fun! , diffcache , tunable_recipe, lower_bound, upper_bound, label_order);

pred = true will return the solution instead of the loss. Useful for Bayes and plotting.

# Note that 'p' needs specific ordeing determined by 'build_PBPK_problem'.

# Taken directly from https://github.com/metrumresearchgroup/CTSI2025-PBPK-in-Julia
# Don't use if you can avoid.
"""
function loss(x, p, pred=false)
    # extract info
	odeprob = p[1] # ODEProblem stored as parameters to avoid using global variables
	dat = p[2]
    # remake the problem, passing in our new parameter object
	p_assign = Dict([:ka => x.ka, 
					:Kpmu => x.Kpmu, 
					:Kpli => x.Kpli, 
					:Kpad => x.Kpad, 
					:BP => x.BP])
	

    newprob = remake(odeprob; p = p_assign)
    sol = solve(newprob, Tsit5(), saveat = dat.time);

    if pred; return sol; end

	return sum(abs2, dat.conc .- sol[:CP])
end


"""
    build_PBPK_problem(dat)

This function builds the PBPK ODEProblem from the MTK model 
along with all structures needed to efficiently solve it in an optimization loop.

    dat = data to be compared against.

Returns a tuple of the form:
	(odeprob , dat , reset_param_fun! , diffcache , tunable_recipe, lower_bound, upper_bound, label_order);

	- odeprob: ODEProblem object
	- dat: data to be compared against
	- reset_param_fun!: internal function to reset the parameters in the ODEProblem during optimization
	- diffcache: DiffCache object for avoiding allocations and ensuring dual typing
	- tunable_recipe: recipe for reconstructing the component-array from a vector
	- lower_bound: lower bound for the parameters
	- upper_bound: upper bound for the parameters
	- label_order: order of the labels in the component-array

Returns θ, a component-array of the parameters that will be optimized.
Note that θ does not necessarily contain all model parameter, only those optimized.
	
# Note that all functions that solve the ODEProblem need to use this p_fixed ordering.
"""
function build_PBPK_problem(dat)
	##### Instantiate model and ODEProblem
	@mtkbuild odesys = PBPK();
	odeprob = ODEProblem(odesys, [], (0.0, 24.0), []);

	##### Construct parameter component-array. 
	# Also construct a few utilities for accessing and constructing the component-array.
	θ = ComponentArray(ka = odeprob.ps[:ka], 
						Kpmu = odeprob.ps[:Kpmu], 
						Kpli = odeprob.ps[:Kpli], 
						Kpad = odeprob.ps[:Kpad], 
						BP = odeprob.ps[:BP]);

	label_order = ComponentArrays.labels(θ); # Provides the parameter order in the CA.
	tunable_recipe = getaxes(θ);             # Recipe for reconstructing the component-array from a vector.

	# Construct a function that can efficiently replace parameters in the ODEProblem.
	# This will be required in the loss function for updating the ODEProblem during optimizatin.
	# Make sure the parameter order is the same as the one in the component array.
	# Using Symbol.(label_order) is important to ensure the order is correct.
	# reset_param_fun! = setp(odeprob, [:ka, :Kpmu, :Kpli, :Kpad, :BP]); # Not used. More error prone.
	reset_param_fun! = setp(odeprob, Symbol.(label_order));

	##### Constuct lower and upper bounds
	lower_bound = ComponentArray(ka = 0.1, Kpmu = 0.01, Kpli = 0.1, Kpad = 0.1,BP = 0.5); 
	upper_bound = ComponentArray(ka = 5.0, Kpmu = 20.0, Kpli = 20.0, Kpad = 30.0,BP = 2.0);

	# Ensure the Component Array ordering of the bounds matches the initial state
	if ~(ComponentArrays.labels(θ) == ComponentArrays.labels(lower_bound))
		error("Lower bound array ordering does not match previous parameter order.");
	end

	if ~(ComponentArrays.labels(θ) == ComponentArrays.labels(upper_bound))
		error("Upper bound array ordering does not match previous parameter order.");
	end
	

	# lower_bound_vec = getdata(lower_bound);
	# upper_bound_vec = getdata(upper_bound);

	# `DiffCache` to avoid allocations.
	# `copy` prevents the buffer stored by `DiffCache` from aliasing the one in
	# `parameter_values(odeprob)`.
	diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(odeprob))[1]));

	p_fixed =(odeprob , dat , reset_param_fun! , diffcache , tunable_recipe, lower_bound, upper_bound, label_order);

	return p_fixed, θ
end