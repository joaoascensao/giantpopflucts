Data and analysis shown in Figure 5 (and associated supplemental figures)
tc_data.csv - raw data
calculate_fitness.py - calculates between-day logit displacement. Outputs fitness.csv
stan_model/linear_model_noncentered.py - noncentered model to infer effects of intrinsic and extrinsic decoupling noise and memory-like effects from sharing mother cultures. Outputs are posterior_{}.csv files.
stan_model/plot_variance_components.py - plot variance components
stan_model/plot_trajectories.py - plot all time course trajectories
stan_model/plot_alphas.py - plot logit displacements along with inferred day-wise extrinsic fitness effect
