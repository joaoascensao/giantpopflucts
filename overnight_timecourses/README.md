Data and analysis shown in Figure 4 (and associated supplemental figures)
eighthourtc_freqs.csv - raw data for 8-hour dense time course
ON_freqs.csv - raw data for 24-hour time course
timecourse_var_corr.py - calculate frequency variance over time and rank correlation with final time
chaos_analysis/aic.py - helper function to calculate AIC
chaos_analysis/variance_aic_fits.py - calculate AIC for fits of various curves to variance trajectory. Point estimates output in aic_point.csv, bootstrapped sampled in aic_boot.csv
chaos_analysis/rosenstein_method.py - use method modified from Rosenstein et al. (1993) to calculate Lyapnuov exponents from time course data, for different embedding dimensions and time lags. Uses shuffle-splitting cross-validation to calculate an out-of-sample mean-squared error.
chaos_analysis/cross_validation_results.py - plots cross-validation results
