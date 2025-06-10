
import scipy.io
import numpy as np
pymc3_installed = False
try:
    import pymc3 as pm
    import arviz as az
    pymc3_installed = True
except ImportError:
    print("PyMC3 or ArviZ not installed. Skipping Bayesian modeling.")
    print("Please ensure your environment has PyMC3 and ArviZ installed to perform Bayesian analysis.")

if pymc3_installed:
    # --- Data Loading ---
    try:
        print("Loading 2016 CHM and GSMAP data...")
        chm_data_full = scipy.io.loadmat('/sandbox/datasets/CHMdata/CHM_2016.mat')['data']
        gsmap_data_full = scipy.io.loadmat('/sandbox/datasets/GSMAPdata/GSMAP_2016.mat')['data']
        print("Data loaded successfully.")

        # --- Data Preprocessing and Masking ---
        # Create the valid mask from CHM (non-NaN values)
        valid_mask = ~np.isnan(chm_data_full)

        # Apply the mask to both datasets
        chm_valid = chm_data_full[valid_mask]
        gsmap_valid = gsmap_data_full[valid_mask]

        # Ensure no remaining NaNs in the data used for the model (should be handled by previous steps)
        # If any NaNs slipped through (e.g., in GSMAP where CHM was valid), remove them
        combined_data = np.stack([chm_valid, gsmap_valid], axis=-1)
        combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

        chm_data = combined_data[:, 0]
        gsmap_data = combined_data[:, 1]

        num_valid_points = len(chm_data)
        print(f"Total valid data points for 2016 (after masking and NaN removal): {num_valid_points}")

        # --- Subsampling for MCMC Efficiency ---
        max_sample_size = 50000
        if num_valid_points > max_sample_size:
            print(f"Subsampling {max_sample_size} points for Bayesian modeling due to large dataset size ({num_valid_points}).")
            indices = np.random.choice(num_valid_points, max_sample_size, replace=False)
            chm_data_sampled = chm_data[indices]
            gsmap_data_sampled = gsmap_data[indices]
        else:
            chm_data_sampled = chm_data
            gsmap_data_sampled = gsmap_data
        
        print(f"Using {len(chm_data_sampled)} data points for Bayesian model training.")

        # --- Bayesian Linear Regression Model ---
        print("\n--- Starting Bayesian Linear Regression Model ---")
        with pm.Model() as basic_model:
            # Priors for the parameters
            alpha = pm.Normal('alpha', mu=0, sd=10) # Intercept
            beta = pm.Normal('beta', mu=0, sd=10)   # Slope for GSMAP
            sigma = pm.HalfNormal('sigma', sd=1) # Standard deviation of the likelihood

            # Expected value of outcome (linear model)
            mu = alpha + beta * gsmap_data_sampled

            # Likelihood (sampling distribution of observations)
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=chm_data_sampled)

            # MCMC Sampling
            # Tune: number of iterations for the sampler to adapt
            # Draws: number of posterior samples to draw
            # Chains: number of independent MCMC chains
            print("Sampling posterior distributions (this may take a few minutes)...")
            # Use `nuts` sampler as it's generally robust for continuous parameters
            trace = pm.sample(draws=1000, tune=1000, chains=2, random_seed=42, return_inferencedata=False)
            print("Sampling complete.")

        # --- Analyze Posterior Distributions ---
        print("\n--- Posterior Parameter Summaries ---")
        param_summary = az.summary(trace, var_names=['alpha', 'beta', 'sigma'])
        print(param_summary)

        # --- Quantify Predictive Uncertainty ---
        print("\n--- Quantifying Predictive Uncertainty ---")
        # Generate posterior predictive samples
        with basic_model:
            ppc = pm.sample_posterior_predictive(trace, var_names=['Y_obs'], samples=500, random_seed=42)

        # Calculate 2.5th, 50th (median), and 97.5th percentiles of predictions
        # These represent the 95% credible interval for predictions
        ppc_mean = ppc['Y_obs'].mean(axis=0)
        ppc_lower = np.percentile(ppc['Y_obs'], 2.5, axis=0)
        ppc_upper = np.percentile(ppc['Y_obs'], 97.5, axis=0)
        ppc_median = np.percentile(ppc['Y_obs'], 50, axis=0)

        print(f"Predictive Mean (first 10 samples): {ppc_mean[:10]}")
        print(f"Predictive 2.5th percentile (first 10 samples): {ppc_lower[:10]}")
        print(f"Predictive 97.5th percentile (first 10 samples): {ppc_upper[:10]}")
        print(f"Predictive Median (first 10 samples): {ppc_median[:10]}")

        # Calculate average width of the 95% predictive interval
        avg_predictive_interval_width = np.mean(ppc_upper - ppc_lower)
        print(f"\nAverage width of 95% Predictive Interval: {avg_predictive_interval_width:.4f}")
        print("This value provides a direct measure of the average uncertainty in the model's predictions.")

        print("\n--- Bayesian Uncertainty Study Complete ---")
        print("Parameter uncertainty is reflected in the 'sd' (standard deviation) and 'hdi_3%' / 'hdi_97%' (95% Highest Density Interval) columns of the parameter summary.")
        print("Predictive uncertainty is quantified by the width of the predictive intervals.")

    except FileNotFoundError as e:
        print(f"Error: One or more .mat files not found. Please ensure they are in the correct /sandbox/datasets/ subdirectories. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data loading or processing: {e}")
        import traceback
        traceback.print_exc()
else:
    print("PyMC3 or ArviZ not installed. Cannot perform Bayesian analysis.")
