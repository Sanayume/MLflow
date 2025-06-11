
import numpy as np
import scipy.io
import pandas as pd
from datetime import datetime, timedelta
import os
from scipy.ndimage import generic_filter, maximum_filter

# --- Configuration ---
MASK_PATH = '/sandbox/datasets/mask/mask.mat'
RAINFALL_DATA_DIR = '/sandbox/datasets/rainfalldata/'
OUTPUT_DIR = '/sandbox/outputs/长江特征工程/' # New directory for features
PRODUCTS = ['CMORPH', 'CHIRPS', 'GSMAP', 'IMERG', 'PERSIANN', 'SM2RAIN', 'CHM']
START_DATE = '2016-01-01'
END_DATE = '2020-12-31'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Step 1: Load Mask and Prepare Time Index ---")
# Load mask
mask_data = scipy.io.loadmat(MASK_PATH)['data']
# Changjiang mask: values >= 2
changjiang_mask = (mask_data >= 2)
N_lat, N_lon = changjiang_mask.shape
changjiang_indices_rows = changjiang_mask.nonzero()[0] # Get row indices of True values
changjiang_indices_cols = changjiang_mask.nonzero()[1] # Get col indices of True values
N_points = len(changjiang_indices_rows)

print(f"Mask loaded. Spatial shape: {N_lat}x{N_lon}. Yangtze River basin points: {N_points}")

# Prepare time index
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
N_days = len(date_range)
print(f"Time range from {START_DATE} to {END_DATE}. Total days: {N_days}")

# --- Time Features ---
print("--- Step 2: Generate and Save Time Features ---")
# Sin/Cos Day of Year
day_of_year = date_range.dayofyear
# Use 366 for leap year consistency to map days to a circle
sin_day_of_year = np.sin(2 * np.pi * day_of_year / 366.0).astype(np.float32)
cos_day_of_year = np.cos(2 * np.pi * day_of_year / 366.0).astype(np.float32)
np.save(os.path.join(OUTPUT_DIR, '长江_sin_day_of_year.npy'), sin_day_of_year)
np.save(os.path.join(OUTPUT_DIR, '长江_cos_day_of_year.npy'), cos_day_of_year)
print("Saved 长江_sin_day_of_year.npy and 长江_cos_day_of_year.npy")
del day_of_year, sin_day_of_year, cos_day_of_year

# Seasonal Dummies
# Assuming Northern Hemisphere seasons based on typical definitions
season_onehot_spring = ((date_range.month >= 3) & (date_range.month <= 5)).astype(np.int8)
season_onehot_summer = ((date_range.month >= 6) & (date_range.month <= 8)).astype(np.int8)
season_onehot_autumn = ((date_range.month >= 9) & (date_range.month <= 11)).astype(np.int8)
season_onehot_winter = ((date_range.month == 12) | (date_range.month <= 2)).astype(np.int8)
np.save(os.path.join(OUTPUT_DIR, '长江_season_onehot_spring.npy'), season_onehot_spring)
np.save(os.path.join(OUTPUT_DIR, '长江_season_onehot_summer.npy'), season_onehot_summer)
np.save(os.path.join(OUTPUT_DIR, '长江_season_onehot_autumn.npy'), season_onehot_autumn)
np.save(os.path.join(OUTPUT_DIR, '长江_season_onehot_winter.npy'), season_onehot_winter)
print("Saved seasonal one-hot features.")
del season_onehot_spring, season_onehot_summer, season_onehot_autumn, season_onehot_winter

# --- Process Product by Product for Raw, Lag, Diff, Sliding Window, Spatial ---
print("--- Step 3: Process Product by Product ---")
# This list will temporarily hold raw data for multi-product features
raw_product_features_list = []

for product_name in PRODUCTS:
    print(f"Processing product: {product_name}")
    product_file = os.path.join(RAINFALL_DATA_DIR, f"{product_name}data", f"{product_name}_2016_2020.mat")
    try:
        product_full_data = scipy.io.loadmat(product_file)['data'].astype(np.float32)
        # Ensure data dimensions match expected (time, lat, lon)
        if product_full_data.shape[0] != N_days or product_full_data.shape[1] != N_lat or product_full_data.shape[2] != N_lon:
            print(f"Warning: {product_name} data shape {product_full_data.shape} does not match expected ({N_days}, {N_lat}, {N_lon}). Skipping.")
            del product_full_data
            continue
    except FileNotFoundError:
        print(f"Error: {product_file} not found. Skipping product {product_name}.")
        continue
    except Exception as e:
        print(f"Error loading {product_file}: {e}. Skipping product {product_name}.")
        continue

    # --- Raw Feature ---
    # Extract only Yangtze basin points for raw data
    raw_feature_data_masked = product_full_data[:, changjiang_indices_rows, changjiang_indices_cols]
    np.save(os.path.join(OUTPUT_DIR, f'长江_raw_{product_name}.npy'), raw_feature_data_masked)
    print(f"Saved 长江_raw_{product_name}.npy")
    raw_product_features_list.append(raw_feature_data_masked) # Keep for multi-product features

    # --- Lag Features ---
    for lag in [1, 2, 3]:
        lag_feature = np.roll(raw_feature_data_masked, lag, axis=0)
        # Set initial 'lag' days to NaN as they don't have a preceding value
        lag_feature[:lag, :] = np.nan
        np.save(os.path.join(OUTPUT_DIR, f'长江_raw_{product_name}_lag{lag}.npy'), lag_feature)
        # print(f"Saved 长江_raw_{product_name}_lag{lag}.npy") # Suppress for brevity
        del lag_feature

    # --- Difference Features ---
    diff_1_feature = raw_feature_data_masked - np.roll(raw_feature_data_masked, 1, axis=0)
    diff_1_feature[0, :] = np.nan # First day has no previous value
    np.save(os.path.join(OUTPUT_DIR, f'长江_diff_1_raw_{product_name}.npy'), diff_1_feature)
    # print(f"Saved 长江_diff_1_raw_{product_name}.npy") # Suppress for brevity
    del diff_1_feature

    # --- Sliding Window Features ---
    for window in [3, 7, 15]:
        for stat_func, stat_name in [(np.nanmean, 'mean'), (np.nanstd, 'std'),
                                     (np.nanmax, 'max'), (np.nansum, 'sum'),
                                     (lambda x: np.nanmax(x) - np.nanmin(x) if x.size > 0 and not np.all(np.isnan(x)) else np.nan, 'range')]:
            # Apply rolling window along the time axis
            window_feature = np.full_like(raw_feature_data_masked, np.nan, dtype=np.float32)
            for i in range(window - 1, N_days):
                window_feature[i, :] = stat_func(raw_feature_data_masked[i-window+1 : i+1, :], axis=0)
            np.save(os.path.join(OUTPUT_DIR, f'长江_sw_{window}day_{stat_name}_of_raw_{product_name}.npy'), window_feature)
            # print(f"Saved 长江_sw_{window}day_{stat_name}_of_raw_{product_name}.npy") # Suppress for brevity
            del window_feature

    # --- Spatial Features (operate on full grid, then mask) ---
    print(f"Calculating spatial features for {product_name}...")
    # Neighbor 3x3 mean, std, max
    neighbor_3x3_mean = np.full_like(product_full_data, np.nan, dtype=np.float32)
    neighbor_3x3_std = np.full_like(product_full_data, np.nan, dtype=np.float32)
    neighbor_3x3_max = np.full_like(product_full_data, np.nan, dtype=np.float32)

    # Note: generic_filter can be slow. Consider optimizing or skipping if too slow.
    # For speed, we can use uniform_filter for mean, and pre-calculate for std, or use custom Cython/Numba.
    # For this task, generic_filter is acceptable for correctness.
    for t in range(N_days):
        day_data = product_full_data[t, :, :]
        # Use np.nan as cval for constant mode to propagate NaNs correctly
        # Minimum number of data points for calculation to avoid NaN output from filter on all-NaN window
        min_valid_for_mean_std = 1 # At least one valid point for mean/std
        neighbor_3x3_mean[t, :, :] = generic_filter(day_data, np.nanmean, size=(3,3), mode='constant', cval=np.nan)
        neighbor_3x3_std[t, :, :] = generic_filter(day_data, np.nanstd, size=(3,3), mode='constant', cval=np.nan)
        neighbor_3x3_max[t, :, :] = maximum_filter(day_data, size=(3,3), mode='constant', cval=np.nan)

    # Apply mask and save
    np.save(os.path.join(OUTPUT_DIR, f'长江_neighbor_3x3_mean_{product_name}.npy'), neighbor_3x3_mean[:, changjiang_indices_rows, changjiang_indices_cols])
    np.save(os.path.join(OUTPUT_DIR, f'长江_neighbor_3x3_std_{product_name}.npy'), neighbor_3x3_std[:, changjiang_indices_rows, changjiang_indices_cols])
    np.save(os.path.join(OUTPUT_DIR, f'长江_neighbor_3x3_max_{product_name}.npy'), neighbor_3x3_max[:, changjiang_indices_rows, changjiang_indices_cols])
    print(f"Saved 长江_neighbor_3x3_mean/std/max_{product_name}.npy")

    # Center point vs. neighborhood mean difference
    # Need to reload or ensure raw_feature_data_masked is still available if it was del-ed
    # (It's still in raw_product_features_list for now)
    center_minus_neighbor_3x3_mean = raw_feature_data_masked - neighbor_3x3_mean[:, changjiang_indices_rows, changjiang_indices_cols]
    np.save(os.path.join(OUTPUT_DIR, f'长江_center_minus_neighbor_3x3_mean_{product_name}.npy'), center_minus_neighbor_3x3_mean)
    print(f"Saved 长江_center_minus_neighbor_3x3_mean_{product_name}.npy")

    # Spatial Gradients
    gradient_magnitude = np.full_like(product_full_data, np.nan, dtype=np.float32)
    gradient_direction = np.full_like(product_full_data, np.nan, dtype=np.float32)

    for t in range(N_days):
        day_data = product_full_data[t, :, :]
        # np.gradient handles NaNs by propagating them
        grad_y, grad_x = np.gradient(day_data)
        gradient_magnitude[t, :, :] = np.sqrt(grad_y**2 + grad_x**2)
        gradient_direction[t, :, :] = np.arctan2(grad_y, grad_x) # Radians

    # Apply mask and save
    np.save(os.path.join(OUTPUT_DIR, f'长江_gradient_magnitude_{product_name}.npy'), gradient_magnitude[:, changjiang_indices_rows, changjiang_indices_cols])
    np.save(os.path.join(OUTPUT_DIR, f'长江_gradient_direction_{product_name}.npy'), gradient_direction[:, changjiang_indices_rows, changjiang_indices_cols])
    print(f"Saved 长江_gradient_magnitude/direction_{product_name}.npy")

    # Free memory for full product data and spatial intermediates
    del product_full_data, neighbor_3x3_mean, neighbor_3x3_std, neighbor_3x3_max, gradient_magnitude, gradient_direction, center_minus_neighbor_3x3_mean
    print(f"Finished spatial features for {product_name}. Memory cleared.")

# --- Multi-Product Synergy Features ---
print("--- Step 4: Calculate Multi-Product Synergy Features ---")
if len(raw_product_features_list) == len(PRODUCTS):
    # Stack all raw product features for multi-product calculations
    combined_data_masked = np.stack(raw_product_features_list, axis=-1) # Shape: (N_days, N_points, N_products)
    print(f"Combined data shape for synergy features: {combined_data_masked.shape}")

    mean_all_products = np.nanmean(combined_data_masked, axis=-1).astype(np.float32)
    std_all_products = np.nanstd(combined_data_masked, axis=-1).astype(np.float32)
    median_all_products = np.nanmedian(combined_data_masked, axis=-1).astype(np.float32)
    max_all_products = np.nanmax(combined_data_masked, axis=-1).astype(np.float32)
    min_all_products = np.nanmin(combined_data_masked, axis=-1).astype(np.float32)
    range_all_products = max_all_products - min_all_products

    # Handle cases where mean is zero for CV calculation
    cv_all_products = np.full_like(mean_all_products, np.nan, dtype=np.float32)
    # Avoid division by zero and NaN in mean
    non_zero_mean_mask = (mean_all_products != 0) & (~np.isnan(mean_all_products))
    cv_all_products[non_zero_mean_mask] = std_all_products[non_zero_mean_mask] / mean_all_products[non_zero_mean_mask]

    # Count raining products (threshold > 0.1 mm/d)
    count_raining_products = np.sum(combined_data_masked > 0.1, axis=-1).astype(np.int8)

    # Save synergy features
    np.save(os.path.join(OUTPUT_DIR, '长江_mean_all_products.npy'), mean_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_std_all_products.npy'), std_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_median_all_products.npy'), median_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_max_all_products.npy'), max_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_min_all_products.npy'), min_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_range_all_products.npy'), range_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_cv_all_products.npy'), cv_all_products)
    np.save(os.path.join(OUTPUT_DIR, '长江_count_raining_products.npy'), count_raining_products)
    print("Saved multi-product synergy features.")

    # --- Lag and Difference for Synergy Features ---
    print("Calculating lag and difference features for synergy products...")
    for lag in [1, 2, 3]:
        # Lag for mean_all_products
        lag_mean = np.roll(mean_all_products, lag, axis=0)
        lag_mean[:lag, :] = np.nan
        np.save(os.path.join(OUTPUT_DIR, f'长江_mean_all_products_lag{lag}.npy'), lag_mean)
        del lag_mean

        # Lag for std_all_products
        lag_std = np.roll(std_all_products, lag, axis=0)
        lag_std[:lag, :] = np.nan
        np.save(os.path.join(OUTPUT_DIR, f'长江_std_all_products_lag{lag}.npy'), lag_std)
        del lag_std

        # Lag for count_raining_products
        lag_count = np.roll(count_raining_products, lag, axis=0)
        lag_count[:lag, :] = np.nan
        np.save(os.path.join(OUTPUT_DIR, f'长江_count_raining_products_lag{lag}.npy'), lag_count)
        del lag_count
    print("Saved lag features for synergy products.")

    # Lag1-Lag2 difference for mean_all_products
    mean_lag1 = np.roll(mean_all_products, 1, axis=0)
    mean_lag1[0, :] = np.nan
    mean_lag2 = np.roll(mean_all_products, 2, axis=0)
    mean_lag2[:2, :] = np.nan
    lag1_lag2_mean_diff = (mean_lag1 - mean_lag2).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_lag1_lag2_mean_diff.npy'), lag1_lag2_mean_diff)
    print("Saved 长江_lag1_lag2_mean_diff.npy")
    del mean_lag1, mean_lag2, lag1_lag2_mean_diff

    # Difference for mean_all_products
    diff_1_mean_all_products = (mean_all_products - np.roll(mean_all_products, 1, axis=0)).astype(np.float32)
    diff_1_mean_all_products[0, :] = np.nan
    np.save(os.path.join(OUTPUT_DIR, '长江_diff_1_mean_all_products.npy'), diff_1_mean_all_products)
    print("Saved 长江_diff_1_mean_all_products.npy")
    del diff_1_mean_all_products

    # --- Sliding Window for Synergy Features ---
    print("Calculating sliding window features for synergy products...")
    for window in [3, 7, 15]:
        for stat_func, stat_name in [(np.nanmean, 'mean'), (np.nanstd, 'std'),
                                     (np.nanmax, 'max'), (np.nansum, 'sum'),
                                     (lambda x: np.nanmax(x) - np.nanmin(x) if x.size > 0 and not np.all(np.isnan(x)) else np.nan, 'range')]:
            window_feature_mean = np.full_like(mean_all_products, np.nan, dtype=np.float32)
            for i in range(window - 1, N_days):
                window_feature_mean[i, :] = stat_func(mean_all_products[i-window+1 : i+1, :], axis=0)
            np.save(os.path.join(OUTPUT_DIR, f'长江_sw_{window}day_{stat_name}_of_mean_all_products.npy'), window_feature_mean)
            del window_feature_mean
    print("Saved sliding window features for synergy products.")

    # --- Weak Signal Enhancement / Fuzziness Features ---
    print("Calculating weak signal enhancement features...")
    # Distance to threshold 0.1mm
    distance_to_threshold_0_1mm = np.abs(mean_all_products - 0.1).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_distance_to_threshold_0_1mm.npy'), distance_to_threshold_0_1mm)
    print("Saved 长江_distance_to_threshold_0_1mm.npy")
    del distance_to_threshold_0_1mm

    # Low intensity threshold for conditional features
    low_intensity_threshold = 0.5 # mm/d
    mean_low_mask = (mean_all_products < low_intensity_threshold) # boolean mask for points where mean is low

    # Conditional standard deviation (std_all_products_if_mean_low)
    # Create a temporary array where values are nan if mean is not low, otherwise original combined_data_masked values
    temp_combined_for_conditional_std = np.where(mean_low_mask[:, :, np.newaxis], combined_data_masked, np.nan)
    std_all_products_if_mean_low = np.nanstd(temp_combined_for_conditional_std, axis=-1).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_std_all_products_if_mean_low.npy'), std_all_products_if_mean_low)
    print("Saved 长江_std_all_products_if_mean_low.npy")
    del temp_combined_for_conditional_std

    # Conditional CV (cv_if_mean_low)
    cv_if_mean_low = np.full_like(std_all_products_if_mean_low, np.nan, dtype=np.float32)
    # Only calculate where mean_all_products is low AND not zero
    non_zero_mean_low_mask = (mean_low_mask & (mean_all_products != 0) & (~np.isnan(mean_all_products)))
    cv_if_mean_low[non_zero_mean_low_mask] = std_all_products_if_mean_low[non_zero_mean_low_mask] / mean_all_products[non_zero_mean_low_mask]
    np.save(os.path.join(OUTPUT_DIR, '长江_cv_if_mean_low.npy'), cv_if_mean_low)
    print("Saved 长江_cv_if_mean_low.npy")

    # Fraction of products in 0 to 0.5mm range
    fraction_products_in_0_to_0_5mm_range = (np.sum((combined_data_masked > 0) & (combined_data_masked <= 0.5), axis=-1) / len(PRODUCTS)).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_fraction_products_in_0_to_0_5mm_range.npy'), fraction_products_in_0_to_0_5mm_range)
    print("Saved 长江_fraction_products_in_0_to_0_5mm_range.npy")
    del fraction_products_in_0_to_0_5mm_range

    # Free memory for synergy feature calculation
    del combined_data_masked, mean_all_products, std_all_products, median_all_products, max_all_products, min_all_products, range_all_products, cv_all_products, count_raining_products
    del raw_product_features_list # Ensure this is cleared
    print("Cleared memory for multi-product synergy features.")

else:
    print("Not all product raw data was loaded successfully. Skipping multi-product synergy features and dependent features.")


# --- High-Order Interaction Features ---
print("--- Step 5: Calculate High-Order Interaction Features ---")
# Reload necessary features as they might have been cleared
try:
    std_all_products_loaded = np.load(os.path.join(OUTPUT_DIR, '长江_std_all_products.npy')).astype(np.float32)
    sin_day_of_year_loaded = np.load(os.path.join(OUTPUT_DIR, '长江_sin_day_of_year.npy')).astype(np.float32)
    count_raining_products_loaded = np.load(os.path.join(OUTPUT_DIR, '长江_count_raining_products.npy')).astype(np.int8)
    std_all_products_if_mean_low_loaded = np.load(os.path.join(OUTPUT_DIR, '长江_std_all_products_if_mean_low.npy')).astype(np.float32)
    cv_all_products_loaded = np.load(os.path.join(OUTPUT_DIR, '长江_cv_all_products.npy')).astype(np.float32)

    # Reshape sin_day_of_year to (N_days, 1) for broadcasting with (N_days, N_points) arrays
    sin_day_of_year_reshaped = sin_day_of_year_loaded[:, np.newaxis]

    # product_std_times_sin_day = std_all_products * sin_day_of_year
    product_std_times_sin_day = (std_all_products_loaded * sin_day_of_year_reshaped).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_product_std_times_sin_day.npy'), product_std_times_sin_day)
    print("Saved 长江_product_std_times_sin_day.npy")
    del product_std_times_sin_day

    # rain_count_std_interaction = count_raining_products * std_all_products
    rain_count_std_interaction = (count_raining_products_loaded * std_all_products_loaded).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_rain_count_std_interaction.npy'), rain_count_std_interaction)
    print("Saved 长江_rain_count_std_interaction.npy")
    del rain_count_std_interaction

    # low_intensity_std_times_cv = std_all_products_if_mean_low * cv_all_products
    low_intensity_std_times_cv = (std_all_products_if_mean_low_loaded * cv_all_products_loaded).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, '长江_low_intensity_std_times_cv.npy'), low_intensity_std_times_cv)
    print("Saved 长江_low_intensity_std_times_cv.npy")
    del low_intensity_std_times_cv

    del std_all_products_loaded, sin_day_of_year_loaded, sin_day_of_year_reshaped, count_raining_products_loaded, std_all_products_if_mean_low_loaded, cv_all_products_loaded
    print("Cleared memory for high-order interaction features.")

except FileNotFoundError as e:
    print(f"Skipping some high-order interaction features due to missing dependency: {e}")
except Exception as e:
    print(f"An error occurred during high-order interaction features calculation: {e}")

print("--- Feature Engineering Complete ---")
print(f"All generated .npy files are saved in: {OUTPUT_DIR}")
