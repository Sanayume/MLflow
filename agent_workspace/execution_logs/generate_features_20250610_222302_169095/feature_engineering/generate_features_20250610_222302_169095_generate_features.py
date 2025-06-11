
import numpy as np
import scipy.io
import datetime
import gc
import os
import scipy.ndimage

# --- Constants ---
DATA_PATH = '/sandbox/datasets/rainfalldata/'
MASK_PATH = '/sandbox/datasets/mask/mask.mat'
OUTPUT_DIR = '/sandbox/outputs/features/'
PRODUCTS = ['CMORPH', 'CHIRPS', 'GSMAP', 'IMERG', 'PERSIANN', 'SM2RAIN', 'CHM']
START_DATE = datetime.date(2016, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
RAINFALL_THRESHOLD = 0.1 # mm/d for 'raining' definition
LOW_RAINFALL_THRESHOLD = 0.5 # mm/d for low intensity rainfall

print("Starting feature engineering process...")

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Ensured output directory exists: {OUTPUT_DIR}")

# --- Load Mask ---
print("Loading mask file...")
try:
    mask_mat = scipy.io.loadmat(MASK_PATH)
    mask_array = mask_mat['data']
    yangtze_mask = (mask_array >= 2) # Boolean mask for Yangtze River basin (values >= 2)
    num_masked_points = np.sum(yangtze_mask)
    print(f"Mask loaded. Number of masked points in Yangtze River basin: {num_masked_points}")
    del mask_mat, mask_array
    gc.collect()
except Exception as e:
    print(f"Error loading mask file: {e}")
    exit(1) # Exit if mask cannot be loaded, as it's critical

# --- Generate Dates ---
dates = []
current_date = START_DATE
while current_date <= END_DATE:
    dates.append(current_date)
    current_date += datetime.timedelta(days=1)
num_days = len(dates)
print(f"Total number of days to process: {num_days} from {START_DATE} to {END_DATE}")

# --- Load All Raw Data & Apply Mask (initial load for basic and synergy features) ---
# This array will hold masked data for all products: (num_days, num_masked_points, num_products)
all_raw_data_masked = np.zeros((num_days, num_masked_points, len(PRODUCTS)), dtype=np.float32)

print("\n--- Loading and masking all raw rainfall data ---")
for i, product in enumerate(PRODUCTS):
    print(f"  Loading data for product: {product}")
    file_path = f"{DATA_PATH}{product}_2016_2020.mat"
    try:
        product_data_mat = scipy.io.loadmat(file_path)
        product_full_data = product_data_mat['data'] # Expected shape: (num_days, 144, 256)

        if product_full_data.shape[0] != num_days:
            raise ValueError(f"Data for {product} has {product_full_data.shape[0]} days, expected {num_days}")
        if product_full_data.shape[1] != 144 or product_full_data.shape[2] != 256:
            raise ValueError(f"Data for {product} has shape {product_full_data.shape[1:]}, expected (144, 256)")

        # Apply mask to each day and store flattened
        for d in range(num_days):
            day_data = product_full_data[d, :, :]
            all_raw_data_masked[d, :, i] = day_data[yangtze_mask]
        print(f"  Finished loading and masking data for product: {product}")
        del product_data_mat, product_full_data # Release memory
        gc.collect()

    except FileNotFoundError:
        print(f"  Warning: File {file_path} not found. Skipping product {product}. Filling with NaNs.")
        all_raw_data_masked[:, :, i] = np.nan
    except Exception as e:
        print(f"  Error loading {product} data: {e}. Filling with NaNs.")
        all_raw_data_masked[:, :, i] = np.nan

# --- Basic Information Features (Raw Values) ---
print("\n--- Calculating Basic Information Features ---")
for i, product in enumerate(PRODUCTS):
    feature_name = f"长江_raw_{product}"
    np.save(f"{OUTPUT_DIR}{feature_name}.npy", all_raw_data_masked[:, :, i])
    print(f"  Saved {feature_name}.npy")
gc.collect()

# --- Multi-Product Synergy Features ---
print("\n--- Calculating Multi-Product Synergy Features ---")
mean_all_products = np.nanmean(all_raw_data_masked, axis=2)
np.save(f"{OUTPUT_DIR}长江_mean_all_products.npy", mean_all_products)
print("  Saved 长江_mean_all_products.npy")

std_all_products = np.nanstd(all_raw_data_masked, axis=2)
np.save(f"{OUTPUT_DIR}长江_std_all_products.npy", std_all_products)
print("  Saved 长江_std_all_products.npy")

median_all_products = np.nanmedian(all_raw_data_masked, axis=2)
np.save(f"{OUTPUT_DIR}长江_median_all_products.npy", median_all_products)
print("  Saved 长江_median_all_products.npy")

max_all_products = np.nanmax(all_raw_data_masked, axis=2)
np.save(f"{OUTPUT_DIR}长江_max_all_products.npy", max_all_products)
print("  Saved 长江_max_all_products.npy")

min_all_products = np.nanmin(all_raw_data_masked, axis=2)
np.save(f"{OUTPUT_DIR}长江_min_all_products.npy", min_all_products)
print("  Saved 长江_min_all_products.npy")

range_all_products = max_all_products - min_all_products
np.save(f"{OUTPUT_DIR}长江_range_all_products.npy", range_all_products)
print("  Saved 长江_range_all_products.npy")
del max_all_products, min_all_products, range_all_products # Release memory
gc.collect()

count_raining_products = np.sum(all_raw_data_masked > RAINFALL_THRESHOLD, axis=2)
np.save(f"{OUTPUT_DIR}长江_count_raining_products.npy", count_raining_products)
print("  Saved 长江_count_raining_products.npy")

cv_all_products = np.full_like(mean_all_products, np.nan)
non_zero_mean_idx = (mean_all_products != 0) & (~np.isnan(mean_all_products))
cv_all_products[non_zero_mean_idx] = std_all_products[non_zero_mean_idx] / mean_all_products[non_zero_mean_idx]
np.save(f"{OUTPUT_DIR}长江_cv_all_products.npy", cv_all_products)
print("  Saved 长江_cv_all_products.npy")
gc.collect()

# --- Temporal Dynamic Features ---
print("\n--- Calculating Temporal Dynamic Features ---")

# Cyclical Features (Day of Year)
day_of_year = np.array([d.timetuple().tm_yday for d in dates])
sin_day_of_year = np.sin(2 * np.pi * day_of_year / 366).reshape(-1, 1) # Max days in year is 366
cos_day_of_year = np.cos(2 * np.pi * day_of_year / 366).reshape(-1, 1)

np.save(f"{OUTPUT_DIR}长江_sin_day_of_year.npy", np.tile(sin_day_of_year, (1, num_masked_points)))
print("  Saved 长江_sin_day_of_year.npy")
np.save(f"{OUTPUT_DIR}长江_cos_day_of_year.npy", np.tile(cos_day_of_year, (1, num_masked_points)))
print("  Saved 长江_cos_day_of_year.npy")
del sin_day_of_year, cos_day_of_year
gc.collect()

# Seasonal Dummies
season_spring = np.array([1 if d.month in [3,4,5] else 0 for d in dates]).reshape(-1, 1)
season_summer = np.array([1 if d.month in [6,7,8] else 0 for d in dates]).reshape(-1, 1)
season_autumn = np.array([1 if d.month in [9,10,11] else 0 for d in dates]).reshape(-1, 1)
season_winter = np.array([1 if d.month in [12,1,2] else 0 for d in dates]).reshape(-1, 1)

np.save(f"{OUTPUT_DIR}长江_season_onehot_spring.npy", np.tile(season_spring, (1, num_masked_points)))
np.save(f"{OUTPUT_DIR}长江_season_onehot_summer.npy", np.tile(season_summer, (1, num_masked_points)))
np.save(f"{OUTPUT_DIR}长江_season_onehot_autumn.npy", np.tile(season_autumn, (1, num_masked_points)))
np.save(f"{OUTPUT_DIR}长江_season_onehot_winter.npy", np.tile(season_winter, (1, num_masked_points)))
print("  Saved seasonal one-hot features.")
del season_spring, season_summer, season_autumn, season_winter
gc.collect()

# Lag Features and Difference Features
print("  Calculating Lag and Difference Features...")
# Load necessary base features for lagging
mean_all_products_loaded = np.load(f"{OUTPUT_DIR}长江_mean_all_products.npy")
std_all_products_loaded = np.load(f"{OUTPUT_DIR}长江_std_all_products.npy")
count_raining_products_loaded = np.load(f"{OUTPUT_DIR}长江_count_raining_products.npy")


# Dictionary to hold data arrays to be lagged, including raw products
features_to_lag_data = {
    'mean_all_products': mean_all_products_loaded,
    'std_all_products': std_all_products_loaded,
    'count_raining_products': count_raining_products_loaded
}
for i, product in enumerate(PRODUCTS):
    features_to_lag_data[f'raw_{product}'] = all_raw_data_masked[:, :, i]

# Store lagged features for later use (e.g., differences)
lagged_features_cache = {}

for feat_name, data_array in features_to_lag_data.items():
    for lag in [1, 2, 3]:
        lagged_data = np.roll(data_array, shift=lag, axis=0)
        lagged_data[:lag, :] = np.nan # Fill initial lag values with NaN
        feature_output_name = f"长江_{feat_name}_lag{lag}"
        np.save(f"{OUTPUT_DIR}{feature_output_name}.npy", lagged_data)
        print(f"  Saved {feature_output_name}.npy")
        lagged_features_cache[feature_output_name] = lagged_data # Cache for differences
        # No del lagged_data here as it's cached, will be deleted later

    # Difference Features (t0 - t-1)
    if f"长江_{feat_name}_lag1" in lagged_features_cache:
        diff_data = data_array - lagged_features_cache[f"长江_{feat_name}_lag1"]
        np.save(f"{OUTPUT_DIR}长江_diff_1_{feat_name}.npy", diff_data)
        print(f"  Saved 长江_diff_1_{feat_name}.npy")
        del diff_data
        gc.collect()

# Lag differences (mean_all_products_lag1 - mean_all_products_lag2)
if f"长江_mean_all_products_lag1" in lagged_features_cache and \
   f"长江_mean_all_products_lag2" in lagged_features_cache:
    lag1_lag2_mean_diff = lagged_features_cache[f"长江_mean_all_products_lag1"] - \
                          lagged_features_cache[f"长江_mean_all_products_lag2"]
    np.save(f"{OUTPUT_DIR}长江_lag1_lag2_mean_diff.npy", lag1_lag2_mean_diff)
    print("  Saved 长江_lag1_lag2_mean_diff.npy")
    del lag1_lag2_mean_diff
    gc.collect()

del features_to_lag_data, lagged_features_cache, mean_all_products_loaded, std_all_products_loaded, count_raining_products_loaded # Clean up cache and loaded features
gc.collect()

# Sliding Window Statistics
print("  Calculating Sliding Window Statistics...")

def rolling_window_stats_custom(arr, window_size, stat_type):
    """Custom rolling window statistics for 2D arrays (time, points)."""
    if window_size <= 0:
        return np.full_like(arr, np.nan)
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        start_idx = max(0, i - window_size + 1)
        window_data = arr[start_idx : i + 1, :]
        if stat_type == 'mean':
            result[i, :] = np.nanmean(window_data, axis=0)
        elif stat_type == 'std':
            result[i, :] = np.nanstd(window_data, axis=0)
        elif stat_type == 'max':
            result[i, :] = np.nanmax(window_data, axis=0)
        elif stat_type == 'sum':
            result[i, :] = np.nansum(window_data, axis=0)
        elif stat_type == 'min':
            result[i, :] = np.nanmin(window_data, axis=0)
        elif stat_type == 'range':
            result[i, :] = np.nanmax(window_data, axis=0) - np.nanmin(window_data, axis=0)
        else:
            result[i, :] = np.nan
    return result

window_sizes = [3, 7, 15]
stats_types = ['mean', 'std', 'max', 'sum', 'min', 'range']
features_for_rolling = {
    'mean_all_products': np.load(f"{OUTPUT_DIR}长江_mean_all_products.npy")
}
for p_idx, p in enumerate(PRODUCTS):
    features_for_rolling[f'raw_{p}'] = all_raw_data_masked[:, :, p_idx] # Use the already loaded masked raw data

for feat_name, data_array in features_for_rolling.items():
    print(f"    Calculating rolling window features for {feat_name}...")
    for window in window_sizes:
        for stat in stats_types:
            rolled_data = rolling_window_stats_custom(data_array, window, stat)
            np.save(f"{OUTPUT_DIR}长江_sw_{window}day_{stat}_of_{feat_name}.npy", rolled_data)
            print(f"    Saved 长江_sw_{window}day_{stat}_of_{feat_name}.npy")
            del rolled_data
            gc.collect()
# Release the loaded data for rolling features
for arr in features_for_rolling.values():
    del arr
gc.collect()

# --- Weak Signal Enhancement / Fuzziness Features ---
print("\n--- Calculating Weak Signal Enhancement / Fuzziness Features ---")

# Reload needed variables for this section
mean_all_products_reloaded = np.load(f"{OUTPUT_DIR}长江_mean_all_products.npy")
std_all_products_reloaded = np.load(f"{OUTPUT_DIR}长江_std_all_products.npy")
cv_all_products_reloaded = np.load(f"{OUTPUT_DIR}长江_cv_all_products.npy")

# Distance to specific rainfall threshold (0.1mm)
distance_to_threshold_0_1mm = np.abs(mean_all_products_reloaded - RAINFALL_THRESHOLD)
np.save(f"{OUTPUT_DIR}长江_distance_to_threshold_0_1mm.npy", distance_to_threshold_0_1mm)
print("  Saved 长江_distance_to_threshold_0_1mm.npy")
del distance_to_threshold_0_1mm
gc.collect()

# Low intensity rainfall conditions
low_intensity_mask = mean_all_products_reloaded < LOW_RAINFALL_THRESHOLD

# Conditional std_all_products if mean is low
std_all_products_if_mean_low = np.full_like(std_all_products_reloaded, np.nan)
std_all_products_if_mean_low[low_intensity_mask] = std_all_products_reloaded[low_intensity_mask]
np.save(f"{OUTPUT_DIR}长江_std_all_products_if_mean_low.npy", std_all_products_if_mean_low)
print("  Saved 长江_std_all_products_if_mean_low.npy")
gc.collect()

# Fraction of products in 0 to 0.5mm range (uses all_raw_data_masked which is still in memory)
fraction_products_in_0_to_0_5mm_range = np.sum((all_raw_data_masked >= 0) & (all_raw_data_masked <= LOW_RAINFALL_THRESHOLD), axis=2) / len(PRODUCTS)
np.save(f"{OUTPUT_DIR}长江_fraction_products_in_0_to_0_5mm_range.npy", fraction_products_in_0_to_0_5mm_range)
print("  Saved 长江_fraction_products_in_0_to_0_5mm_range.npy")
del fraction_products_in_0_to_0_5mm_range
gc.collect()

# Conditional CV if mean is low
cv_if_mean_low = np.full_like(cv_all_products_reloaded, np.nan)
cv_if_mean_low[low_intensity_mask] = cv_all_products_reloaded[low_intensity_mask]
np.save(f"{OUTPUT_DIR}长江_cv_if_mean_low.npy", cv_if_mean_low)
print("  Saved 长江_cv_if_mean_low.npy")
del cv_if_mean_low, low_intensity_mask # Release memory for these two
gc.collect()

# --- Release large core data arrays before Spatial Features ---
print("\n--- Releasing large core data arrays to free up memory ---")
del all_raw_data_masked, mean_all_products_reloaded, std_all_products_reloaded, cv_all_products_reloaded
gc.collect()

# --- Spatial Correlation Features ---
print("\n--- Calculating Spatial Correlation Features ---")

# Define convolution kernels for 3x3 and 5x5 mean
kernel_3x3_mean = np.ones((3, 3)) / 9.0
kernel_5x5_mean = np.ones((5, 5)) / 25.0

for product_idx, product in enumerate(PRODUCTS):
    print(f"  Processing spatial features for product: {product}")
    file_path = f"{DATA_PATH}{product}_2016_2020.mat"
    try:
        product_data_mat = scipy.io.loadmat(file_path)
        product_full_data = product_data_mat['data'] # (num_days, 144, 256)
        # Ensure data dimensions match, otherwise skip
        if product_full_data.shape[0] != num_days or product_full_data.shape[1] != 144 or product_full_data.shape[2] != 256:
            print(f"    Skipping spatial features for {product} due to unexpected data shape: {product_full_data.shape}")
            del product_data_mat, product_full_data
            gc.collect()
            continue

        # Neighborhood Mean (3x3, 5x5)
        for kernel_size, kernel in [(3, kernel_3x3_mean), (5, kernel_5x5_mean)]:
            print(f"    Calculating {kernel_size}x{kernel_size} neighborhood mean for {product}")
            spatial_mean_features = np.zeros((num_days, num_masked_points), dtype=np.float32)
            for d in range(num_days):
                day_data = product_full_data[d, :, :]
                # Use mode='constant', cval=np.nan to handle NaNs at boundaries
                convolved_data = scipy.ndimage.convolve(day_data, kernel, mode='constant', cval=np.nan)
                spatial_mean_features[d, :] = convolved_data[yangtze_mask]
            np.save(f"{OUTPUT_DIR}长江_neighbor_{kernel_size}x{kernel_size}_mean_{product}.npy", spatial_mean_features)
            print(f"    Saved 长江_neighbor_{kernel_size}x{kernel_size}_mean_{product}.npy")
            del spatial_mean_features
            gc.collect()

        # Center point vs. neighborhood mean difference (for raw product)
        raw_product_masked_data = np.load(f"{OUTPUT_DIR}长江_raw_{product}.npy")
        if os.path.exists(f"{OUTPUT_DIR}长江_neighbor_3x3_mean_{product}.npy"):
            neighbor_3x3_mean_product = np.load(f"{OUTPUT_DIR}长江_neighbor_3x3_mean_{product}.npy")
            center_minus_neighbor_mean_product = raw_product_masked_data - neighbor_3x3_mean_product
            np.save(f"{OUTPUT_DIR}长江_center_minus_neighbor_3x3_mean_{product}.npy", center_minus_neighbor_mean_product)
            print(f"    Saved 长江_center_minus_neighbor_3x3_mean_{product}.npy")
            del center_minus_neighbor_mean_product, neighbor_3x3_mean_product
            gc.collect()
        del raw_product_masked_data
        gc.collect()

        # Neighborhood Std (3x3, 5x5)
        for kernel_size in [3, 5]:
            print(f"    Calculating {kernel_size}x{kernel_size} neighborhood std for {product}")
            spatial_std_features = np.zeros((num_days, num_masked_points), dtype=np.float32)
            size_tuple = (kernel_size, kernel_size)
            for d in range(num_days):
                day_data = product_full_data[d, :, :]
                convolved_std = scipy.ndimage.generic_filter(day_data, np.nanstd, size=size_tuple, mode='constant', cval=np.nan)
                spatial_std_features[d, :] = convolved_std[yangtze_mask]
            np.save(f"{OUTPUT_DIR}长江_neighbor_{kernel_size}x{kernel_size}_std_{product}.npy", spatial_std_features)
            print(f"    Saved 长江_neighbor_{kernel_size}x{kernel_size}_std_{product}.npy")
            del spatial_std_features
            gc.collect()

        # Neighborhood Max (3x3, 5x5)
        for kernel_size in [3, 5]:
            print(f"    Calculating {kernel_size}x{kernel_size} neighborhood max for {product}")
            spatial_max_features = np.zeros((num_days, num_masked_points), dtype=np.float32)
            size_tuple = (kernel_size, kernel_size)
            for d in range(num_days):
                day_data = product_full_data[d, :, :]
                convolved_max = scipy.ndimage.maximum_filter(day_data, size=size_tuple, mode='constant', cval=np.nan)
                spatial_max_features[d, :] = convolved_max[yangtze_mask]
            np.save(f"{OUTPUT_DIR}长江_neighbor_{kernel_size}x{kernel_size}_max_{product}.npy", spatial_max_features)
            print(f"    Saved 长江_neighbor_{kernel_size}x{kernel_size}_max_{product}.npy")
            del spatial_max_features
            gc.collect()

        # Spatial Gradient Features (Magnitude and Direction)
        print(f"    Calculating spatial gradient for {product}")
        gradient_magnitude = np.zeros((num_days, num_masked_points), dtype=np.float32)
        gradient_direction = np.zeros((num_days, num_masked_points), dtype=np.float32)

        for d in range(num_days):
            day_data = product_full_data[d, :, :]
            grad_y, grad_x = np.gradient(day_data)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)

            gradient_magnitude[d, :] = magnitude[yangtze_mask]
            gradient_direction[d, :] = direction[yangtze_mask]

        np.save(f"{OUTPUT_DIR}长江_gradient_magnitude_{product}.npy", gradient_magnitude)
        print(f"    Saved 长江_gradient_magnitude_{product}.npy")
        np.save(f"{OUTPUT_DIR}长江_gradient_direction_{product}.npy", gradient_direction)
        print(f"    Saved 长江_gradient_direction_{product}.npy")
        del gradient_magnitude, gradient_direction
        gc.collect()

        del product_full_data, product_data_mat # Release memory for current product's full data
        gc.collect()
    except Exception as e:
        print(f"  An error occurred during spatial feature calculation for {product}: {e}")
        continue

# --- High-Order Interaction Features ---
print("\n--- Calculating High-Order Interaction Features ---")

# Reload necessary base features for interaction
std_all_products_final = np.load(f"{OUTPUT_DIR}长江_std_all_products.npy")
cv_all_products_final = np.load(f"{OUTPUT_DIR}长江_cv_all_products.npy")
sin_day_of_year_broadcasted_final = np.load(f"{OUTPUT_DIR}长江_sin_day_of_year.npy")
std_all_products_if_mean_low_final = np.load(f"{OUTPUT_DIR}长江_std_all_products_if_mean_low.npy")
count_raining_products_final = np.load(f"{OUTPUT_DIR}长江_count_raining_products.npy")


# product_std_times_sin_day = std_all_products * sin_day_of_year
product_std_times_sin_day = std_all_products_final * sin_day_of_year_broadcasted_final
np.save(f"{OUTPUT_DIR}长江_product_std_times_sin_day.npy", product_std_times_sin_day)
print("  Saved 长江_product_std_times_sin_day.npy")
del product_std_times_sin_day, sin_day_of_year_broadcasted_final
gc.collect()

# low_intensity_std_times_cv = std_all_products_if_mean_low * cv_all_products
low_intensity_std_times_cv = std_all_products_if_mean_low_final * cv_all_products_final
np.save(f"{OUTPUT_DIR}长江_low_intensity_std_times_cv.npy", low_intensity_std_times_cv)
print("  Saved 长江_low_intensity_std_times_cv.npy")
del low_intensity_std_times_cv, std_all_products_if_mean_low_final
gc.collect()

# rain_count_std_interaction = count_raining_products * std_all_products
rain_count_std_interaction = count_raining_products_final * std_all_products_final
np.save(f"{OUTPUT_DIR}长江_rain_count_std_interaction.npy", rain_count_std_interaction)
print("  Saved 长江_rain_count_std_interaction.npy")
del rain_count_std_interaction
gc.collect()

# Final cleanup of remaining loaded arrays
del std_all_products_final, cv_all_products_final, count_raining_products_final
gc.collect()
del yangtze_mask # Mask itself is not needed anymore
gc.collect()

print("\nFeature engineering complete. All features saved as .npy files in /sandbox/outputs/features/.")
