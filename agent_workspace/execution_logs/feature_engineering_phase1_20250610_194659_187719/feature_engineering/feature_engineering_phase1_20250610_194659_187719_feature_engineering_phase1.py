
import scipy.io
import numpy as np
import os

# Define product names
PRODUCTS = ['CHIRPS', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
BASE_PATH = '/sandbox/datasets/'
OUTPUT_PATH = '/sandbox/outputs/features/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Starting feature engineering for 5 years (2016-2020).")
print("Phase 1: Raw Values and Multi-Product Synergy Features.")

# Load CHM data for masking
print("Loading CHM_2016_2020.mat for masking...")
try:
    chm_data_full = scipy.io.loadmat(os.path.join(BASE_PATH, 'CHMdata', 'CHM_2016_2020.mat'))['data']
    chm_data_full[chm_data_full < 0] = 0
    global_valid_mask = ~np.isnan(chm_data_full)
    feature_shape = chm_data_full.shape # (144, 256, 1827)
    print(f"Global valid mask created. Total valid points: {np.sum(global_valid_mask)}")
except FileNotFoundError:
    print("ERROR: CHM_2016_2020.mat not found. Cannot create global mask. Aborting.")
    raise

# Initialize accumulators for synergy features
# These will hold the sum, sum of squares, min, max, count for mean, count for raining products
# Initialize with NaNs where invalid, or 0s/inf for accumulation
sum_all_products = np.full(feature_shape, 0.0, dtype=np.float64)
sum_sq_all_products = np.full(feature_shape, 0.0, dtype=np.float64) # For standard deviation
min_all_products_acc = np.full(feature_shape, np.inf, dtype=np.float64) # Initialize with positive infinity
max_all_products_acc = np.full(feature_shape, -np.inf, dtype=np.float64) # Initialize with negative infinity
count_for_mean_acc = np.full(feature_shape, 0, dtype=np.int32)
count_raining_products_acc = np.full(feature_shape, 0, dtype=np.int32)

# Keep track of products that were actually loaded and preprocessed for specific fills
loaded_products_for_fill = {} 

# Loop through each product to load, preprocess, save raw, and accumulate for synergy features
for product_name in PRODUCTS:
    file_path = os.path.join(BASE_PATH, f'{product_name}data', f'{product_name}_2016_2020.mat')
    
    print(f"Loading {product_name}_2016_2020.mat from {file_path}...")
    try:
        data = scipy.io.loadmat(file_path)['data']
        data[data < 0] = 0 # Handle negative values

        # Apply global valid mask (set non-valid areas to NaN)
        data[~global_valid_mask] = np.nan
        
        # Save Raw Values feature immediately
        output_file_raw = os.path.join(OUTPUT_PATH, f'raw_{product_name}.mat')
        scipy.io.savemat(output_file_raw, {'data': data})
        print(f"  Saved raw_{product_name}.mat")

        # Store data for specific fills if needed
        if product_name in ['GSMAP', 'IMERG', 'CHIRPS', 'sm2rain']:
            loaded_products_for_fill[product_name] = data
        
        # Accumulate for synergy features (only consider non-NaN values within the valid mask)
        valid_points_in_current_product = ~np.isnan(data) & global_valid_mask
        
        sum_all_products[valid_points_in_current_product] += data[valid_points_in_current_product]
        sum_sq_all_products[valid_points_in_current_product] += (data[valid_points_in_current_product] ** 2)
        
        # Use np.fmin/fmax for element-wise min/max that handles NaNs correctly
        min_all_products_acc = np.fmin(min_all_products_acc, data)
        max_all_products_acc = np.fmax(max_all_products_acc, data)
        
        count_for_mean_acc[valid_points_in_current_product] += 1
        
        # For count_raining_products
        raining_threshold = 0.1
        is_raining_current_product = (data > raining_threshold) & valid_points_in_current_product
        count_raining_products_acc[is_raining_current_product] += 1

        # Clear data from memory if it's not one of the products needed for specific fills
        if product_name not in ['GSMAP', 'IMERG', 'CHIRPS', 'sm2rain']:
            del data
            print(f"  {product_name} data cleared from memory after processing.")

    except FileNotFoundError:
        print(f"WARNING: File not found for {product_name} at {file_path}. Skipping this product.")
        if product_name in ['GSMAP', 'IMERG', 'CHIRPS', 'sm2rain']:
            loaded_products_for_fill[product_name] = None


print("Handling specific NaNs in CHIRPS and sm2rain within the valid mask (after raw feature saving and initial accumulation)...")
# Determine fallback fill method if GSMAP or IMERG are not available
if loaded_products_for_fill.get('GSMAP') is None or loaded_products_for_fill.get('IMERG') is None:
    print("WARNING: GSMAP or IMERG not available. Cannot perform mean-based NaN filling for CHIRPS/sm2rain.")
    fallback_fill_method = 'zero' # Fallback to 0 fill if source products are missing or not loaded
else:
    fallback_fill_method = None # Use mean fill

for product_name_to_fill in ['CHIRPS', 'sm2rain']:
    if loaded_products_for_fill.get(product_name_to_fill) is None:
        print(f"  Skipping NaN fill for {product_name_to_fill} as data not loaded or already processed.")
        continue

    data_to_fill = loaded_products_for_fill[product_name_to_fill] # Get the preprocessed data

    # Identify NaNs only within the global valid mask
    nan_indices = np.isnan(data_to_fill) & global_valid_mask
    if np.sum(nan_indices) > 0:
        print(f"  {product_name_to_fill} original NaNs in valid mask: {np.sum(nan_indices)}")

        if fallback_fill_method == 'zero':
            data_to_fill[nan_indices] = 0.0
            print(f"  {product_name_to_fill} filled with 0 (fallback).")
        else:
            gsmap_values = loaded_products_for_fill['GSMAP'][nan_indices]
            imerg_values = loaded_products_for_fill['IMERG'][nan_indices]
            
            fill_values = np.nanmean(np.array([gsmap_values, imerg_values]), axis=0) # Calculate mean for each NaN point
            
            data_to_fill[nan_indices] = fill_values
            
            remaining_nan_indices = np.isnan(data_to_fill) & global_valid_mask
            if np.sum(remaining_nan_indices) > 0:
                print(f"  {product_name_to_fill} remaining NaNs after mean fill: {np.sum(remaining_nan_indices)}. Filling with 0.")
                data_to_fill[remaining_nan_indices] = 0.0
    
        # Update the data in loaded_products_for_fill (this also updates the original array if it was passed by reference)
        loaded_products_for_fill[product_name_to_fill] = data_to_fill
        print(f"  {product_name_to_fill} NaNs in valid mask after fill: {np.sum(np.isnan(data_to_fill) & global_valid_mask)}")
    else:
        print(f"  {product_name_to_fill} has no additional NaNs in valid mask. No fill needed.")


# --- Multi-Product Synergy Features from Accumulators ---
print("Calculating Multi-Product Synergy Features from accumulators...")
features_to_save = {}

# Multi-product Mean
mean_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
# Only calculate where count_for_mean_acc > 0 and it's in the global valid mask
non_zero_count_mask = (count_for_mean_acc > 0) & global_valid_mask
mean_all_products[non_zero_count_mask] = sum_all_products[non_zero_count_mask] / count_for_mean_acc[non_zero_count_mask]
features_to_save['mean_all_products'] = mean_all_products
print(f"  mean_all_products calculated. Shape: {mean_all_products.shape}")

# Multi-product Standard Deviation
std_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
# Only calculate where count is > 1 for meaningful std dev, and it's in the global valid mask
valid_std_mask = (count_for_mean_acc > 1) & global_valid_mask
# Variance = (Sum(x^2) - (Sum(x))^2 / N) / N
variance = (sum_sq_all_products[valid_std_mask] - (sum_all_products[valid_std_mask] ** 2) / count_for_mean_acc[valid_std_mask]) / count_for_mean_acc[valid_std_mask]
std_all_products[valid_std_mask] = np.sqrt(np.maximum(0, variance)) # Ensure non-negative for sqrt
features_to_save['std_all_products'] = std_all_products
print(f"  std_all_products calculated. Shape: {std_all_products.shape}")

# Multi-product Median - Cannot be efficiently calculated with current accumulator strategy. Will skip for now.

# Multi-product Max (from accumulator)
# Replace -inf with NaN where count is 0 (i.e., no valid data was accumulated)
max_all_products_acc[count_for_mean_acc == 0] = np.nan
max_all_products_acc[~global_valid_mask] = np.nan # Ensure non-valid areas are NaN
features_to_save['max_all_products'] = max_all_products_acc
print(f"  max_all_products calculated. Shape: {max_all_products_acc.shape}")

# Multi-product Min (from accumulator)
# Replace inf with NaN where count is 0
min_all_products_acc[count_for_mean_acc == 0] = np.nan
min_all_products_acc[~global_valid_mask] = np.nan # Ensure non-valid areas are NaN
features_to_save['min_all_products'] = min_all_products_acc
print(f"  min_all_products calculated. Shape: {min_all_products_acc.shape}")

# Multi-product Range
range_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
# Calculate only where both max and min are not NaN (i.e., count > 0)
valid_range_mask = (~np.isnan(max_all_products_acc)) & (~np.isnan(min_all_products_acc)) & global_valid_mask
range_all_products[valid_range_mask] = max_all_products_acc[valid_range_mask] - min_all_products_acc[valid_range_mask]
features_to_save['range_all_products'] = range_all_products
print(f"  range_all_products calculated. Shape: {range_all_products.shape}")

# Indicator for raining products (count_raining_products)
count_raining_products = np.full(feature_shape, np.nan, dtype=np.float64) # Store as float for consistency
count_raining_products[global_valid_mask] = count_raining_products_acc[global_valid_mask].astype(float)
features_to_save['count_raining_products'] = count_raining_products
print(f"  count_raining_products calculated. Shape: {count_raining_products.shape}")

# Coefficient of Variation (CV): std_all_products / mean_all_products
cv_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
# Only calculate where mean is not zero and not NaN, and std is not NaN
non_zero_mean_and_valid_std_mask = (mean_all_products != 0) & (~np.isnan(mean_all_products)) & (~np.isnan(std_all_products)) & global_valid_mask
cv_all_products[non_zero_mean_and_valid_std_mask] = std_all_products[non_zero_mean_and_valid_std_mask] / mean_all_products[non_zero_mean_and_valid_std_mask]
features_to_save['cv_all_products'] = cv_all_products
print(f"  cv_all_products calculated. Shape: {cv_all_products.shape}")


print("Saving calculated Multi-Product Synergy Features to .mat files...")
for feature_name, feature_data in features_to_save.items():
    output_file = os.path.join(OUTPUT_PATH, f'{feature_name}.mat')
    scipy.io.savemat(output_file, {'data': feature_data})
    print(f"  Saved {feature_name}.mat")

# Clear accumulators and loaded_products_for_fill to free memory
del sum_all_products, sum_sq_all_products, min_all_products_acc, max_all_products_acc, count_for_mean_acc, count_raining_products_acc
del loaded_products_for_fill
print("Accumulators and temporary data cleared from memory.")

print("Phase 1: Raw Values and Multi-Product Synergy Features generation complete.")
print("Proceeding to Phase 2: Temporal Dynamic Features (Lag, Difference, Sliding Window). This will be done in a subsequent call due to complexity and memory management.")
