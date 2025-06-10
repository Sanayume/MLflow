
import scipy.io
import numpy as np
import os

# Define product names and file paths
PRODUCTS = ['CHIRPS', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
YEARS = range(2016, 2021) # This is for conceptual understanding, actual files are _2016_2020.mat
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
    print(f"Global valid mask created. Total valid points: {np.sum(global_valid_mask)}")
except FileNotFoundError:
    print("ERROR: CHM_2016_2020.mat not found. Cannot create global mask. Aborting.")
    raise

# Dictionary to store all product data after initial preprocessing
# Keys: product_name, Values: 3D numpy array (144, 256, 1827)
all_product_data = {}

# Load all product data for 5 years and apply preprocessing
for product_name in PRODUCTS:
    # Corrected file path for PERSIANN and others
    file_path = os.path.join(BASE_PATH, f'{product_name}data', f'{product_name}_2016_2020.mat')
    
    print(f"Loading {product_name}_2016_2020.mat from {file_path}...")
    try:
        data = scipy.io.loadmat(file_path)['data']
        data[data < 0] = 0 # Handle negative values

        # Apply global valid mask (set non-valid areas to NaN)
        data[~global_valid_mask] = np.nan
        
        all_product_data[product_name] = data
        print(f"  {product_name} loaded. Shape: {data.shape}")

        # Save Raw Values feature
        output_file_raw = os.path.join(OUTPUT_PATH, f'raw_{product_name}.mat')
        scipy.io.savemat(output_file_raw, {'data': data})
        print(f"  Saved raw_{product_name}.mat")

    except FileNotFoundError:
        print(f"WARNING: File not found for {product_name} at {file_path}. Skipping this product.")
        all_product_data[product_name] = None # Mark as None

print("Handling specific NaNs in CHIRPS and sm2rain within the valid mask...")
# Determine fallback fill method if GSMAP or IMERG are not available
if all_product_data.get('GSMAP') is None or all_product_data.get('IMERG') is None:
    print("WARNING: GSMAP or IMERG not available. Cannot perform mean-based NaN filling for CHIRPS/sm2rain.")
    fallback_fill_method = 'zero' # Fallback to 0 fill if source products are missing or not loaded
else:
    fallback_fill_method = None # Use mean fill

for product_name_to_fill in ['CHIRPS', 'sm2rain']:
    if all_product_data.get(product_name_to_fill) is None:
        print(f"  Skipping NaN fill for {product_name_to_fill} as data not loaded.")
        continue

    data_to_fill = all_product_data[product_name_to_fill]
    
    # Identify NaNs only within the global valid mask
    nan_indices = np.isnan(data_to_fill) & global_valid_mask
    print(f"  {product_name_to_fill} original NaNs in valid mask: {np.sum(nan_indices)}")

    if fallback_fill_method == 'zero':
        data_to_fill[nan_indices] = 0.0
        print(f"  {product_name_to_fill} filled with 0 (fallback).")
    else:
        gsmap_values = all_product_data['GSMAP'][nan_indices]
        imerg_values = all_product_data['IMERG'][nan_indices]
        
        # Calculate mean of GSMAP and IMERG for filling, handling cases where both are NaN
        # np.array([gsmap_values, imerg_values]) creates a 2xN array.
        # np.nanmean along axis=0 computes mean for each column (each NaN point).
        fill_values = np.nanmean(np.array([gsmap_values, imerg_values]), axis=0)
        
        # Fill with calculated mean
        data_to_fill[nan_indices] = fill_values
        
        # Handle cases where fill_values themselves are NaN (both GSMAP and IMERG were NaN at that point)
        remaining_nan_indices = np.isnan(data_to_fill) & global_valid_mask
        if np.sum(remaining_nan_indices) > 0:
            print(f"  {product_name_to_fill} remaining NaNs after mean fill: {np.sum(remaining_nan_indices)}. Filling with 0.")
            data_to_fill[remaining_nan_indices] = 0.0
    
    all_product_data[product_name_to_fill] = data_to_fill
    print(f"  {product_name_to_fill} NaNs in valid mask after fill: {np.sum(np.isnan(data_to_fill) & global_valid_mask)}")

print("All product data loaded and preprocessed for synergy features.")

# Filter out products that failed to load
available_products = [p for p in PRODUCTS if all_product_data.get(p) is not None]
if not available_products:
    raise ValueError("No product data loaded successfully for synergy feature calculation. Aborting.")

# --- Multi-Product Synergy Features ---
print("Calculating Multi-Product Synergy Features...")

# Prepare a list to store valid data points for stacking
valid_product_data_list = []
for product_name in available_products:
    data = all_product_data[product_name]
    valid_data_points = data[global_valid_mask] # Extract only valid (non-NaN) data points
    valid_product_data_list.append(valid_data_points)

# Stack the valid data points along a new axis
# Shape will be (num_valid_points, num_products)
if valid_product_data_list: # Ensure list is not empty
    stacked_valid_products = np.stack(valid_product_data_list, axis=-1)
    print(f"Stacked valid products shape: {stacked_valid_products.shape}")
else:
    print("No valid product data to stack for synergy features. Skipping.")
    stacked_valid_products = None # Mark as None to skip calculations

features_to_save = {}
feature_shape = chm_data_full.shape # (144, 256, 1827) - this is the target shape for each feature

if stacked_valid_products is not None:
    # Multi-product Mean
    mean_valid = np.nanmean(stacked_valid_products, axis=-1)
    mean_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
    mean_all_products[global_valid_mask] = mean_valid
    features_to_save['mean_all_products'] = mean_all_products
    print(f"  mean_all_products calculated. Shape: {mean_all_products.shape}")

    # Multi-product Standard Deviation
    std_valid = np.nanstd(stacked_valid_products, axis=-1)
    std_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
    std_all_products[global_valid_mask] = std_valid
    features_to_save['std_all_products'] = std_all_products
    print(f"  std_all_products calculated. Shape: {std_all_products.shape}")

    # Multi-product Median
    median_valid = np.nanmedian(stacked_valid_products, axis=-1)
    median_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
    median_all_products[global_valid_mask] = median_valid
    features_to_save['median_all_products'] = median_all_products
    print(f"  median_all_products calculated. Shape: {median_all_products.shape}")

    # Multi-product Max
    max_valid = np.nanmax(stacked_valid_products, axis=-1)
    max_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
    max_all_products[global_valid_mask] = max_valid
    features_to_save['max_all_products'] = max_all_products
    print(f"  max_all_products calculated. Shape: {max_all_products.shape}")

    # Multi-product Min
    min_valid = np.nanmin(stacked_valid_products, axis=-1)
    min_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
    min_all_products[global_valid_mask] = min_valid
    features_to_save['min_all_products'] = min_all_products
    print(f"  min_all_products calculated. Shape: {min_all_products.shape}")

    # Multi-product Range
    range_all_products = max_all_products - min_all_products # Can be calculated directly from max/min 3D arrays
    features_to_save['range_all_products'] = range_all_products
    print(f"  range_all_products calculated. Shape: {range_all_products.shape}")

    # Indicator for raining products (count_raining_products)
    raining_threshold = 0.1
    is_raining_valid = stacked_valid_products > raining_threshold
    count_raining_products_valid = np.sum(is_raining_valid, axis=-1).astype(float)
    count_raining_products = np.full(feature_shape, np.nan, dtype=np.float64)
    count_raining_products[global_valid_mask] = count_raining_products_valid
    features_to_save['count_raining_products'] = count_raining_products
    print(f"  count_raining_products calculated. Shape: {count_raining_products.shape}")

    # Coefficient of Variation (CV): std_all_products / mean_all_products
    cv_all_products = np.full(feature_shape, np.nan, dtype=np.float64)
    # Only calculate where mean is not zero and not NaN
    non_zero_mean_mask = (mean_all_products != 0) & (~np.isnan(mean_all_products))
    cv_all_products[non_zero_mean_mask] = std_all_products[non_zero_mean_mask] / mean_all_products[non_zero_mean_mask]
    features_to_save['cv_all_products'] = cv_all_products
    print(f"  cv_all_products calculated. Shape: {cv_all_products.shape}")
else:
    print("Skipping Multi-Product Synergy Features due to no valid product data.")

print("Saving calculated Multi-Product Synergy Features to .mat files...")
for feature_name, feature_data in features_to_save.items():
    output_file = os.path.join(OUTPUT_PATH, f'{feature_name}.mat')
    scipy.io.savemat(output_file, {'data': feature_data})
    print(f"  Saved {feature_name}.mat")

print("Phase 1: Raw Values and Multi-Product Synergy Features generation complete.")
print("Proceeding to Phase 2: Temporal Dynamic Features (Lag, Difference, Sliding Window). This will require a more iterative approach to manage memory.")
