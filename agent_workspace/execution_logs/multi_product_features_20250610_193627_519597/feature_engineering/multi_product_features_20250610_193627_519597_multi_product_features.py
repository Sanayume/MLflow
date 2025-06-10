
import scipy.io
import numpy as np
import os

# Define product names and file paths
PRODUCTS = ['CHIRPS', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
YEARS = range(2016, 2021)
BASE_PATH = '/sandbox/datasets/'
OUTPUT_PATH = '/sandbox/outputs/features/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Starting feature engineering for 5 years (2016-2020)...")

# Load CHM data for masking
print("Loading CHM_2016_2020.mat for masking...")
chm_data_full = scipy.io.loadmat(os.path.join(BASE_PATH, 'CHMdata', 'CHM_2016_2020.mat'))['data']
# Ensure CHM data is float and handle negative values
chm_data_full[chm_data_full < 0] = 0
# Create the global valid mask (where CHM is not NaN)
global_valid_mask = ~np.isnan(chm_data_full)
print(f"Global valid mask created. Total valid points: {np.sum(global_valid_mask)}")

# Dictionary to store all product data
all_product_data = {}

# Load all product data for 5 years and apply preprocessing
for product_name in PRODUCTS:
    # Corrected file path for PERSIANN
    if product_name == 'PERSIANN':
        file_path = os.path.join(BASE_PATH, f'{product_name}data', f'{product_name}_2016_2020.mat')
    else:
        file_path = os.path.join(BASE_PATH, f'{product_name}data', f'{product_name}_2016_2020.mat')
    
    print(f"Loading {product_name}_2016_2020.mat from {file_path}...")
    try:
        data = scipy.io.loadmat(file_path)['data']
        data[data < 0] = 0 # Handle negative values
        
        # Apply global valid mask
        data[~global_valid_mask] = np.nan
        
        all_product_data[product_name] = data
        print(f"{product_name} loaded. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"WARNING: File not found for {product_name} at {file_path}. Skipping this product.")
        # If a product file is not found, we should still proceed with others
        # but mark this product as unavailable for stacking.
        all_product_data[product_name] = None # Mark as None

print("Handling specific NaNs in CHIRPS and sm2rain within the valid mask...")
for product_name_to_fill in ['CHIRPS', 'sm2rain']:
    if all_product_data.get(product_name_to_fill) is None:
        print(f"  Skipping NaN fill for {product_name_to_fill} as data not loaded.")
        continue

    data_to_fill = all_product_data[product_name_to_fill]
    
    nan_indices = np.isnan(data_to_fill) & global_valid_mask
    print(f"  {product_name_to_fill} original NaNs in valid mask: {np.sum(nan_indices)}")

    # Check if GSMAP and IMERG are available for filling
    if all_product_data.get('GSMAP') is not None and all_product_data.get('IMERG') is not None:
        gsmap_values = all_product_data['GSMAP'][nan_indices]
        imerg_values = all_product_data['IMERG'][nan_indices]
        
        # Calculate mean of GSMAP and IMERG for filling
        fill_values = np.nanmean(np.array([gsmap_values, imerg_values]), axis=0)
        
        # Fill with calculated mean
        data_to_fill[nan_indices] = fill_values
        
        # Handle cases where fill_values themselves are NaN (both GSMAP and IMERG were NaN at that point)
        remaining_nan_indices = np.isnan(data_to_fill) & global_valid_mask
        if np.sum(remaining_nan_indices) > 0:
            print(f"  {product_name_to_fill} remaining NaNs after mean fill: {np.sum(remaining_nan_indices)}. Filling with 0.")
            data_to_fill[remaining_nan_indices] = 0.0
    else:
        print(f"  GSMAP or IMERG not available for filling {product_name_to_fill}. Filling remaining NaNs with 0.")
        data_to_fill[nan_indices] = 0.0 # Fallback to 0 fill if source products are missing

    all_product_data[product_name_to_fill] = data_to_fill
    print(f"  {product_name_to_fill} NaNs in valid mask after fill: {np.sum(np.isnan(data_to_fill) & global_valid_mask)}")

print("All product data loaded and preprocessed.")

# Filter out products that failed to load
available_products = [p for p in PRODUCTS if all_product_data.get(p) is not None]
if not available_products:
    raise ValueError("No product data loaded successfully. Cannot proceed with feature calculation.")

# Stack all available product data for multi-product calculations
stacked_products = np.stack([all_product_data[p] for p in available_products], axis=-1)
print(f"Stacked products shape: {stacked_products.shape}")

# --- Multi-Product Synergy Features ---
print("Calculating Multi-Product Synergy Features...")

features = {}

# Multi-product Mean
mean_all_products = np.nanmean(stacked_products, axis=-1)
features['mean_all_products'] = mean_all_products
print(f"  mean_all_products calculated. Shape: {mean_all_products.shape}")

# Multi-product Standard Deviation
std_all_products = np.nanstd(stacked_products, axis=-1)
features['std_all_products'] = std_all_products
print(f"  std_all_products calculated. Shape: {std_all_products.shape}")

# Multi-product Median
median_all_products = np.nanmedian(stacked_products, axis=-1)
features['median_all_products'] = median_all_products
print(f"  median_all_products calculated. Shape: {median_all_products.shape}")

# Multi-product Max
max_all_products = np.nanmax(stacked_products, axis=-1)
features['max_all_products'] = max_all_products
print(f"  max_all_products calculated. Shape: {max_all_products.shape}")

# Multi-product Min
min_all_products = np.nanmin(stacked_products, axis=-1)
features['min_all_products'] = min_all_products
print(f"  min_all_products calculated. Shape: {min_all_products.shape}")

# Multi-product Range
range_all_products = max_all_products - min_all_products
features['range_all_products'] = range_all_products
print(f"  range_all_products calculated. Shape: {range_all_products.shape}")

# Indicator for raining products (count_raining_products)
# Threshold for "raining": > 0.1 mm/d
raining_threshold = 0.1
# Create a boolean array where True means raining for that product at that point
is_raining = stacked_products > raining_threshold
# Count how many products are raining at each point
count_raining_products = np.sum(is_raining, axis=-1).astype(float) # Convert to float for consistency
# Ensure NaN areas remain NaN in the count
count_raining_products[~global_valid_mask] = np.nan
features['count_raining_products'] = count_raining_products
print(f"  count_raining_products calculated. Shape: {count_raining_products.shape}")

# Coefficient of Variation (CV): std_all_products / mean_all_products
# Handle division by zero for mean_all_products == 0
cv_all_products = np.full_like(mean_all_products, np.nan)
non_zero_mean_mask = (mean_all_products != 0) & (~np.isnan(mean_all_products))
cv_all_products[non_zero_mean_mask] = std_all_products[non_zero_mean_mask] / mean_all_products[non_zero_mean_mask]
features['cv_all_products'] = cv_all_products
print(f"  cv_all_products calculated. Shape: {cv_all_products.shape}")

print("Saving calculated features to .mat files...")
for feature_name, feature_data in features.items():
    output_file = os.path.join(OUTPUT_PATH, f'{feature_name}.mat')
    scipy.io.savemat(output_file, {'data': feature_data})
    print(f"  Saved {feature_name}.mat")

print("Multi-Product Synergy Features generation complete.")
print("Next, we can proceed with Temporal Dynamic Features.")
