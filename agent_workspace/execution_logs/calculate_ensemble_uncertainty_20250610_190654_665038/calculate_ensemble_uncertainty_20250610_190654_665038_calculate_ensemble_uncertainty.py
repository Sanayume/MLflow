
import scipy.io
import numpy as np

# Define file paths for 2016 data
base_path = '/sandbox/datasets/'
chm_file = base_path + 'CHMdata/CHM_2016.mat'
chirps_file = base_path + 'CHIRPSdata/chirps_2016.mat'
cmorph_file = base_path + 'CMORPHdata/CMORPH_2016.mat'
gsmap_file = base_path + 'GSMAPdata/GSMAP_2016.mat'
imerg_file = base_path + 'IMERGdata/IMERG_2016.mat'
persiann_file = base_path + 'PERSIANNdata/PERSIANN_2016.mat' # Corrected path
sm2rain_file = base_path + 'sm2raindata/sm2rain_2016.mat'

# --- Function to load and preprocess data ---
def load_and_preprocess(file_path, chm_valid_mask, fill_value=0):
    try:
        data = scipy.io.loadmat(file_path)['data']
        # Ensure non-negative values
        data[data < 0] = 0
        # Apply CHM valid mask
        data[~chm_valid_mask] = np.nan
        return data
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping this product.")
        return None
    except Exception as e:
        print(f"Error loading or processing {file_path}: {e}")
        return None

# --- Main execution ---
print("Starting uncertainty calculation for 2016 data...")

# 1. Load CHM data to get the valid mask
chm_data_raw = scipy.io.loadmat(chm_file)['data']
chm_valid_mask = ~np.isnan(chm_data_raw)

# 2. Load and preprocess all product data
products_data = {}
product_names = ['CHIRPS', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
product_files = {
    'CHIRPS': chirps_file,
    'CMORPH': cmorph_file,
    'GSMAP': gsmap_file,
    'IMERG': imerg_file,
    'PERSIANN': persiann_file,
    'sm2rain': sm2rain_file
}

for name in product_names:
    file_path = product_files[name]
    processed_data = load_and_preprocess(file_path, chm_valid_mask)
    if processed_data is not None:
        products_data[name] = processed_data

# 3. Handle specific NaN filling for CHIRPS and sm2rain within the valid mask
#    Using GSMAP and IMERG for filling as they had best correlation (excluding CHIRPS/sm2rain)

if 'GSMAP' in products_data and 'IMERG' in products_data:
    gsmap_data = products_data['GSMAP']
    imerg_data = products_data['IMERG']

    products_to_fill = ['CHIRPS', 'sm2rain']

    for name in products_to_fill:
        if name in products_data:
            data_to_fill = products_data[name]
            
            # Identify NaNs within the CHM valid mask
            nan_indices = np.where(np.isnan(data_to_fill) & chm_valid_mask)
            
            if nan_indices[0].size > 0:
                print(f"Found {nan_indices[0].size} NaNs in {name} within CHM valid mask. Attempting to fill...")
                
                # Get corresponding GSMAP and IMERG values
                gsmap_values = gsmap_data[nan_indices]
                imerg_values = imerg_data[nan_indices]
                
                # Calculate mean, handling cases where both are NaN
                filled_values = np.nanmean(np.stack([gsmap_values, imerg_values], axis=-1), axis=-1)
                
                # Fill the data_to_fill array
                data_to_fill[nan_indices] = filled_values
                
                # After filling, check if any NaNs remain (e.g., if gsmap/imerg were also NaN)
                remaining_nans_after_fill = np.sum(np.isnan(data_to_fill) & chm_valid_mask)
                if remaining_nans_after_fill > 0:
                    print(f"Warning: {remaining_nans_after_fill} NaNs still remain in {name} after filling with GSMAP/IMERG mean. Filling remaining with 0.")
                    data_to_fill[np.isnan(data_to_fill) & chm_valid_mask] = 0
                
                print(f"{name} filling complete. No NaNs in valid mask.")
            else:
                print(f"No NaNs found in {name} within CHM valid mask. No filling needed.")
        else:
            print(f"Product {name} not loaded, skipping NaN filling.")
else:
    print("GSMAP or IMERG data not available for filling. Skipping specific NaN filling.")


# 4. Stack all processed product data (excluding CHM) for uncertainty calculation
#    Only include products that were successfully loaded and processed

products_for_std = []
for name in product_names:
    if name != 'CHM' and name in products_data:
        products_for_std.append(products_data[name])

if not products_for_std:
    print("No valid products available for standard deviation calculation.")
else:
    # Stack along a new axis to get (lat, lon, time, product)
    stacked_products = np.stack(products_for_std, axis=-1)

    # 5. Calculate standard deviation across products for each (lat, lon, time) point
    #    np.nanstd will handle the NaNs outside the valid mask correctly
    uncertainty_map = np.nanstd(stacked_products, axis=-1)

    # 6. Analyze and print results
    print("\n--- Uncertainty (Standard Deviation Across Products) Analysis ---")
    
    # Filter out NaNs (from the mask) for statistics
    valid_uncertainty_values = uncertainty_map[chm_valid_mask]

    if valid_uncertainty_values.size > 0:
        print(f"Total valid uncertainty points: {valid_uncertainty_values.size}")
        print(f"Mean Uncertainty: {np.mean(valid_uncertainty_values):.4f}")
        print(f"Median Uncertainty: {np.median(valid_uncertainty_values):.4f}")
        print(f"Min Uncertainty: {np.min(valid_uncertainty_values):.4f}")
        print(f"Max Uncertainty: {np.max(valid_uncertainty_values):.4f}")
        print(f"Standard Deviation of Uncertainty: {np.std(valid_uncertainty_values):.4f}")
    else:
        print("No valid uncertainty values to analyze after applying mask.")

print("\nUncertainty calculation process finished.")
