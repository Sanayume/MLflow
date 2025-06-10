
import scipy.io
import numpy as np

# Define file paths for 2016 data
base_path = '/sandbox/datasets/'
chm_file = base_path + 'CHMdata/CHM_2016.mat'
chirps_file = base_path + 'CHIRPSdata/chirps_2016.mat'
sm2rain_file = base_path + 'sm2raindata/sm2rain_2016.mat'
gsmap_file = base_path + 'GSMAPdata/GSMAP_2016.mat'
imerg_file = base_path + 'IMERGdata/IMERG_2016.mat'

# Load data
try:
    chm_data = scipy.io.loadmat(chm_file)['data']
    chirps_data = scipy.io.loadmat(chirps_file)['data']
    sm2rain_data = scipy.io.loadmat(sm2rain_file)['data']
    gsmap_data = scipy.io.loadmat(gsmap_file)['data']
    imerg_data = scipy.io.loadmat(imerg_file)['data']
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Ensure all data are float type for NaN operations and consistency
chm_data = chm_data.astype(np.float64)
chirps_data = chirps_data.astype(np.float64)
sm2rain_data = sm2rain_data.astype(np.float64)
gsmap_data = gsmap_data.astype(np.float64)
imerg_data = imerg_data.astype(np.float64)

# Create the CHM valid mask (True where CHM is not NaN)
chm_valid_mask = ~np.isnan(chm_data)

# --- Process CHIRPS data ---
# Find NaNs in CHIRPS within the CHM valid mask
chirps_nan_mask_in_chm_valid = np.isnan(chirps_data) & chm_valid_mask

# Store original NaN count for verification
original_chirps_nan_count = np.sum(chirps_nan_mask_in_chm_valid)
print(f"Original CHIRPS NaNs in CHM valid mask: {original_chirps_nan_count}")

# Get values from GSMAP and IMERG where CHIRPS is NaN and CHM is valid
gsmap_vals_for_fill_chirps = gsmap_data[chirps_nan_mask_in_chm_valid]
imerg_vals_for_fill_chirps = imerg_data[chirps_nan_mask_in_chm_valid]

# Stack them and compute nanmean along the new axis (axis=0)
# This will compute the mean for each (lat, lon, day) slice where chirps was NaN
# If both GSMAP and IMERG are NaN, nanmean will return NaN
filling_values_chirps = np.nanmean(np.array([gsmap_vals_for_fill_chirps, imerg_vals_for_fill_chirps]), axis=0)

# Create a copy to store filled data
filled_chirps_data = np.copy(chirps_data)
# Replace NaNs in chirps_data with calculated filling_values
filled_chirps_data[chirps_nan_mask_in_chm_valid] = filling_values_chirps

# Handle cases where filling_values_chirps themselves are NaN (i.e., both GSMAP and IMERG were NaN)
# Replace these remaining NaNs with 0 as per the refined logic for precipitation data.
remaining_chirps_nan_after_fill = np.isnan(filled_chirps_data) & chirps_nan_mask_in_chm_valid
filled_chirps_data[remaining_chirps_nan_after_fill] = 0

final_chirps_nan_count = np.sum(np.isnan(filled_chirps_data) & chm_valid_mask)
print(f"Final CHIRPS NaNs in CHM valid mask after fill: {final_chirps_nan_count}")

# --- Process sm2rain data ---
# Find NaNs in sm2rain within the CHM valid mask
sm2rain_nan_mask_in_chm_valid = np.isnan(sm2rain_data) & chm_valid_mask

# Store original NaN count for verification
original_sm2rain_nan_count = np.sum(sm2rain_nan_mask_in_chm_valid)
print(f"Original sm2rain NaNs in CHM valid mask: {original_sm2rain_nan_count}")

# Get values from GSMAP and IMERG where sm2rain is NaN and CHM is valid
gsmap_vals_for_fill_sm2rain = gsmap_data[sm2rain_nan_mask_in_chm_valid]
imerg_vals_for_fill_sm2rain = imerg_data[sm2rain_nan_mask_in_chm_valid]

filling_values_sm2rain = np.nanmean(np.array([gsmap_vals_for_fill_sm2rain, imerg_vals_for_fill_sm2rain]), axis=0)

filled_sm2rain_data = np.copy(sm2rain_data)
filled_sm2rain_data[sm2rain_nan_mask_in_chm_valid] = filling_values_sm2rain

remaining_sm2rain_nan_after_fill = np.isnan(filled_sm2rain_data) & sm2rain_nan_mask_in_chm_valid
filled_sm2rain_data[remaining_sm2rain_nan_after_fill] = 0

final_sm2rain_nan_count = np.sum(np.isnan(filled_sm2rain_data) & chm_valid_mask)
print(f"Final sm2rain NaNs in CHM valid mask after fill: {final_sm2rain_nan_count}")

# Verify that the original NaN values in CHM (the mask itself) are preserved
# This means points outside the valid mask should still be NaN in the filled data
assert np.all(np.isnan(chm_data[~chm_valid_mask]) == np.isnan(filled_chirps_data[~chm_valid_mask]))
assert np.all(np.isnan(chm_data[~chm_valid_mask]) == np.isnan(filled_sm2rain_data[~chm_valid_mask]))

print("\nMissing value imputation complete for CHIRPS and sm2rain data.")
print("CHIRPS data has been filled. Remaining NaNs in CHM valid mask: ", final_chirps_nan_count)
print("sm2rain data has been filled. Remaining NaNs in CHM valid mask: ", final_sm2rain_nan_count)

# To make the filled data available for subsequent steps, we would typically save them.
# For now, we confirm the filling and indicate readiness for Bayesian study.
# If you wish to save these filled arrays, please let me know the desired filenames.

