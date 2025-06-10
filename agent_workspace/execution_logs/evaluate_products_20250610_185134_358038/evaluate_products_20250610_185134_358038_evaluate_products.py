
import scipy.io
import numpy as np

def evaluate_product(chm_data, product_data, product_name):
    # Ensure both arrays are flattened
    chm_flat = chm_data.flatten()
    product_flat = product_data.flatten()

    # Filter out NaN values from both arrays to only consider valid geographical regions
    # A data point is valid if it's not NaN in EITHER chm_flat or product_flat
    # However, based on previous discussion, NaNs are consistent masks, so we can filter where CHM is not NaN.
    # Let's find indices where both are not NaN for robust comparison
    valid_indices = ~np.isnan(chm_flat) & ~np.isnan(product_flat)

    chm_valid = chm_flat[valid_indices]
    product_valid = product_flat[valid_indices]

    # Handle potential negative values by setting them to 0
    chm_valid[chm_valid < 0] = 0
    product_valid[product_valid < 0] = 0

    print(f"\n--- Evaluating {product_name} ---")
    print(f"Valid data points for {product_name}: {len(chm_valid)}")

    if len(chm_valid) == 0:
        print(f"No common valid data points found for {product_name}. Skipping evaluation.")
        return

    # Calculate RMSE
    rmse = np.sqrt(np.mean((product_valid - chm_valid)**2))
    print(f"RMSE for {product_name}: {rmse:.4f}")

    # Calculate Pearson Correlation Coefficient
    # Handle case where standard deviation might be zero (e.g., all values are same)
    if np.std(chm_valid) == 0 or np.std(product_valid) == 0:
        correlation = np.nan # Correlation is undefined if one array has no variance
    else:
        correlation = np.corrcoef(chm_valid, product_valid)[0, 1]
    print(f"Pearson Correlation for {product_name}: {correlation:.4f}")


# Load CHM true data (assuming it's the reference for all comparisons)
chm_mat = scipy.io.loadmat('/sandbox/datasets/CHMdata/CHM_2016_2020.mat')
chm_data = chm_mat['data']

# List of other products to evaluate
products_to_evaluate = [
    ('CMORPHdata/CMORPH_2016_2020.mat', 'CMORPH'),
    ('GSMAPdata/GSMAP_2016_2020.mat', 'GSMAP'),
    ('IMERGdata/IMERG_2016_2020.mat', 'IMERG'),
    ('PERSIANNdata/PERSIANN_2016_2020.mat', 'PERSIANN'),
    ('sm2raindata/sm2rain_2016_2020.mat', 'sm2rain'),
]

for path, name in products_to_evaluate:
    try:
        product_mat = scipy.io.loadmat(f'/sandbox/datasets/{path}')
        product_data = product_mat['data']
        evaluate_product(chm_data, product_data, name)
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

print("\n--- Evaluation of all products complete ---")
