
import scipy.io
import numpy as np

mask_file_path = '/sandbox/datasets/mask/mask.mat'

try:
    mask_data = scipy.io.loadmat(mask_file_path)

    print(f"Loaded mask file: {mask_file_path}")
    print("Variables in mask.mat:")
    for var_name in mask_data:
        if not var_name.startswith("__"):
            var = mask_data[var_name]
            print(f"  - {var_name}: shape={var.shape}, dtype={var.dtype}")

            if var_name == 'mask' or var_name == 'data': # Assuming the mask variable is named 'mask' or 'data'
                print(f"    Unique values: {np.unique(var)}")
                print(f"    Min value: {np.nanmin(var)}")
                print(f"    Max value: {np.nanmax(var)}")

except FileNotFoundError:
    print(f"Error: mask.mat not found at {mask_file_path}. Please ensure the file is in the correct location.")
except Exception as e:
    print(f"An error occurred while loading or processing mask.mat: {e}")
