
import scipy.io

try:
    # Load the mask.mat file
    mat_data = scipy.io.loadmat('/sandbox/datasets/mask/mask.mat')

    # Print keys to see what variables are inside
    print(f"Variables in mask.mat: {mat_data.keys()}")

    # Assuming the mask data is in a variable named 'data' or similar
    # We will iterate through keys to find a suitable candidate if 'data' is not present
    mask_variable_found = False
    for key, value in mat_data.items():
        if not key.startswith('__'): # Ignore metadata keys
            if isinstance(value, (list, tuple, dict)): # Check if it's a scalar or simple array
                continue
            print(f"Variable '{key}':")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Min value: {value.min()}")
            print(f"  Max value: {value.max()}")
            print(f"  Unique values (first 10 if many): {np.unique(value)[:10]}")
            mask_variable_found = True
            break # Assuming only one main data variable

    if not mask_variable_found:
        print("No apparent data variable found in mask.mat, or its structure is unexpected.")

except FileNotFoundError:
    print("Error: mask.mat not found in /sandbox/datasets/mask/")
except Exception as e:
    print(f"An error occurred while loading or processing mask.mat: {e}")
