
import scipy.io
import numpy as np

file_path = '/sandbox/datasets/rainfalldata/CHIRPSdata/chirps_2016.mat'

try:
    mat_data = scipy.io.loadmat(file_path)
    print(f"Successfully loaded {file_path}")
    print("Keys found in the .mat file:")
    for key, value in mat_data.items():
        if not key.startswith('__'): # Ignore MATLAB internal keys
            print(f"  Key: {key}")
            print(f"    Type: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"    Shape: {value.shape}")
                if value.size > 0: # Check if array is not empty
                    print(f"    Sample data (first 5 elements/rows):\n{value.flatten()[:5] if value.ndim == 1 else value[:5]}")
            else:
                print(f"    Value: {value}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
