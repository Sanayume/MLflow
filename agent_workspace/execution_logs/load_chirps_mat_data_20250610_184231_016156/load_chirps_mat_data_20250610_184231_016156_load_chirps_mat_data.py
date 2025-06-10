
import scipy.io
import os

file_path = '/sandbox/datasets/CHIRPSdata/chirps_2016_2020.mat'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}. Please ensure it's in the correct directory.")
else:
    try:
        mat_data = scipy.io.loadmat(file_path)
        print(f"Successfully loaded {file_path}. Keys in the .mat file are:")
        for key in mat_data.keys():
            # 排除MATLAB内部的元数据键
            if not key.startswith('__'):
                print(f"- {key}: shape={mat_data[key].shape}, dtype={mat_data[key].dtype}")
    except Exception as e:
        print(f"Error loading or processing .mat file: {e}")
