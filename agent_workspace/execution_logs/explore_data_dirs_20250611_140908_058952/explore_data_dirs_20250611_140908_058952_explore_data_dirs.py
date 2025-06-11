
import os

print("--- Listing /sandbox/datasets/mask/ ---")
try:
    mask_files = os.listdir('/sandbox/datasets/mask/')
    print(mask_files)
    # 尝试读取一个文件，如果存在且是常见格式
    if mask_files:
        first_mask_file = mask_files[0]
        file_path = os.path.join('/sandbox/datasets/mask/', first_mask_file)
        if first_mask_file.endswith(('.txt', '.csv', '.json', '.geojson', '.tif', '.nc')):
            print(f"\n--- Attempting to read first few lines/info of {first_mask_file} ---")
            if first_mask_file.endswith(('.txt', '.csv', '.json')):
                with open(file_path, 'r', errors='ignore') as f:
                    print(f.read(500)) # Read first 500 characters
            elif first_mask_file.endswith(('.tif', '.nc')):
                print(f"Skipping direct read of binary/geospatial file: {first_mask_file}. You might need specific libraries like rasterio or xarray.")
        else:
            print(f"Skipping direct read of file with unknown extension: {first_mask_file}")
except Exception as e:
    print(f"Error listing or reading /sandbox/datasets/mask/: {e}")

print("\n--- Listing /sandbox/datasets/rainfalldata/ ---")
try:
    rainfall_files = os.listdir('/sandbox/datasets/rainfalldata/')
    print(rainfall_files)
    # 尝试读取一个文件，如果存在且是常见格式
    if rainfall_files:
        first_rainfall_file = rainfall_files[0]
        file_path = os.path.join('/sandbox/datasets/rainfalldata/', first_rainfall_file)
        if first_rainfall_file.endswith(('.txt', '.csv', '.json', '.nc', '.grib', '.hdf', '.dat')):
            print(f"\n--- Attempting to read first few lines/info of {first_rainfall_file} ---")
            if first_rainfall_file.endswith(('.txt', '.csv', '.json')):
                with open(file_path, 'r', errors='ignore') as f:
                    print(f.read(500)) # Read first 500 characters
            elif first_rainfall_file.endswith(('.nc', '.grib', '.hdf', '.dat')):
                print(f"Skipping direct read of binary/scientific file: {first_rainfall_file}. You might need specific libraries like xarray, netCDF4, pygrib.")
        else:
            print(f"Skipping direct read of file with unknown extension: {first_rainfall_file}")
except Exception as e:
    print(f"Error listing or reading /sandbox/datasets/rainfalldata/: {e}")
