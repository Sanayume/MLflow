
import os
import xarray as xr

data_dir = "/sandbox/datasets/rainfalldata/"

print(f"Listing files in: {data_dir}")
files = os.listdir(data_dir)
for f in files:
    print(f"- {f}")

# Try to open the first .nc file found to inspect its structure
nc_files = [f for f in files if f.endswith(".nc")]
if nc_files:
    first_nc_file = os.path.join(data_dir, nc_files[0])
    print(f"\nAttempting to open and inspect: {first_nc_file}")
    try:
        with xr.open_dataset(first_nc_file) as ds:
            print("Dataset Info:")
            print(ds)
            print("\nCoordinates:")
            for coord in ds.coords:
                print(f"- {coord}: {ds.coords[coord].dims}")
            print("\nData Variables:")
            for var in ds.data_vars:
                print(f"- {var}: {ds.data_vars[var].dims}")
    except Exception as e:
        print(f"Error opening or inspecting {first_nc_file}: {e}")
else:
    print("\nNo .nc files found in the directory.")
