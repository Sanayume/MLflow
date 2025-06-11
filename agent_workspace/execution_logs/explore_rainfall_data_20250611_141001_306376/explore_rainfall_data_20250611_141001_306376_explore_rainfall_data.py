
import os

print("Listing files in /sandbox/datasets/rainfalldata/CHIRPSdata/:")
try:
    chirps_files = os.listdir('/sandbox/datasets/rainfalldata/CHIRPSdata/')
    print(chirps_files)
except FileNotFoundError:
    print("Directory /sandbox/datasets/rainfalldata/CHIRPSdata/ not found.")

print("\nListing files in /sandbox/datasets/rainfalldata/CMORPHdata/:")
try:
    cmorph_files = os.listdir('/sandbox/datasets/rainfalldata/CMORPHdata/')
    print(cmorph_files)
except FileNotFoundError:
    print("Directory /sandbox/datasets/rainfalldata/CMORPHdata/ not found.")

# Further exploration will depend on the file types found.
# For now, just list the files.
