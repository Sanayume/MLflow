
import os

datasets_path = "/sandbox/datasets/"

print(f"Listing contents of {datasets_path}:")
try:
    # List all files and directories in /sandbox/datasets/
    for entry in os.listdir(datasets_path):
        full_path = os.path.join(datasets_path, entry)
        if os.path.isfile(full_path):
            print(f"  File: {entry}")
        elif os.path.isdir(full_path):
            print(f"  Directory: {entry}/")

    # Check for common description files
    description_files = [
        "README.md",
        "README.txt",
        "data_description.txt",
        "description.md",
        "schema.json",
    ]

    found_description_files = []
    for desc_file in description_files:
        if os.path.exists(os.path.join(datasets_path, desc_file)):
            found_description_files.append(desc_file)

    if found_description_files:
        print("\nFound potential description files:")
        for f in found_description_files:
            print(f"  - {f}")
            # Read and print content of description files (first 1000 chars)
            try:
                with open(os.path.join(datasets_path, f), "r") as f_read:
                    content = f_read.read(1000) # Read first 1000 characters
                    print(f"\nContent of {f} (first 1000 chars):\n---\n{content}\n---")
            except Exception as e:
                print(f"Could not read {f}: {e}")
    else:
        print("\nNo common description files found.")

except FileNotFoundError:
    print(f"Error: The directory {datasets_path} does not exist in the sandbox.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
