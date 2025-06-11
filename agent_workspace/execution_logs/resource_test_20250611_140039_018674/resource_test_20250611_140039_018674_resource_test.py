
import time
import os
import sys

print(f"Starting resource consumption test. PID: {os.getpid()}")

# CPU intensive task for 60 seconds
print("Starting CPU intensive task for 60 seconds...")
start_time_cpu = time.time()
end_time_cpu = start_time_cpu + 60
i = 0
while time.time() < end_time_cpu:
    i += 1
    # Perform some arbitrary calculation to keep CPU busy
    _ = i * i * i / 2.0 + (i % 7)
print(f"CPU intensive task finished. Iterations: {i}")

# Memory allocation task (up to 17 GB)
print("Starting memory allocation task (up to 17 GB)...")
memory_to_allocate_gb = 17
chunk_size_mb = 100  # Allocate in 100 MB chunks
chunk_size_bytes = chunk_size_mb * 1024 * 1024
total_allocated_bytes = 0
allocated_chunks = []

try:
    while total_allocated_bytes < memory_to_allocate_gb * 1024 * 1024 * 1024:
        # Create a large byte array
        chunk = bytearray(chunk_size_bytes)
        allocated_chunks.append(chunk)
        total_allocated_bytes += chunk_size_bytes
        print(f"Allocated {total_allocated_bytes / (1024 * 1024):.2f} MB of memory.")
        # Brief pause to allow system to react and print output
        time.sleep(0.1)
    print(f"Successfully allocated approximately {total_allocated_bytes / (1024 * 1024 * 1024):.2f} GB of memory.")
except MemoryError:
    print("MemoryError: Failed to allocate more memory. Reached system/container limit.")
    print(f"Total memory allocated before error: {total_allocated_bytes / (1024 * 1024 * 1024):.2f} GB")
except Exception as e:
    print(f"An unexpected error occurred during memory allocation: {e}")
    print(f"Total memory allocated before error: {total_allocated_bytes / (1024 * 1024 * 1024):.2f} GB")

# Keep the allocated memory in scope until the script finishes
# This line is not strictly necessary as allocated_chunks is already in scope, but good for emphasis
# del allocated_chunks # This would release memory, but we want to hold it until script ends

print("Resource consumption test completed.")
