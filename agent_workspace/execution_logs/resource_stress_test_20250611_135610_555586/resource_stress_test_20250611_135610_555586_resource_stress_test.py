
import time
import sys
import os

print("Starting resource consumption test...")

# --- CPU Intensive Task ---
cpu_duration = 60 # seconds
start_time_cpu = time.time()
print(f"Starting CPU intensive task for {cpu_duration} seconds...")
i = 0
while (time.time() - start_time_cpu) < cpu_duration:
    # Perform some arbitrary calculation
    _ = i * i + i / 2 - i % 3
    i += 1
print(f"CPU intensive task finished. Iterations: {i}")

# --- Memory Intensive Task ---
target_memory_gb = 17
print(f"Attempting to allocate {target_memory_gb} GB of memory...")

# A list to hold memory chunks
memory_chunks = []
chunk_size_mb = 100 # Allocate in 100 MB chunks
chunk_size_bytes = chunk_size_mb * 1024 * 1024
total_chunks_needed = int(target_memory_gb * 1024 / chunk_size_mb)

print(f"Allocating {total_chunks_needed} chunks of {chunk_size_mb} MB each...")

try:
    for j in range(total_chunks_needed):
        # Create a byte array of chunk_size_bytes
        chunk = bytearray(chunk_size_bytes)
        memory_chunks.append(chunk)
        if (j + 1) % 10 == 0: # Print progress every 10 chunks
            print(f"Allocated {(j + 1) * chunk_size_mb} MB...")
    print(f"Successfully attempted to allocate {target_memory_gb} GB of memory.")
except Exception as e:
    print(f"Error during memory allocation: {e}")
    print("Memory allocation might have failed due to insufficient resources.")

print("Resource consumption test finished.")
print("Please check the system's resource monitor on your host machine to observe the impact.")
