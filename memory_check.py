import psutil

# Get the memory usage
memory_info = psutil.virtual_memory()

# Convert bytes to gigabytes
total_memory_gb = memory_info.total / (1024 ** 3)
available_memory_gb = memory_info.available / (1024 ** 3)
used_memory_gb = memory_info.used / (1024 ** 3)

# Print the memory usage details in GB
print(f"Total memory: {total_memory_gb:.2f} GB")
print(f"Available memory: {available_memory_gb:.2f} GB")
print(f"Used memory: {used_memory_gb:.2f} GB")
print(f"Memory usage percentage: {memory_info.percent}%")
