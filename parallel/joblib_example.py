from joblib import Parallel, delayed
import multiprocessing

# Define a function to apply
def square(x):
    return x ** 2

# Create a list of numbers
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a Parallel object with the desired number of jobs (processes)
num_cores = multiprocessing.cpu_count()
parallel = Parallel(n_jobs=num_cores)

# Apply the function to each element of the list in parallel
squared_nums = parallel(delayed(square)(num) for num in nums)

print(squared_nums)