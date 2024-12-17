from joblib import Parallel, delayed

def func(x, a):
    return x * x * a

a = 2
x_list = [1, 2, 3]

# Choose "threading" or "multiprocessing".
results = Parallel(n_jobs=2, backend="threading")(
    delayed(func)(x, a) for x in x_list
)

print("Results:", results)
