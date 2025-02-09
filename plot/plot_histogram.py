import matplotlib.pyplot as plt

# histogram
plt.hist( distribution, bins=np.arange(1,1000,100), density=False, edgecolor='black' )  # bin size [1,100), [100,200), ...
plt.hist( distribution, bins=10, density=False )  # bins=10 means there will be 10 bins  
