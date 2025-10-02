import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 6)  # input
y = X.mean(axis=1)*1.0  # target
y_pred = y + 0.1*np.random.rand(100)

# Histogram for each features.
d = X.shape[1]
cols = min(4, d)
rows = int(np.ceil(d / cols))
plt.figure(figsize=(14, 3*rows))
for j in range(d):
    plt.subplot(rows, cols, j+1)
    plt.hist(X[:, j], bins=40)
    plt.title(f'Feature {j}')
plt.tight_layout()
plt.show(); plt.close()

# Outlier detection: A sudden elbow reveals a natural cut-off. Those top-k outliers? Audit them before they poison your loss.
mu = X.mean(axis=0, keepdims=True)
sigma = X.std(axis=0, keepdims=True) + 1e-9
z = np.abs((X - mu) / sigma).max(axis=1)  # max anomaly per row
idx = np.argsort(z)
plt.figure(figsize=(10,4))
plt.plot(z[idx], 'x')
plt.title('Max |z| per sample (sorted)')
plt.xlabel('Samples'); plt.ylabel('Outlier score')
plt.show(); plt.close()

# Visualize Nulls.
mask = ~np.isnan(X)
stripe = mask.mean(axis=0)  # share present by feature
plt.figure(figsize=(10,2))
plt.imshow(stripe[None, :], aspect='auto')
plt.yticks([]); plt.xticks(range(X.shape[1]))
plt.title('Feature presence (1=present, 0=missing)')
plt.show(); plt.close()

# Feature correlation Heat map.
C = np.corrcoef(X, rowvar=False)
plt.figure(figsize=(6,5))
plt.imshow(C, vmin=-1, vmax=1)
plt.colorbar(); plt.title('Feature correlation')
plt.show(); plt.close()

#------------ Regression -----------------------------
# Target vs. Preds: Bucket the true target and inspect residuals by bin. “Where do we over/under-predict?”
y = np.asarray(y).ravel()
y_pred = np.asarray(y_pred).ravel()
q = np.quantile(y, np.linspace(0,1,11))
bins = np.digitize(y, q[1:-1])
resid = y - y_pred
bin_mu = np.array([resid[bins==b].mean() for b in range(10)])
plt.figure(figsize=(8,4))
plt.bar(range(10), bin_mu)
plt.title('Mean residual by target decile'); plt.xlabel('Target decile'); plt.ylabel('Mean residual')
plt.show(); plt.close()

# Feature/Target pearson correlaton.
def safe_corr(a, b):
    a = (a - a.mean())/(a.std()+1e-9)
    b = (b - b.mean())/(b.std()+1e-9)
    return (a*b).mean()
sn = np.array([safe_corr(X[:, j], y) for j in range(X.shape[1])])
plt.figure(figsize=(10,2))
plt.stem(sn, use_line_collection=True)
plt.title('Feature↔Target correlation')
plt.xlabel('Feature idx'); plt.ylabel('corr')
plt.show(); plt.close()

# Partial Residual Curves: 
# If you see flat linear line near zero, the feature is good. If not, you may need feature-engineer(eg. log/square the feature)
# If you see peaks, try binning the feature.
j = 0  # pick a feature
resid = y - y_pred
xj = X[:, j]
order = np.argsort(xj)
xs, rs = xj[order], resid[order]
w = 10 # smooth via simple window mean
pad = w//2
rs_pad = np.pad(rs, (pad,pad), mode='edge')
smooth = np.convolve(rs_pad, np.ones(w)/w, mode='valid')
plt.figure(figsize=(8,4))
plt.plot(xs, smooth[1:])
plt.title(f'Partial residual vs Feature {j}')
plt.xlabel(f'X[:, {j}]'); plt.ylabel('Residual (smoothed)')
plt.show(); plt.close()

#------------ Classification ------------------
# Class Boundary Strip (for Binary): Plotting histogram along two controids(mu0 & mu1, see below)
y = np.round(y)
mu0 = X[y==0].mean(axis=0); mu1 = X[y==1].mean(axis=0) # compute class centroids.
w = mu1 - mu0
proj = X @ w
plt.figure(figsize=(8,4))
plt.hist(proj[y==0], bins=40, alpha=0.6, label='0')
plt.hist(proj[y==1], bins=40, alpha=0.6, label='1')
plt.legend(); plt.title('Projected separation'); plt.xlabel('distance(1D)')
plt.show(); plt.close()

# Calibration plot: if y_pred equals the actual probability, the points will be along diagonal.  
# Note: when model yields 0.4, it may not mean the probability of the class is 0.4%.  You may need to calibrate if not.  
p = np.clip(y_pred, 1e-6, 1-1e-6)  # probabilities
edges = np.linspace(0,1,11)
bins = np.digitize(p, edges[1:-1])
bin_p = np.array([p[bins==b].mean() for b in range(10)])
bin_y = np.array([y[bins==b].mean() for b in range(10)])
plt.figure(figsize=(5,5))
plt.plot([0,1], [0,1], '--')
plt.scatter(bin_p, bin_y)
plt.xlabel('Predicted'); plt.ylabel('Observed'); plt.title('Calibration')
plt.show(); 