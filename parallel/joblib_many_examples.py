# Examples of Joblib

#--------- Multi-process -----------------
from math import factorial
from joblib import Parallel, delayed

def heavy(x: int) -> int:
    return factorial(x)          # CPU-bound

numbers = list(range(10_000))
result = Parallel(n_jobs=-1)(delayed(heavy)(n) for n in numbers)

print(result[:5])


#----------- Threading with I/O task -------------------
import requests, time
from joblib import Parallel, delayed

URLS = [
    "https://example.com",
    "https://httpbin.org/delay/2",
    "https://python.org",
    "https://www.wikipedia.org"
]

def fetch(url: str) -> tuple[str, float]:
    start = time.perf_counter()
    r = requests.get(url, timeout=10)
    duration = time.perf_counter() - start
    return url, duration, len(r.content)

results = Parallel(n_jobs=4, prefer="threads")(
    delayed(fetch)(u) for u in URLS
)

for url, dt, size in results:
    print(f"{url} → {size} bytes in {dt:.2f}s")
	

#------- Multi process with virtual memory ----------
import numpy as np
from joblib import Parallel, delayed
import os, tempfile

tmp = tempfile.mkdtemp()
file = os.path.join(tmp, "sat_images.mmap")

# Save a dummy 10 000-image cube
cube = np.random.randint(0, 256, size=(10_000, 1024, 1024), dtype=np.uint8)
fp = np.memmap(file, dtype='uint8', mode='w+', shape=cube.shape)
fp[:] = cube[:]
del fp   # Flush to disk

# Re-open as memmap
imgs = np.memmap(file, dtype='uint8', mode='r', shape=cube.shape)

def mean_intensity(img):
    return img.mean()

means = Parallel(n_jobs=8)(
    delayed(mean_intensity)(imgs[i]) for i in range(imgs.shape[0])
)

print(f"Computed mean brightness for {len(means)} images.")


#---------- Multi process with Caching in local ------------
from joblib import Memory
import requests, datetime as dt

memory = Memory("~/.cache/joblib_currency", verbose=0)

@memory.cache
def get_rate(date: str, pair="EURUSD") -> float:
    print(f"Requesting FX rate for {date}")
    r = requests.get(
        f"https://api.exchangerate.host/{date}",
        params={"base": pair[:3], "symbols": pair[3:]}
    ).json()
    return r["rates"][pair[3:]]

today = dt.date.today().isoformat()
print(get_rate(today))
print(get_rate(today))    # Instant: served from cache


