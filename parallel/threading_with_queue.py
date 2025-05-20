# Treading with queue.  Use queue when you have many threads(workers).
# https://python.plainenglish.io/from-single-script-to-scalable-beast-how-i-built-a-python-task-queue-system-that-runs-10x-faster-d4a5e9874244


import queue
import threading

task_queue = queue.Queue()

def worker():
    while not task_queue.empty():
        url = task_queue.get()
        download_page(url)
        task_queue.task_done()

for url in urls:
    task_queue.put(url)

for _ in range(5):
    t = threading.Thread(target=worker)
    t.start()

task_queue.join()