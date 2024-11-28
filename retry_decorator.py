import functools
import time
from ray.exceptions import RayActorError, RayTaskError
# You could use Tenacity library too.
from tenacity import retry, stop_after_attempt

class myClass():
    @staticmethod
    def retry(max_tries=3, delay=10, exceptions=(RayActorError, RayTaskError, RuntimeError, TimeoutError)):
        def decorator_retry(func):
            @functools.wraps(func)
            def wrapper_retry(*args, **kwargs):
                tries = 0
                while tries < max_tries:
                    try:
                        retrn func(*args, **kwargs)
                    except exceptions as e:
                        tries += 1
                        if tries==max_tries:
                            raise e
                        print(f"Attempt {tries} failed. Retrying...")
                        time.sleep(delay)
            return wrapper_retry
        return decorator_retry

    def main(self):
        
        @ray.remote()   # ray also hase built-in "max_tries" arguments too.
        @self.retry()
        def run_parallel(i):
            raise RuntimeError

        ray.init()
        res_list=[]
        for i in range(i):
            res = run_parallel.remote(i)
            res_list.append(res)
        results=ray.get(res_list)
        ray.shutdown()



