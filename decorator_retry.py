import functools
import time

def retry(max_tries: int = 3, delay_seconds: int = 5):
    """
    A decorator that retries a function if it raises a NumberTooLowError.
    Args:
        max_tries (int): Maximum number of attempts to retry the function. Defaults to 3.
        delay_seconds (int): Number of seconds to wait between retries. Defaults to 5.
    Returns:
        Callable: A decorator function that can be applied to other functions.
    """
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    # Attempt to execute the decorated function
                    return func(*args, **kwargs)
                except NumberTooLowError as e:
                    tries += 1
                    if tries == max_tries:
                        # If we've reached the maximum number of tries, re-raise the exception
                        raise e
                    print(
                        f"Attempt {tries} failed. Retrying in {delay_seconds} seconds..."
                    )
                    time.sleep(delay_seconds)
        return wrapper_retry
    return decorator_retry

# You could use Tenacity library too.
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
    def function():
        print("Attempting to execute ...")
        raise Exception("Failed")
function()
