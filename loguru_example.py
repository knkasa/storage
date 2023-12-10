import pdb
import os
from loguru import logger
import time
from memory_profiler import profile

# Run as pyhton -m memory_profiler loguru.py


@profile
def logging():

    t = time.perf_counter()

    logger.debug(f"This is a debug message at {t}.")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

if __name__=="__main__":

    os.chdir('C:/Users/knkas/Desktop/NLP_example')

    logger.add(
        "example.log", 
        rotation="100MB", 
        level="DEBUG", 
        
        # {level: <8} means put 8 spaces after the level text.  
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}", 
    )

    logging()
    
