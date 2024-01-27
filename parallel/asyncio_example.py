# Synthetic Difference in Difference.
# https://pypi.org/project/synthdid/

import pdb  
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import numpy as np
import asyncio


import asyncio

async def my_coroutine(task_name):
    print(f"{task_name} Start")
    await asyncio.sleep(1)
    print(f"{task_name} End")

async def main():
    tasks = [asyncio.create_task(my_coroutine(f"Task-{i}")) for i in range(1, 4)]
    await asyncio.gather(*tasks)

asyncio.run(main())
