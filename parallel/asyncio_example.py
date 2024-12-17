import asyncio

def func(x, a):
    return x * x * a

async def run_func(x, a):
    # Wrap the synchronous function in an async context
    result = await asyncio.to_thread(func, x, a)
    return result

async def main():
    a = 2
    x_list = [1, 2, 3]

    # Use asyncio.gather to run tasks in parallel
    tasks = [run_func(x, a) for x in x_list]
    res = await asyncio.gather(*tasks)

    print("Results:", res)

# Run the asyncio event loop
asyncio.run(main())
