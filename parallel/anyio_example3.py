# in this example, "Doiing other work" may not finish, but get_data() can start running, then finaly print(result).

import asyncio

async def get_data():
    print("A: Requesting data")
    await asyncio.sleep(3)  # Simulate waiting for a database
    print("B: Data received")
    return 100

async def main():
    task = asyncio.create_task(get_data())

    print("C: Doing other work")

    result = await task  # start running get_data() here.
    print("D: Result:", result)

asyncio.run(main())
