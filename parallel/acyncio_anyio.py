import anyio

async def worker(name):
    await anyio.sleep(1)
    print(name)

async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, "A")
        tg.start_soon(worker, "B")

anyio.run(main)
