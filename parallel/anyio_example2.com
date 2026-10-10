import anyio

async def fetch_data(name: str):
    print(f"Starting {name}")
    await anyio.sleep(2)
    print(f"Finished {name}")

async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(fetch_data, "database")
        tg.start_soon(fetch_data, "LLM")
        tg.start_soon(fetch_data, "search")

anyio.run(main)
