import asyncio

async def sync_files():
    ...

async def send_report():
    ...

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(sync_files())
        tg.create_task(send_report())

asyncio.run(main())
