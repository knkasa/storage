import anyio
import httpx

async def fetch_status(client: httpx.AsyncClient, url: str) -> None:
    response = await client.get(url, timeout=5)
    response.raise_for_status()
    print(f"{url}: {response.status_code}")

async def check_services(urls: list[str]) -> None:
    async with httpx.AsyncClient() as client:
        async with anyio.create_task_group() as task_group:
            for url in urls:
                task_group.start_soon(fetch_status, client, url)

anyio.run(
    check_services,
    [
        "https://api.example.com/health",
        "https://worker.example.com/health",
        "https://billing.example.com/health",
    ],
)
