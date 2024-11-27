import asyncio
import aiohttp
from pathlib import Path
import sys

file_ids = [
    # "b69u8",
    # "yjcqp",
    # "5evgd",
    # "6b4tq",
    # "vxrja",
    # "bcsm8",
    # "wftvu",
    # "xr4f9",
    "rh28b",
    "wf2xp",
]

# URL template for downloads
file_url = "https://osf.io/{file_id}/download"


async def download_file(session, file_id):
    """Download a single file asynchronously."""
    url = file_url.format(file_id=file_id)
    output_file = Path(f"downloaded_{file_id}.dat")

    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to download {file_id}. Status: {response.status}")
                return False

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(output_file, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (
                            (downloaded_size / total_size) * 100 if total_size else 0
                        )
                        print(f"{file_id}: {progress:.1f}%", end="\r", file=sys.stderr)

            print(f"\n{file_id}: Download complete")
            return True

    except Exception as e:
        print(f"\nError downloading {file_id}: {str(e)}")
        return False


async def download_all_files():
    """Download all files concurrently."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        tasks = [download_file(session, file_id) for file_id in file_ids]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        print(f"\nDownloaded {successful} out of {len(file_ids)} files successfully")


if __name__ == "__main__":
    asyncio.run(download_all_files())
