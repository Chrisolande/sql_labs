import asyncio
from collections.abc import Awaitable, Callable, Sequence

from aiolimiter import AsyncLimiter


async def process_in_batches[T, U](
    items: Sequence[T],
    processor: Callable[[T], Awaitable[U]],
    batch_size: int = 10,
    max_concurrency: int = 3,
    return_exceptions: bool = True,
    rate_per_second: float | None = None,
) -> list[U | BaseException]:
    from itertools import batched

    semaphore = asyncio.Semaphore(max_concurrency)
    limiter = AsyncLimiter(rate_per_second, 1) if rate_per_second else None

    async def _run(item: T) -> U:
        async with semaphore:
            if limiter:
                async with limiter:
                    return await processor(item)
            return await processor(item)

    results: list[U | BaseException] = []
    for batch in batched(items, batch_size, strict=False):
        tasks = [asyncio.create_task(_run(item)) for item in batch]
        batch_results = await asyncio.gather(
            *tasks, return_exceptions=return_exceptions
        )
        results.extend(batch_results)
    return results
