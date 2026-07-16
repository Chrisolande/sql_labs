from collections.abc import AsyncIterator, Callable
from typing import Any


async def paginate_api[T](
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[dict], list[T]],
    per_page: int = 50,
    max_results: int | None = None,
    page_cap: int = 50,
) -> AsyncIterator[T]:
    page = 1
    yielded = 0

    while True:
        data = await fetch_page(page, per_page)
        items = extract_items(data) if isinstance(data, dict) else []
        if not items:
            break
        for item in items:
            yield item
            yielded += 1
            if max_results and yielded >= max_results:
                return
        if len(items) < per_page:
            break
        found = data.get("found", 0) if isinstance(data, dict) else 0
        if page * per_page >= found:
            break
        if page >= page_cap:
            break
        page += 1
