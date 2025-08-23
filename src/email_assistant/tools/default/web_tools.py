import os
from typing import List
from langchain_core.tools import tool

try:
    from googleapiclient.discovery import build  # type: ignore
except Exception:
    build = None  # type: ignore


def _format_results(items: List[dict]) -> str:
    lines = []
    for i, it in enumerate(items or [], start=1):
        title = it.get("title", "")
        link = it.get("link", "")
        snippet = it.get("snippet", "") or (it.get("pagemap", {}).get("metatags", [{}])[0].get("og:description", ""))
        lines.append(f"{i}. {title}\n{link}\n{snippet}\n")
    return "\n".join(lines) if lines else "No results."


@tool
def google_search(query: str, num_results: int = 5) -> str:
    """Search Google for up-to-date information. Requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX env vars."""
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_CX")
    if not api_key or not cx:
        return "Google Custom Search not configured. Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX."
    if build is None:
        return "google-api-python-client is not available to perform search."
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        resp = service.cse().list(q=query, cx=cx, num=min(max(int(num_results), 1), 10)).execute()
        items = resp.get("items", [])
        return _format_results(items)
    except Exception as e:
        return f"Search error: {e}"

