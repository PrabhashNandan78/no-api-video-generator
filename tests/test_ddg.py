import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from duckduckgo_search import DDGS
    print("Found duckduckgo_search package")
    results = DDGS().news(keywords="artificial intelligence", max_results=3)
    print(f"DDGS news results: {len(results)}")
    for r in results:
        print(f"Title: {r.get('title')}, Source: {r.get('source')}")
except Exception as e:
    print(f"DDGS error or not found: {e}")

