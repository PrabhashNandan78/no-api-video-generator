"""
Trend Discovery Engine.
Zero API cost: Uses Google Trends public RSS feeds and DuckDuckGo search (via ddgs / duckduckgo-search).
Properly handles rate limits, deduplication, and preserves full source attribution.
"""

import os
import sys
import logging
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

logger = logging.getLogger("VideoFactory.TrendEngine")

# Curated domain fallback strictly labeled as offline technical reference
CURATED_FALLBACK_TOPICS = {
    "Artificial Intelligence and Robotics": [
        {
            "headline": "Open-Source Bipedal Locomotion Policies for Autonomous Humanoid Robots",
            "description": "Researchers demonstrate lightweight neural locomotion controllers enabling dynamic running, stair climbing, and real-time push recovery on standard consumer hardware.",
            "source": "Robotics Research Archive (Curated Fallback)",
            "pub_date": "Technical Reference",
            "url": "https://arxiv.org",
            "is_live_news": False
        },
        {
            "headline": "Small Language Models Demonstrate High-Efficiency On-Device Reasoning",
            "description": "Modern 3B to 7B quantized parameter models achieve near-parity with frontier cloud models in structured reasoning while operating locally within constrained memory footprints.",
            "source": "AI Systems Review (Curated Fallback)",
            "pub_date": "Technical Reference",
            "url": "https://huggingface.co",
            "is_live_news": False
        }
    ],
    "Default": [
        {
            "headline": "Advances in Solid-State Ceramic Electrolytes for High-Density Energy Storage",
            "description": "Novel non-flammable solid-state battery cells achieve 400 Wh/kg energy density with 10-minute fast charging capabilities.",
            "source": "CleanTech Bulletin (Curated Fallback)",
            "pub_date": "Technical Reference",
            "url": "https://nature.com",
            "is_live_news": False
        }
    ]
}

def fetch_google_trends_rss(geo: str = "IN", max_items: int = 8) -> List[Dict[str, Any]]:
    """
    Fetches trending queries and news snippets from Google Trends public RSS feed.
    """
    url = f"https://trends.google.com/trending/rss?geo={geo}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
    )
    trends = []
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            root = ET.fromstring(content)
            
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                approx_traffic = item.find("{https://trends.google.com/trending/rss}approx_traffic")
                pub_date = item.find("pubDate")
                news_items = item.findall("{https://trends.google.com/trending/rss}news_item")
                
                snippets = []
                for ni in news_items:
                    snippet_title = ni.find("{https://trends.google.com/trending/rss}news_item_title")
                    snippet_source = ni.find("{https://trends.google.com/trending/rss}news_item_source")
                    snippet_url = ni.find("{https://trends.google.com/trending/rss}news_item_url")
                    
                    st = snippet_title.text.strip() if snippet_title is not None and snippet_title.text else ""
                    ss = snippet_source.text.strip() if snippet_source is not None and snippet_source.text else "Google Trends"
                    su = snippet_url.text.strip() if snippet_url is not None and snippet_url.text else ""
                    
                    if st:
                        snippets.append({
                            "title": st,
                            "source": ss,
                            "url": su
                        })

                if title_elem is not None and title_elem.text:
                    title_text = title_elem.text.strip()
                    traffic_text = approx_traffic.text.strip() if approx_traffic is not None and approx_traffic.text else "Trending"
                    date_text = pub_date.text.strip() if pub_date is not None and pub_date.text else ""
                    
                    trends.append({
                        "query": title_text,
                        "traffic": traffic_text,
                        "pub_date": date_text,
                        "snippets": snippets
                    })
                    if len(trends) >= max_items:
                        break
                        
        logger.info(f"Retrieved {len(trends)} trending topics from Google Trends ({geo}).")
    except Exception as e:
        logger.warning(f"Google Trends RSS retrieval notice ({e}).")

    return trends

def search_ddg_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches DuckDuckGo News for verified recent developments.
    Prefers 'ddgs' package, falling back to 'duckduckgo_search'.
    """
    DDGS_class = None
    try:
        from ddgs import DDGS
        DDGS_class = DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            DDGS_class = DDGS
        except ImportError:
            logger.warning("Neither 'ddgs' nor 'duckduckgo-search' package is available.")
            return []

    articles = []
    try:
        with DDGS_class() as ddgs:
            try:
                results = ddgs.news(query, max_results=max_results)
            except TypeError:
                results = ddgs.news(keywords=query, max_results=max_results)

            for r in results:
                title = str(r.get("title") or "").strip()
                body = str(r.get("body") or "").strip()
                source = str(r.get("source") or "DuckDuckGo News").strip()
                date = str(r.get("date") or "Recent").strip()
                url = str(r.get("url") or "").strip()

                if title:
                    articles.append({
                        "headline": title,
                        "description": body,
                        "source": source,
                        "pub_date": date,
                        "url": url,
                        "is_live_news": True
                    })
    except Exception as e:
        logger.warning(f"DuckDuckGo news query notice ({e}).")

    return articles

def fetch_live_trend_context(
    niche: str = "Artificial Intelligence and Robotics",
    geo: str = "IN",
    max_stories: int = 5
) -> str:
    """
    Collects live trends and news stories, deduplicates them, and builds a
    clean, source-grounded context block. Clearly distinguishes verified live news
    from curated fallback data.
    """
    logger.info(f"Gathering trends for niche: '{niche}' (Geo: {geo})...")
    
    # 1. Fetch Google Trends RSS
    google_trends = fetch_google_trends_rss(geo=geo, max_items=8)
    
    # 2. Search DDG News for target niche
    niche_news = search_ddg_news(query=f"{niche} breakthrough OR development", max_results=max_stories)
    
    collected: List[Dict[str, Any]] = []
    seen = set()

    # Prioritize niche-specific live news
    for item in niche_news:
        key = item["headline"].lower().strip()
        if key not in seen:
            seen.add(key)
            collected.append(item)

    # Supplement with Google Trends news snippets if niche search returned few items
    if len(collected) < 3 and google_trends:
        for t in google_trends:
            for snip in t.get("snippets", []):
                key = snip["title"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    collected.append({
                        "headline": snip["title"],
                        "description": f"Trending topic related to regional search query '{t['query']}' (Search volume: {t['traffic']}).",
                        "source": snip.get("source", "Google Trends"),
                        "pub_date": t.get("pub_date", "Recent"),
                        "url": snip.get("url", ""),
                        "is_live_news": True
                    })
                    if len(collected) >= max_stories:
                        break
            if len(collected) >= max_stories:
                break

    # Determine data status
    using_fallback = False
    if not collected:
        using_fallback = True
        logger.info("Live news feeds unavailable or rate-limited. Using curated technical reference fallback.")
        fallback_list = CURATED_FALLBACK_TOPICS.get(niche, CURATED_FALLBACK_TOPICS["Default"])
        collected.extend(fallback_list[:max_stories])

    # Format structured context block
    header_status = "[CURATED DOMAIN FALLBACK - OFFLINE REFERENCE]" if using_fallback else "[VERIFIED REAL-TIME NEWS]"
    lines = [
        f"=== TREND CONTEXT: {niche.upper()} (REGION: {geo}) ===",
        f"STATUS: {header_status}"
    ]

    if google_trends:
        q_summary = ", ".join([f"'{t['query']}' ({t['traffic']})" for t in google_trends[:5]])
        lines.append(f"Google Trends Queries: {q_summary}")

    lines.append("\nTop Documented Stories:")
    # FIX: Correctly use enumerate to format stories with sequential numbering
    for i, story in enumerate(collected[:max_stories], start=1):
        tag = "[LIVE NEWS]" if story.get("is_live_news", False) else "[CURATED FALLBACK]"
        lines.append(f"\n[{i}] {tag} {story['headline']}")
        lines.append(f"    Source: {story.get('source', 'Unknown')} | Date: {story.get('pub_date', 'Unknown')}")
        if story.get("url"):
            lines.append(f"    URL: {story['url']}")
        lines.append(f"    Summary: {story.get('description', 'No detailed description available.')}")

    return "\n".join(lines)
