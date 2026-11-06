import urllib.request
import xml.etree.ElementTree as ET
import json
import logging
from typing import List, Dict, Any

def test_trends_rss(geo: str = "IN"):
    url = f"https://trends.google.com/trending/rss?geo={geo}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            root = ET.fromstring(content)
            items = []
            for item in root.findall(".//item"):
                title = item.find("title")
                approx_traffic = item.find("{https://trends.google.com/trending/rss}approx_traffic")
                pub_date = item.find("pubDate")
                news_items = item.findall("{https://trends.google.com/trending/rss}news_item")
                news_titles = [n.find("{https://trends.google.com/trending/rss}news_item_title").text for n in news_items if n.find("{https://trends.google.com/trending/rss}news_item_title") is not None]
                if title is not None and title.text:
                    items.append({
                        "title": title.text.strip(),
                        "traffic": approx_traffic.text if approx_traffic is not None else "N/A",
                        "pub_date": pub_date.text if pub_date is not None else "",
                        "news_snippets": news_titles[:2]
                    })
            print(f"Retrieved {len(items)} trending items for geo={geo}")
            for it in items[:5]:
                print(f" - {it['title']} ({it['traffic']}): {it['news_snippets']}")
            return items
    except Exception as e:
        print(f"Error fetching Google Trends RSS: {e}")
        return []

if __name__ == "__main__":
    test_trends_rss()

