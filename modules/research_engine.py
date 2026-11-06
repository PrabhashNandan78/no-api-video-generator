"""
Source-Grounded Topic Research and Brief Engine.
Extracts factual evidence from retrieved stories without fabricating unverified statistics.
"""

import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("VideoFactory.ResearchEngine")

def evaluate_and_select_topic(trend_context: str, niche: str) -> Dict[str, Any]:
    """
    Parses retrieved stories from trend context, selects the most relevant evidence-backed topic,
    and produces a source-grounded research brief. Avoids fabricating statistics.
    """
    logger.info("Evaluating retrieved topics for source grounding and audience relevance...")
    
    # Parse available stories from trend context
    parsed_stories = []
    current_story: Dict[str, Any] = {}
    
    is_fallback = "STATUS: [CURATED DOMAIN FALLBACK" in trend_context
    
    for line in trend_context.splitlines():
        line_s = line.strip()
        if line_s.startswith("[") and "]" in line_s and not line_s.startswith("=== ") and not line_s.startswith("STATUS:"):
            # Header like [1] [LIVE NEWS] Headline or [1] Headline
            if current_story and current_story.get("headline"):
                parsed_stories.append(current_story)
            current_story = {"headline": "", "source": "Unknown", "date": "Unknown", "url": "", "summary": ""}
            parts = line_s.split("]", 1)
            raw_title = parts[1].strip()
            if raw_title.startswith("[LIVE NEWS]"):
                current_story["headline"] = raw_title.replace("[LIVE NEWS]", "").strip()
            elif raw_title.startswith("[CURATED FALLBACK]"):
                current_story["headline"] = raw_title.replace("[CURATED FALLBACK]", "").strip()
            else:
                current_story["headline"] = raw_title
        elif line_s.startswith("Source:"):
            parts = line_s.split("|")
            current_story["source"] = parts[0].replace("Source:", "").strip()
            if len(parts) > 1 and "Date:" in parts[1]:
                current_story["date"] = parts[1].replace("Date:", "").strip()
        elif line_s.startswith("URL:"):
            current_story["url"] = line_s.replace("URL:", "").strip()
        elif line_s.startswith("Summary:"):
            current_story["summary"] = line_s.replace("Summary:", "").strip()

    if current_story and current_story.get("headline"):
        parsed_stories.append(current_story)

    # Select the best documented story
    if parsed_stories:
        # Prioritize stories with substantial descriptions
        best_story = max(parsed_stories, key=lambda s: len(s.get("summary", "")))
    else:
        best_story = {
            "headline": f"Current Innovations in {niche}",
            "source": "Curated Domain Archive",
            "date": "Reference",
            "url": "https://arxiv.org",
            "summary": f"Exploration of active technical developments in {niche}."
        }

    # Ground key facts strictly in the available summary rather than hallucinating 10x claims
    headline = best_story["headline"]
    source = best_story.get("source", "News Wire")
    date = best_story.get("date", "Recent")
    url = best_story.get("url", "")
    summary = best_story.get("summary", "")

    # Extract source-grounded bullet points
    key_facts = []
    if summary and summary != "No detailed description available.":
        key_facts.append(f"Reported development: {summary}")
    else:
        key_facts.append(f"Primary development announced: {headline}")
    
    key_facts.append(f"Attributed Source: {source} (Published: {date})")
    if url:
        key_facts.append(f"Verification URL: {url}")
    key_facts.append("Factual Note: Technical claims must remain strictly aligned with documented source reports without fabricating unverified metrics.")

    evidence_status = "Curated Technical Reference (Offline Fallback)" if is_fallback else f"Verified Live Source: {source} ({date})"

    research_brief = {
        "selected_topic": headline,
        "niche": niche,
        "evidence_status": evidence_status,
        "is_live_news": not is_fallback,
        "primary_source": source,
        "publication_date": date,
        "source_url": url,
        "summary": summary,
        "why_it_matters": (
            f"This development was reported by {source} and represents a noteworthy milestone in {niche}."
        ),
        "key_facts": key_facts,
        "narrative_hook": f"What is happening right now in {niche}? Here is the documented report.",
        "call_to_action": "Follow for source-verified tech updates and real AI developments."
    }

    logger.info(f"Selected Topic: '{research_brief['selected_topic']}' [Status: {evidence_status}]")
    return research_brief
