"""
SEO and Social Caption Engine.
Formats multi-platform viral captions and exports structured video metadata.
Identifies Piper TTS as the CPU voice engine.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("VideoFactory.SEOEngine")

def generate_social_deliverables(
    video_plan: Any,
    actual_durations: List[float],
    total_duration: float,
    output_dir: Path,
    engine_used: str = "sdxl"
) -> Dict[str, str]:
    """
    Creates and saves:
    1. caption.txt (optimized for YouTube Shorts, Reels, TikTok)
    2. video_metadata.json (production telemetry with Piper TTS metadata)
    """
    topic = video_plan.selected_topic
    hook = video_plan.hook_title
    hashtags = " ".join(video_plan.hashtags[:6])

    caption_content = f"""=== VIRAL SOCIAL MEDIA DELIVERABLE ===

[TITLE / HOOK]
{hook}

[YOUTUBE SHORTS DESCRIPTION]
{hook} 🔥
{video_plan.social_caption}

⏱️ TIMESTAMPS:
0:00 - {video_plan.scenes[0].subtitle_text}
0:15 - Breakdown & Evidence
0:35 - Key Takeaway
0:45 - What's Next?

Subscribe to stay ahead of the curve! 💡
{hashtags} #Shorts #Technology

[INSTAGRAM REELS / TIKTOK CAPTION]
{hook} ⚡
{video_plan.social_caption}
Drop your thoughts in the comments! 👇
.
.
{hashtags} #viral #explorepage #tech
"""

    caption_file = output_dir / "caption.txt"
    with open(caption_file, "w", encoding="utf-8") as f:
        f.write(caption_content)

    scene_telemetry = []
    elapsed = 0.0
    for i, sc in enumerate(video_plan.scenes):
        dur = actual_durations[i] if i < len(actual_durations) else 5.0
        scene_telemetry.append({
            "scene_number": sc.scene_number,
            "start_time_seconds": round(elapsed, 2),
            "duration_seconds": round(dur, 2),
            "visual_prompt": sc.visual_prompt,
            "voiceover_text": sc.voiceover_text,
            "subtitle_text": sc.subtitle_text
        })
        elapsed += dur

    metadata = {
        "project": "Zero-API AI Video Factory",
        "video_title": hook,
        "selected_topic": topic,
        "target_audience": video_plan.target_audience,
        "total_duration_seconds": round(total_duration, 2),
        "aspect_ratio": "9:16",
        "resolution": "720x1280",
        "frame_rate": 25,
        "visual_engine": engine_used,
        "llm_engine": "Qwen/Qwen2.5-7B-Instruct (4-bit NF4)",
        "tts_engine": "Piper TTS (CPU, en_US-lessac-medium)",
        "audio_sample_rate_hz": 22050,
        "hashtags": video_plan.hashtags,
        "scene_count": len(video_plan.scenes),
        "scenes": scene_telemetry
    }

    metadata_file = output_dir / "video_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported SEO deliverables to: {caption_file} and {metadata_file}")
    return {
        "caption_path": str(caption_file),
        "metadata_path": str(metadata_file)
    }
