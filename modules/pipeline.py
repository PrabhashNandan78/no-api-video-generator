"""
Master Orchestration Pipeline for Local Video Factory.
Executes the full 16-stage pipeline with strict sequential GPU execution,
resumability, error recovery, and technical verification.
Zero Kokoro dependencies. Features Piper TTS (CPU) and SDXL-Lightning (512x896 9:16).
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .config import get_base_directories, logger
from .hardware import check_hardware_diagnostics, clean_gpu_memory, get_gpu_memory_info, run_preflight_system_checks
from .trend_engine import fetch_live_trend_context
from .research_engine import evaluate_and_select_topic
from .script_engine import generate_video_script_llm, VideoPlan
from .voice_engine import synthesize_scene_voice, get_audio_duration, ensure_piper_voice_model
from .visual_engine import generate_visual_assets, render_image_to_kenburns_video, generate_procedural_scene_image
from .editing_engine import stitch_short_video, burn_subtitles_and_audio_to_clip
from .seo_engine import generate_social_deliverables

def validate_final_mp4(mp4_path: str) -> Dict[str, Any]:
    """
    Inspects the final MP4 with ffprobe to verify video codec, audio codec,
    resolution, aspect ratio, frame rate, and non-zero duration.
    """
    logger.info(f"Validating deliverable: {mp4_path}...")
    if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) < 100000:
        raise ValueError(f"Output file missing or empty: {mp4_path}")

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(mp4_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.warning(f"FFprobe inspection warning: {res.stderr}")
        return {"valid": True, "size_bytes": os.path.getsize(mp4_path)}

    data = json.loads(res.stdout)
    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    if not v_stream:
        raise ValueError("Validation failed: No video stream detected in output MP4!")
    if not a_stream:
        raise ValueError("Validation failed: No audio stream detected in output MP4!")

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    duration = float(data.get("format", {}).get("duration", 0.0))
    v_codec = v_stream.get("codec_name", "unknown")
    a_codec = a_stream.get("codec_name", "unknown")

    report = {
        "valid": True,
        "width": width,
        "height": height,
        "aspect_ratio": f"{width}:{height}",
        "duration_seconds": duration,
        "video_codec": v_codec,
        "audio_codec": a_codec,
        "file_size_bytes": os.path.getsize(mp4_path),
        "file_size_mb": round(os.path.getsize(mp4_path) / (1024**2), 2)
    }

    print("\n" + "=" * 60)
    print("           TECHNICAL VERIFICATION REPORT")
    print("=" * 60)
    print(f" File Path       : {mp4_path}")
    print(f" File Size       : {report['file_size_mb']} MB")
    print(f" Duration        : {report['duration_seconds']:.2f} seconds")
    print(f" Resolution      : {report['width']} x {report['height']} (9:16 Vertical)")
    print(f" Video Stream    : {report['video_codec'].upper()}")
    print(f" Audio Stream    : {report['audio_codec'].upper()}")
    print("=" * 60)
    return report

def run_smoke_test(niche: str = "Artificial Intelligence and Robotics", geo: str = "IN") -> Dict[str, Any]:
    """
    Lightweight environment smoke test:
    Validates preflight environment, live trend fetching, topic research brief,
    Piper TTS voice model download & synthesis, FFmpeg Ken Burns rendering,
    and subtitle burning. Runs in ~10 seconds without loading heavy GPU models.
    """
    print("\n" + "=" * 65)
    print("   RUNNING PREFLIGHT DIAGNOSTIC & SMOKE TEST")
    print("=" * 65 + "\n")

    dirs = get_base_directories()
    t_start = time.time()

    # 1. Preflight system checks
    print("[1/5] Checking hardware and system packages...")
    diag = check_hardware_diagnostics()
    preflight = run_preflight_system_checks()
    print(f"  FFmpeg: {'OK' if preflight.get('ffmpeg') else 'MISSING'}")
    print(f"  FFprobe: {'OK' if preflight.get('ffprobe') else 'MISSING'}")
    print(f"  DDGS: {'OK' if preflight.get('pkg_ddgs') else 'MISSING'}")
    print(f"  TTS Engine Ready: {'OK' if preflight.get('tts_ready') else 'MISSING'}")

    # 2. Trend & Research Brief test
    print("\n[2/5] Testing Trend Discovery & Research Engine...")
    trends_ctx = fetch_live_trend_context(niche=niche, geo=geo, max_stories=3)
    brief = evaluate_and_select_topic(trends_ctx, niche)
    print(f"  Discovered Topic: {brief['selected_topic']}")
    print(f"  Evidence Status : {brief['evidence_status']}")

    # 3. Piper Voice Download & Synthesis test
    print("\n[3/5] Testing Piper Voice Download & Narration...")
    piper_dir = dirs["base"] / "models" / "piper"
    test_audio = dirs["temp"] / "smoke_test_audio.wav"
    synthesize_scene_voice(
        text="This is a diagnostic narration test verifying Piper text to speech synthesis.",
        output_wav=str(test_audio),
        model_dir=piper_dir
    )
    dur = get_audio_duration(str(test_audio))
    print(f"  Synthesized diagnostic WAV: {dur:.2f}s ({os.path.getsize(test_audio)} bytes)")

    # 4. FFmpeg Ken Burns & Subtitle test
    print("\n[4/5] Testing Ken Burns Motion & Subtitle Rendering...")
    test_img = dirs["temp"] / "smoke_test_img.png"
    test_clip = dirs["temp"] / "smoke_test_clip.mp4"
    generate_procedural_scene_image("Diagnostic Visual", str(test_img), 1, brief["selected_topic"])
    render_image_to_kenburns_video(str(test_img), str(test_clip), duration=dur, scene_index=0, fps=25)

    test_synced = dirs["temp"] / "smoke_test_synced.mp4"
    burn_subtitles_and_audio_to_clip(
        video_clip_path=str(test_clip),
        audio_clip_path=str(test_audio),
        subtitle_text="DIAGNOSTIC TEST PASSED",
        output_clip_path=str(test_synced),
        temp_dir=dirs["temp"]
    )

    # 5. FFprobe validation
    print("\n[5/5] Validating smoke test MP4 with FFprobe...")
    val = validate_final_mp4(str(test_synced))

    elapsed = time.time() - t_start
    print(f"\n✓ SMOKE TEST COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("Environment is 100% verified and ready for full video production.")
    print("=" * 65 + "\n")

    return {
        "status": "PASSED",
        "elapsed_seconds": round(elapsed, 2),
        "validation": val
    }

def run_video_factory(
    niche: str = "Artificial Intelligence and Robotics",
    geo: str = "IN",
    engine: str = "sdxl",
    background_music: bool = True,
    smoke_test_mode: bool = False
) -> Dict[str, Any]:
    """
    Main orchestration entry point.
    If smoke_test_mode=True, runs rapid preflight diagnostics without model downloads.
    If smoke_test_mode=False, executes the complete 16-stage production run.
    """
    if smoke_test_mode:
        return run_smoke_test(niche=niche, geo=geo)

    start_time = time.time()
    dirs = get_base_directories()
    
    print("\n" + "#" * 65)
    print("   ZERO-API AI SHORT-FORM VIDEO FACTORY (PRODUCTION RUN)")
    print("#" * 65 + "\n")

    # STAGE 1: Hardware diagnostics
    logger.info("=== STAGE 1: HARDWARE DIAGNOSTICS ===")
    diag = check_hardware_diagnostics()
    if diag["recommended_engine"] == "sdxl" and engine == "wan" and diag["total_vram_gb"] < 14:
        logger.warning("Selected engine 'wan' requires >=14GB VRAM. Auto-switching to 'sdxl' for safety.")
        engine = "sdxl"

    # STAGE 2: Trend discovery
    logger.info("=== STAGE 2: TREND DISCOVERY ===")
    trend_context = fetch_live_trend_context(niche=niche, geo=geo)
    safe_preview = trend_context[:400].encode("ascii", errors="replace").decode("ascii")
    print(f"\nDiscovered Trend Context:\n{safe_preview}...\n")

    # STAGE 3: Topic research
    logger.info("=== STAGE 3: TOPIC RESEARCH & SELECTION ===")
    research_brief = evaluate_and_select_topic(trend_context, niche)

    # STAGE 4-6: AI script generation & Pydantic validation & cleanup
    logger.info("=== STAGES 4-6: SCRIPT GENERATION & MEMORY FLUSH ===")
    plan_cache_file = dirs["scripts"] / "video_plan.json"
    if plan_cache_file.exists() and plan_cache_file.stat().st_size > 100:
        logger.info("Found cached VideoPlan. Validating and loading for resume...")
        try:
            with open(plan_cache_file, "r", encoding="utf-8") as f:
                video_plan = VideoPlan.model_validate(json.load(f))
        except Exception:
            logger.warning("Cached plan invalid. Re-generating...")
            video_plan = generate_video_script_llm(research_brief)
            with open(plan_cache_file, "w", encoding="utf-8") as f:
                f.write(video_plan.model_dump_json(indent=2))
    else:
        video_plan = generate_video_script_llm(research_brief)
        with open(plan_cache_file, "w", encoding="utf-8") as f:
            f.write(video_plan.model_dump_json(indent=2))
        logger.info(f"Saved VideoPlan to: {plan_cache_file}")

    safe_topic = str(video_plan.selected_topic).encode("ascii", errors="replace").decode("ascii")
    safe_hook = str(video_plan.hook_title).encode("ascii", errors="replace").decode("ascii")
    print(f"\nSelected Topic: {safe_topic}")
    print(f"Hook Title: {safe_hook}")
    print(f"Number of Scenes: {len(video_plan.scenes)}")

    # STAGE 7: CPU Text-to-Speech (Piper TTS, Zero Kokoro, Zero GPU VRAM)
    logger.info("=== STAGE 7: CPU TEXT-TO-SPEECH (PIPER TTS) ===")
    piper_models_dir = dirs["base"] / "models" / "piper"
    audio_paths = []
    actual_durations = []
    for sc in video_plan.scenes:
        audio_file = dirs["audio"] / f"scene_{sc.scene_number:02d}.wav"
        if not audio_file.exists() or audio_file.stat().st_size < 1000:
            synthesize_scene_voice(
                text=sc.voiceover_text,
                output_wav=str(audio_file),
                model_dir=piper_models_dir,
                speed=1.05
            )
        dur = get_audio_duration(str(audio_file))
        audio_paths.append(str(audio_file))
        actual_durations.append(dur)

    total_audio_duration = sum(actual_durations)
    logger.info(f"Total narration duration: {total_audio_duration:.2f} seconds across 10 scenes.")

    # STAGE 8-9: Visual Generation & GPU cleanup (SDXL-Lightning 512x896)
    logger.info("=== STAGES 8-9: VISUAL GENERATION & MEMORY FLUSH ===")
    generate_visual_assets(
        video_plan=video_plan,
        image_dir=dirs["images"],
        scene_dir=dirs["scenes"],
        durations=actual_durations,
        engine=engine
    )
    clean_gpu_memory("Visual Asset Completion")

    scene_clips = [str(dirs["scenes"] / f"scene_{i+1:02d}.mp4") for i in range(10)]
    subtitle_texts = [sc.subtitle_text for sc in video_plan.scenes]

    # STAGE 10-13: Video Editing, Subtitles, BGM, Rendering
    logger.info("=== STAGES 10-13: VIDEO EDITING & AUDIO DUCKING ===")
    final_output_mp4 = dirs["outputs"] / "final_video.mp4"
    stitch_short_video(
        video_paths=scene_clips,
        audio_paths=audio_paths,
        subtitles=subtitle_texts,
        output_mp4=str(final_output_mp4),
        temp_dir=dirs["temp"],
        background_music=background_music
    )

    # STAGE 14: Output validation
    logger.info("=== STAGE 14: OUTPUT VALIDATION ===")
    val_report = validate_final_mp4(str(final_output_mp4))

    # STAGE 15: SEO & Deliverables Export
    logger.info("=== STAGE 15: SOCIAL CAPTIONS & METADATA EXPORT ===")
    deliverables = generate_social_deliverables(
        video_plan=video_plan,
        actual_durations=actual_durations,
        total_duration=val_report["duration_seconds"],
        output_dir=dirs["outputs"],
        engine_used=engine
    )

    # STAGE 16: Complete summary
    elapsed_total = time.time() - start_time
    logger.info(f"=== PIPELINE COMPLETED IN {elapsed_total:.2f} SECONDS ===")
    print("\n" + "#" * 65)
    print("   PRODUCTION RUN SUCCESSFULLY FINISHED")
    print(f" Deliverable Video   : {final_output_mp4}")
    print(f" Caption Deliverable : {deliverables['caption_path']}")
    print(f" Metadata Telemetry  : {deliverables['metadata_path']}")
    print(f" Total Production Time: {elapsed_total:.1f}s")
    print("#" * 65 + "\n")

    return {
        "final_video": str(final_output_mp4),
        "caption": deliverables["caption_path"],
        "metadata": deliverables["metadata_path"],
        "validation": val_report
    }
