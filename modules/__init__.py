"""
Zero-API AI Video Factory Package.
"""

from .config import get_base_directories, logger
from .hardware import check_hardware_diagnostics, clean_gpu_memory, run_preflight_system_checks
from .trend_engine import fetch_live_trend_context
from .research_engine import evaluate_and_select_topic
from .script_engine import generate_video_script_llm, VideoPlan, Scene
from .voice_engine import synthesize_scene_voice, get_audio_duration, ensure_piper_voice_model
from .visual_engine import generate_visual_assets
from .editing_engine import stitch_short_video
from .seo_engine import generate_social_deliverables
from .pipeline import run_video_factory, validate_final_mp4, run_smoke_test

__all__ = [
    "get_base_directories",
    "logger",
    "check_hardware_diagnostics",
    "clean_gpu_memory",
    "run_preflight_system_checks",
    "fetch_live_trend_context",
    "evaluate_and_select_topic",
    "generate_video_script_llm",
    "VideoPlan",
    "Scene",
    "synthesize_scene_voice",
    "get_audio_duration",
    "ensure_piper_voice_model",
    "generate_visual_assets",
    "stitch_short_video",
    "generate_social_deliverables",
    "run_video_factory",
    "validate_final_mp4",
    "run_smoke_test",
]
