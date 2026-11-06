"""
Hardware diagnostics and memory management module.
Includes comprehensive preflight checks for Google Colab.
"""

import os
import sys
import gc
import shutil
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger("VideoFactory.Hardware")

def check_hardware_diagnostics() -> Dict[str, Any]:
    """
    Performs comprehensive hardware diagnostic checks:
    - GPU existence, name, total/available VRAM
    - CUDA version, PyTorch version, Python version
    - System RAM (total and available)
    - Available disk storage
    - Recommended execution mode (SDXL-Lightning vs Wan2.1)
    """
    diagnostics = {
        "python_version": sys.version.split()[0],
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "system_ram_gb": 0.0,
        "free_system_ram_gb": 0.0,
        "free_disk_gb": 0.0,
        "recommended_engine": "sdxl",
        "warnings": []
    }

    try:
        import torch
        diagnostics["pytorch_version"] = torch.__version__
        diagnostics["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diagnostics["cuda_version"] = torch.version.cuda
            diagnostics["gpu_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            reserved_bytes = torch.cuda.memory_reserved(0)
            diagnostics["total_vram_gb"] = round(total_bytes / (1024**3), 2)
            diagnostics["free_vram_gb"] = round((total_bytes - reserved_bytes) / (1024**3), 2)
    except Exception as e:
        diagnostics["warnings"].append(f"PyTorch check error: {e}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        diagnostics["system_ram_gb"] = round(mem.total / (1024**3), 2)
        diagnostics["free_system_ram_gb"] = round(mem.available / (1024**3), 2)
    except Exception:
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                info = {line.split(":")[0].strip(): int(line.split(":")[1].split()[0]) for line in lines}
                total_kb = info.get("MemTotal", 0)
                avail_kb = info.get("MemAvailable", 0)
                diagnostics["system_ram_gb"] = round(total_kb / (1024**2), 2)
                diagnostics["free_system_ram_gb"] = round(avail_kb / (1024**2), 2)
            except Exception:
                pass

    try:
        _, _, free_disk = shutil.disk_usage(os.getcwd())
        diagnostics["free_disk_gb"] = round(free_disk / (1024**3), 2)
    except Exception as e:
        diagnostics["warnings"].append(f"Disk check error: {e}")

    if not diagnostics["cuda_available"]:
        diagnostics["recommended_engine"] = "procedural_fallback"
        diagnostics["warnings"].append("No CUDA GPU detected! Running in fallback / CPU mode.")
    elif diagnostics["total_vram_gb"] < 14.0:
        diagnostics["recommended_engine"] = "sdxl"
        diagnostics["warnings"].append(
            f"VRAM ({diagnostics['total_vram_gb']} GB) is under 14 GB. Mode B (SDXL-Lightning) is strongly recommended."
        )
    else:
        # T4 (14.56GB - 15.36GB), L4 (24GB), A100 (40GB)
        diagnostics["recommended_engine"] = "sdxl"

    print("=" * 65)
    print("           HARDWARE & RUNTIME DIAGNOSTICS")
    print("=" * 65)
    print(f" Python Version      : {diagnostics['python_version']}")
    print(f" PyTorch Version     : {diagnostics['pytorch_version']}")
    print(f" CUDA Available      : {diagnostics['cuda_available']}")
    if diagnostics["cuda_available"]:
        print(f" GPU Model           : {diagnostics['gpu_name']}")
        print(f" Total VRAM          : {diagnostics['total_vram_gb']} GB")
        print(f" Available VRAM      : {diagnostics['free_vram_gb']} GB")
        print(f" CUDA Version        : {diagnostics['cuda_version']}")
    print(f" Host RAM (Total)    : {diagnostics['system_ram_gb']} GB")
    print(f" Host RAM (Free)     : {diagnostics['free_system_ram_gb']} GB")
    print(f" Free Disk Space     : {diagnostics['free_disk_gb']} GB")
    print(f" Recommended Engine  : {diagnostics['recommended_engine'].upper()}")
    if diagnostics["warnings"]:
        print("\n Diagnostics Notices / Warnings:")
        for w in diagnostics["warnings"]:
            print(f"  [!] {w}")
    print("=" * 65)

    return diagnostics

def run_preflight_system_checks() -> Dict[str, bool]:
    """
    Lightweight preflight check validating CLI tools and Python packages
    before loading heavy neural models.
    """
    results = {}
    
    # FFmpeg & FFprobe
    for tool in ["ffmpeg", "ffprobe"]:
        try:
            res = subprocess.run([tool, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results[tool] = (res.returncode == 0)
        except Exception:
            results[tool] = False

    # Python packages
    for pkg in ["torch", "transformers", "diffusers", "pydantic", "soundfile"]:
        try:
            __import__(pkg)
            results[f"pkg_{pkg}"] = True
        except ImportError:
            results[f"pkg_{pkg}"] = False

    # DDGS package
    try:
        from ddgs import DDGS
        results["pkg_ddgs"] = True
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            results["pkg_ddgs"] = True
        except ImportError:
            results["pkg_ddgs"] = False

    # Piper TTS / espeak-ng check
    results["piper_cli"] = shutil.which("piper") is not None
    try:
        import piper
        results["piper_python"] = True
    except ImportError:
        results["piper_python"] = False
        
    results["espeak_ng"] = shutil.which("espeak-ng") is not None
    results["tts_ready"] = results["piper_cli"] or results["piper_python"] or results["espeak_ng"]

    return results

def get_gpu_memory_info() -> str:
    """Returns human-readable current VRAM allocation and reservation."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA unavailable (CPU mode)"
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB"
    except Exception as e:
        return f"VRAM info error: {e}"

def clean_gpu_memory(stage_name: str = "Stage"):
    """
    Performs complete memory cleanup:
    1. Python garbage collection
    2. PyTorch CUDA cache flush
    3. PyTorch IPC collect
    """
    mem_before = get_gpu_memory_info()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logger.warning(f"CUDA cache flush notice: {e}")

    mem_after = get_gpu_memory_info()
    logger.info(f"[{stage_name}] Memory cleared. Before: [{mem_before}] -> After: [{mem_after}]")
    return mem_after
