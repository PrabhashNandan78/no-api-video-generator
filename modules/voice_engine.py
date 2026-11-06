"""
CPU Text-to-Speech Engine using Piper TTS.
Zero GPU VRAM: 100% CPU inference with offline espeak-ng fallback.
Zero Kokoro dependencies.
"""

import os
import sys
import subprocess
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("VideoFactory.VoiceEngine")

try:
    import soundfile as sf
except ImportError:
    sf = None

# Piper Model Constants
PIPER_VOICE_NAME = "en_US-lessac-medium"
PIPER_MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{PIPER_VOICE_NAME}.onnx"
PIPER_CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{PIPER_VOICE_NAME}.onnx.json"

_CACHED_PIPER_VOICE = None

def ensure_piper_voice_model(model_dir: Path) -> Tuple[str, str]:
    """
    Ensures Piper ONNX voice model and config exist in model_dir.
    Downloads them from Hugging Face if not present.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / f"{PIPER_VOICE_NAME}.onnx"
    json_path = model_dir / f"{PIPER_VOICE_NAME}.onnx.json"

    # Download config JSON first
    if not json_path.exists() or json_path.stat().st_size < 100:
        logger.info(f"Downloading Piper config to: {json_path}")
        urllib.request.urlretrieve(PIPER_CONFIG_URL, str(json_path))

    # Download ONNX model
    if not onnx_path.exists() or onnx_path.stat().st_size < 1000000:
        logger.info(f"Downloading Piper voice model ({PIPER_VOICE_NAME}, ~63 MB)...")
        urllib.request.urlretrieve(PIPER_MODEL_URL, str(onnx_path))

    if not onnx_path.exists() or onnx_path.stat().st_size < 1000000:
        raise FileNotFoundError(f"Piper ONNX model file invalid or incomplete at {onnx_path}")

    return str(onnx_path), str(json_path)

def normalize_and_trim_audio(raw_wav: str, output_wav: str):
    """
    Uses FFmpeg to normalize volume (-14 LUFS) and trim boundary silence.
    Ensures standard 24kHz / 22.05kHz mono PCM WAV.
    """
    filter_chain = (
        "silenceremove=start_periods=1:start_duration=0.04:start_threshold=-45dB:"
        "stop_periods=1:stop_duration=0.08:stop_threshold=-45dB,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_wav,
        "-af", filter_chain,
        "-ac", "1",
        output_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        logger.warning(f"Audio normalization notice ({e}), retaining raw WAV.")
        if raw_wav != output_wav:
            import shutil
            shutil.copyfile(raw_wav, output_wav)

def get_audio_duration(file_path: str) -> float:
    """Calculates exact duration of an audio file in seconds."""
    if sf is not None:
        try:
            data, samplerate = sf.read(file_path)
            return float(len(data)) / float(samplerate)
        except Exception:
            pass
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 5.0

def synthesize_scene_voice(
    text: str,
    output_wav: str,
    model_dir: Optional[Path] = None,
    speed: float = 1.05
) -> str:
    """
    Synthesizes natural narration using Piper TTS (CPU).
    Falls back to espeak-ng if Piper is not installed.
    Never produces fake noise.
    """
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav = str(output_path.with_suffix(".tmp.wav"))

    if model_dir is None:
        model_dir = Path("/content/drive/MyDrive/VideoFactory/models/piper") if Path("/content/drive/MyDrive").exists() else Path("/content/models/piper")

    # 1. Try Piper TTS
    try:
        onnx_file, json_file = ensure_piper_voice_model(model_dir)

        # Attempt A: Piper Python API
        try:
            import wave
            from piper import PiperVoice
            global _CACHED_PIPER_VOICE
            if _CACHED_PIPER_VOICE is None:
                logger.info(f"Loading Piper voice from {onnx_file}...")
                _CACHED_PIPER_VOICE = PiperVoice.load(onnx_file, config_path=json_file)
            
            with wave.open(temp_wav, "wb") as wf:
                _CACHED_PIPER_VOICE.synthesize(text, wf)
                
            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000:
                normalize_and_trim_audio(temp_wav, str(output_path))
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                dur = get_audio_duration(str(output_path))
                logger.info(f"Synthesized scene voice via Piper (API): {dur:.2f}s -> {output_wav}")
                return str(output_path)
        except ImportError:
            # Attempt B: Piper CLI binary or python -m piper
            pass

        # Attempt B: Piper CLI
        piper_bin = "piper"
        cli_cmds = [
            [piper_bin, "--model", onnx_file, "--config", json_file, "--output_file", temp_wav],
            [sys.executable, "-m", "piper", "--model", onnx_file, "--config", json_file, "--output_file", temp_wav]
        ]
        
        for c in cli_cmds:
            try:
                proc = subprocess.run(
                    c,
                    input=text.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000:
                    normalize_and_trim_audio(temp_wav, str(output_path))
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                    dur = get_audio_duration(str(output_path))
                    logger.info(f"Synthesized scene voice via Piper (CLI): {dur:.2f}s -> {output_wav}")
                    return str(output_path)
            except Exception:
                continue

    except Exception as e:
        logger.warning(f"Piper TTS synthesis notice ({e}). Attempting espeak-ng fallback...")

    # 2. Offline Fallback: espeak-ng
    try:
        cmd_espeak = [
            "espeak-ng", "-v", "en-us", "-s", "160", "-w", temp_wav, text
        ]
        res = subprocess.run(cmd_espeak, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 500:
            normalize_and_trim_audio(temp_wav, str(output_path))
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            dur = get_audio_duration(str(output_path))
            logger.info(f"Synthesized voice via espeak-ng: {dur:.2f}s -> {output_wav}")
            return str(output_path)
    except Exception as e_espeak:
        logger.warning(f"espeak-ng notice: {e_espeak}")

    raise RuntimeError(
        f"TTS Error: Both Piper TTS and espeak-ng failed to generate narration for text: '{text[:40]}...'. "
        "Please ensure 'piper-tts' or 'espeak-ng' is installed."
    )
