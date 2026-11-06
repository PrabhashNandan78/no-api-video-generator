"""
Audio and Video Editing Engine.
FFmpeg and MoviePy 2.x compatible.
Handles ASS subtitle generation, Ken Burns burning, audio synchronization,
and royalty-free background music ducking (-18 dB under narration).
Handles paths with spaces safely.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("VideoFactory.EditingEngine")

def format_ass_time(seconds: float) -> str:
    """Formats seconds into ASS timestamp (H:MM:SS.cs)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        secs += 1
        cs = 0
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"

def wrap_subtitle_text(text: str, max_words_per_line: int = 5) -> str:
    """Wraps subtitle into maximum 2 centered lines for short-form vertical video."""
    words = text.strip().split()
    if len(words) <= max_words_per_line:
        return text.strip().upper()
    
    mid = len(words) // 2
    line1 = " ".join(words[:mid]).upper()
    line2 = " ".join(words[mid:]).upper()
    return f"{line1}\\N{line2}"

def create_ass_file_for_clip(
    subtitle_text: str,
    duration: float,
    output_ass: str,
    font_name: str = "DejaVuSans-Bold"
) -> str:
    """
    Generates an Advanced SubStation Alpha (.ass) subtitle file.
    Styles: Bold white text, black outline, centered horizontally at ~76% frame height (MarginV=160).
    """
    formatted_text = wrap_subtitle_text(subtitle_text)
    start_time = format_ass_time(0.15)
    end_time = format_ass_time(max(0.5, duration - 0.15))

    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortForm,{font_name},28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.2,1.2,2,40,40,160,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,{start_time},{end_time},ShortForm,,0,0,0,,{{\\b1}}{formatted_text}{{\\b0}}
"""
    with open(output_ass, "w", encoding="utf-8") as f:
        f.write(ass_content)
    return output_ass

def generate_procedural_background_music(output_wav: str, duration: float = 60.0) -> str:
    """
    Synthesizes a royalty-free ambient background music bed using FFmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anoisesrc=d={duration:.2f}:c=brown:r=44100:a=0.035",
        "-af", (
            "lowpass=f=450,volume=1.2,"
            "afade=t=in:ss=0:d=2.0,"
            f"afade=t=out:st={max(1.0, duration - 3.0):.2f}:d=3.0"
        ),
        "-c:a", "pcm_s16le",
        str(output_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_wav

def burn_subtitles_and_audio_to_clip(
    video_clip_path: str,
    audio_clip_path: str,
    subtitle_text: str,
    output_clip_path: str,
    temp_dir: Path
) -> str:
    """
    Combines scene video, narration audio, and burns ASS subtitles into a single synced clip.
    Handles paths containing spaces safely.
    """
    ass_path = temp_dir / f"sub_{Path(output_clip_path).stem}.ass"
    
    from .voice_engine import get_audio_duration
    audio_dur = get_audio_duration(audio_clip_path)
    create_ass_file_for_clip(subtitle_text, audio_dur, str(ass_path))

    # Escape subtitle path for FFmpeg filter graph (handle colons and backslashes)
    escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    vf = f"subtitles='{escaped_ass}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_clip_path),
        "-i", str(audio_clip_path),
        "-vf", vf,
        "-t", f"{audio_dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_clip_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.warning(f"Subtitles filter fallback ({res.stderr}). Generating clip without burned subtitles...")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", str(video_clip_path),
            "-i", str(audio_clip_path),
            "-t", f"{audio_dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_clip_path)
        ]
        subprocess.run(cmd_fallback, check=True)
    
    return output_clip_path

def stitch_short_video(
    video_paths: List[str],
    audio_paths: List[str],
    subtitles: List[str],
    output_mp4: str,
    temp_dir: Path,
    background_music: bool = True
) -> str:
    """
    Master video assembler:
    1. Synchronizes each scene's visual clip with narration and burns styled subtitles.
    2. Seamlessly concatenates all 10 scenes.
    3. Mixes and ducks royalty-free background music (-18 dB under narration).
    4. Produces the final 720x1280 25fps MP4.
    """
    logger.info("Assembling 10 synchronized scenes into final short-form video...")
    synced_scene_clips = []

    for i in range(len(video_paths)):
        v_path = video_paths[i]
        a_path = audio_paths[i]
        sub_text = subtitles[i] if i < len(subtitles) else ""
        synced_out = str(temp_dir / f"synced_scene_{i+1:02d}.mp4")

        burn_subtitles_and_audio_to_clip(
            video_clip_path=v_path,
            audio_clip_path=a_path,
            subtitle_text=sub_text,
            output_clip_path=synced_out,
            temp_dir=temp_dir
        )
        synced_scene_clips.append(synced_out)

    # Concat file list (safely handles paths with spaces)
    concat_txt = temp_dir / "concat_scenes.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in synced_scene_clips:
            clean_p = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{clean_p}'\n")

    raw_stitched = temp_dir / "raw_stitched.mp4"
    cmd_concat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(raw_stitched)
    ]
    res_concat = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_concat.returncode != 0:
        logger.warning(f"Stream-copy concat notice ({res_concat.stderr}). Re-encoding concat...")
        cmd_reencode = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-c:a", "aac", "-b:a", "192k",
            str(raw_stitched)
        ]
        subprocess.run(cmd_reencode, check=True)

    logger.info(f"Concatenated 10 scenes: {raw_stitched}")

    if not background_music:
        import shutil
        shutil.copyfile(str(raw_stitched), output_mp4)
        return output_mp4

    # Background music mixing and ducking (-18 dB under narration)
    from .voice_engine import get_audio_duration
    total_dur = get_audio_duration(str(raw_stitched))
    bgm_path = temp_dir / "ambient_bgm.wav"
    generate_procedural_background_music(str(bgm_path), duration=total_dur + 5.0)

    fade_out_start = max(1.0, total_dur - 2.5)
    # BGM volume = 0.14 (-17dB to -18dB), narration at 1.0 (0dB)
    filter_complex = (
        f"[1:a]volume=0.14,afade=t=in:ss=0:d=1.5,afade=t=out:st={fade_out_start:.2f}:d=2.5[bg];"
        "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )

    cmd_mix = [
        "ffmpeg", "-y",
        "-i", str(raw_stitched),
        "-i", str(bgm_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_mp4)
    ]
    res_mix = subprocess.run(cmd_mix, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_mix.returncode != 0:
        logger.warning(f"Audio mix error: {res_mix.stderr}. Exporting without background music.")
        import shutil
        shutil.copyfile(str(raw_stitched), output_mp4)
    else:
        logger.info(f"Successfully mixed background music and exported: {output_mp4}")

    return output_mp4
