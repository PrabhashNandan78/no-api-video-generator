import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

def test_ffmpeg_kenburns_and_subtitles():
    test_dir = r"C:\Users\prabh\.gemini\antigravity\scratch\video_factory\sample_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Generate a test vertical 9:16 image (768 x 1344)
    img_path = os.path.join(test_dir, "test_scene_01.png")
    img = Image.new("RGB", (768, 1344), color=(30, 45, 80))
    draw = ImageDraw.Draw(img)
    # Draw some shapes/gradients to visualize motion
    for i in range(10):
        draw.rectangle([60 + i*10, 200 + i*60, 700 - i*10, 300 + i*60], outline=(100 + i*15, 200, 255 - i*10), width=4)
    draw.text((200, 700), "AI VIDEO FACTORY", fill=(255, 255, 255))
    draw.text((240, 760), "SCENE 1: HOOK", fill=(200, 240, 255))
    img.save(img_path)
    print(f"Created sample image: {img_path}")
    
    # 2. Generate a 5-second silent WAV for audio synchronization test
    wav_path = os.path.join(test_dir, "test_audio_01.wav")
    # Using ffmpeg to generate 5s of tone/silence
    cmd_audio = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anoisesrc=d=5:c=pink:r=24000:a=0.01",
        "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", wav_path
    ]
    subprocess.run(cmd_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Created test audio: {wav_path}")
    
    # 3. Create ASS subtitle file
    ass_path = os.path.join(test_dir, "test_subtitles.ass")
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,30,30,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.50,0:00:04.50,Default,,0,0,0,,{\\b1}THE FUTURE OF ARTIFICIAL INTELLIGENCE{\\b0}
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    print(f"Created ASS subtitle file: {ass_path}")

    # 4. Generate video clip with Ken Burns filter + subtitle
    out_clip = os.path.join(test_dir, "test_clip_01.mp4")
    # Ken burns: zoom in slowly over 125 frames (5s @ 25fps), scale to 720x1280
    # Note: escape colon and backslashes for ffmpeg filter graph
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    
    vf = f"zoompan=z='min(zoom+0.0015,1.25)':d=125:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps=25,subtitles='{escaped_ass}'"
    cmd_render = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img_path,
        "-i", wav_path,
        "-vf", vf,
        "-t", "5",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        out_clip
    ]
    res = subprocess.run(cmd_render, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"FFmpeg error: {res.stderr}")
    else:
        print(f"Rendered video clip successfully: {out_clip} (size: {os.path.getsize(out_clip)} bytes)")

if __name__ == "__main__":
    test_ffmpeg_kenburns_and_subtitles()

