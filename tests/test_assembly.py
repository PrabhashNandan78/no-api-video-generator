import os
import subprocess
import json

def test_full_pipeline_assembly():
    test_dir = r"C:\Users\prabh\.gemini\antigravity\scratch\video_factory\sample_output"
    
    # 1. Create a dummy background music track (10 seconds ambient tone)
    bgm_path = os.path.join(test_dir, "ambient_bgm.wav")
    cmd_bgm = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anoisesrc=d=15:c=brown:r=44100:a=0.03",
        "-af", "lowpass=f=400,afade=t=in:ss=0:d=1,afade=t=out:st=13:d=2",
        "-c:a", "pcm_s16le", bgm_path
    ]
    subprocess.run(cmd_bgm, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Created ambient BGM: {bgm_path}")
    
    # 2. Test concatenation with concat demuxer
    clip1 = os.path.join(test_dir, "test_clip_01.mp4")
    concat_list_file = os.path.join(test_dir, "concat_list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as f:
        # Repeat clip 1 twice to test joining two 5-second clips into a 10s video
        f.write(f"file '{clip1.replace(chr(92), '/')}'\n")
        f.write(f"file '{clip1.replace(chr(92), '/')}'\n")
    
    stitched_raw = os.path.join(test_dir, "stitched_raw.mp4")
    cmd_concat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_file,
        "-c", "copy", stitched_raw
    ]
    subprocess.run(cmd_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Stitched clips: {stitched_raw} (size: {os.path.getsize(stitched_raw)})")
    
    # 3. Test background music ducking:
    # Narration is on [0:a], BGM is on [1:a].
    # BGM volume reduced to 0.15 (-16dB), narration volume 1.0, mixed together.
    final_output = os.path.join(test_dir, "final_video.mp4")
    filter_complex = (
        "[1:a]volume=0.15,afade=t=in:ss=0:d=1,afade=t=out:st=8:d=2[bg];"
        "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    cmd_mix = [
        "ffmpeg", "-y",
        "-i", stitched_raw,
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        final_output
    ]
    res_mix = subprocess.run(cmd_mix, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_mix.returncode != 0:
        print(f"Mix error: {res_mix.stderr}")
    else:
        print(f"Final output produced: {final_output} (size: {os.path.getsize(final_output)} bytes)")
        
    # 4. Validate output with ffprobe
    cmd_probe = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", final_output
    ]
    probe_res = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = json.loads(probe_res.stdout)
    v_stream = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    a_stream = next((s for s in metadata["streams"] if s["codec_type"] == "audio"), None)
    
    print("\n--- FFPROBE VALIDATION ---")
    print(f"Container format: {metadata['format']['format_name']}")
    print(f"Duration: {float(metadata['format']['duration']):.2f} seconds")
    if v_stream:
        print(f"Video: {v_stream['codec_name']} {v_stream['width']}x{v_stream['height']} @ {eval(v_stream['r_frame_rate']):.1f} fps")
    if a_stream:
        print(f"Audio: {a_stream['codec_name']} {a_stream['sample_rate']}Hz {a_stream['channels']} channels")

if __name__ == "__main__":
    test_full_pipeline_assembly()

