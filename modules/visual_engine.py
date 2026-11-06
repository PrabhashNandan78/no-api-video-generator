"""
Visual Generation Engine.
Supports Mode B (SDXL-Lightning 4-step + Ken Burns FFmpeg) at T4-friendly 512x896 (9:16)
and optional Mode A (Wan2.1-T2V-1.3B).
Fully resumable and memory safe.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw

from .hardware import clean_gpu_memory

logger = logging.getLogger("VideoFactory.VisualEngine")

# 10 Distinct cinematic motion profiles for short-form engagement (720x1280 output)
KEN_BURNS_MOTIONS = [
    # Scene 1: Smooth center zoom in
    "zoompan=z='min(zoom+0.0018,1.25)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    # Scene 2: Top to bottom pan / tilt
    "zoompan=z='1.15':d={frames}:x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(on/{frames})':s=720x1280:fps={fps}",
    # Scene 3: Slow zoom out reveal
    "zoompan=z='if(lte(zoom,1.0),1.25,max(1.001,zoom-0.0018))':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    # Scene 4: Smooth pan left to right
    "zoompan=z='1.15':d={frames}:x='(iw-iw/zoom)*(on/{frames})':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    # Scene 5: Diagonal push (bottom-left to top-right)
    "zoompan=z='min(zoom+0.0015,1.20)':d={frames}:x='(iw-iw/zoom)*(1-on/{frames})':y='(ih-ih/zoom)*(on/{frames})':s=720x1280:fps={fps}",
    # Scene 6: Bottom to top pan / tilt up
    "zoompan=z='1.15':d={frames}:x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{frames})':s=720x1280:fps={fps}",
    # Scene 7: Smooth pan right to left
    "zoompan=z='1.15':d={frames}:x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    # Scene 8: Dynamic zoom in with slight offset
    "zoompan=z='min(zoom+0.002,1.28)':d={frames}:x='iw/2-(iw/zoom/2)+10':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    # Scene 9: Center zoom out
    "zoompan=z='if(lte(zoom,1.0),1.22,max(1.001,zoom-0.0016))':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    # Scene 10: Slow majestic center zoom in
    "zoompan=z='min(zoom+0.0015,1.20)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}"
]

def render_image_to_kenburns_video(
    image_path: str,
    output_mp4: str,
    duration: float = 5.0,
    scene_index: int = 0,
    fps: int = 25
) -> str:
    """
    Transforms a static 9:16 vertical image (512x896 or 768x1344) into a smooth 720x1280 Ken Burns MP4 clip.
    Handles file paths with spaces properly.
    """
    frames = max(25, int(duration * fps))
    motion_template = KEN_BURNS_MOTIONS[scene_index % len(KEN_BURNS_MOTIONS)]
    vf = motion_template.format(frames=frames, fps=fps)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(output_mp4)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"FFmpeg Ken Burns failed: {res.stderr}")
        raise RuntimeError(f"FFmpeg Ken Burns failed: {res.stderr}")
    return output_mp4

def generate_procedural_scene_image(
    prompt: str,
    output_image_path: str,
    scene_number: int,
    topic: str
) -> str:
    """
    Procedural visual generator for environments without GPU.
    Generates vibrant 512x896 (9:16) abstract technical artwork.
    """
    width, height = 512, 896
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    color_schemes = [
        ((15, 23, 42), (59, 130, 246)),   # Slate to electric blue
        ((10, 10, 20), (168, 85, 247)),   # Obsidian to neon purple
        ((6, 78, 59), (16, 185, 129)),    # Emerald matrix
        ((30, 27, 75), (236, 72, 153)),   # Deep indigo to magenta
        ((24, 24, 27), (245, 158, 11)),   # Amber and tungsten
    ]
    c_start, c_end = color_schemes[scene_number % len(color_schemes)]

    for y in range(height):
        ratio = y / height
        r = int(c_start[0] * (1 - ratio) + c_end[0] * ratio)
        g = int(c_start[1] * (1 - ratio) + c_end[1] * ratio)
        b = int(c_start[2] * (1 - ratio) + c_end[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    for i in range(10):
        offset = i * 70
        draw.ellipse([50 + i*12, 140 + offset, width - 50 - i*12, 280 + offset], outline=(255, 255, 255, 30), width=2)

    img.save(output_image_path, quality=95)
    return output_image_path

def generate_sdxl_lightning_assets(
    scenes: List[Any],
    image_dir: Path,
    scene_dir: Path,
    durations: List[float],
    topic: str,
    img_width: int = 512,
    img_height: int = 896
):
    """
    Generates 9:16 vertical images using SDXL-Lightning 4-step pipeline and converts them
    to dynamic Ken Burns video clips. Uses T4-friendly 512x896 resolution.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    logger.info(f"Initializing SDXL-Lightning 4-step pipeline on GPU (Resolution: {img_width}x{img_height})...")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    try:
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        unet_weights = load_file(hf_hub_download(repo, ckpt), device="cuda")
        unet.load_state_dict(unet_weights)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.enable_attention_slicing()
    except Exception as e:
        logger.warning(f"Direct UNet load notice ({e}). Using LoRA configuration...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        pipe.load_lora_weights(repo, weight_name="sdxl_lightning_4step_lora.safetensors")
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Generate each scene
    for i, sc in enumerate(scenes):
        scene_num = sc.scene_number
        img_path = image_dir / f"scene_{scene_num:02d}.png"
        clip_path = scene_dir / f"scene_{scene_num:02d}.mp4"
        duration = durations[i] if i < len(durations) else 5.0

        # Validate cached clip
        if clip_path.exists() and clip_path.stat().st_size > 50000:
            logger.info(f"Scene {scene_num} video clip already exists. Skipping.")
            continue

        if not img_path.exists() or img_path.stat().st_size < 10000:
            logger.info(f"Generating SDXL-Lightning image for Scene {scene_num}...")
            prompt = f"{sc.visual_prompt}, 8k resolution, cinematic lighting, photorealistic, 9:16 vertical ratio, masterpiece"
            negative_prompt = "text, watermark, low quality, blurry, deformed, amateur, ugly, oversaturated"
            
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=img_width,
                height=img_height
            ).images[0]
            image.save(str(img_path))
            logger.info(f"Saved image: {img_path}")

        # Render Ken Burns video clip
        logger.info(f"Rendering Ken Burns motion clip for Scene {scene_num} ({duration:.2f}s)...")
        render_image_to_kenburns_video(
            image_path=str(img_path),
            output_mp4=str(clip_path),
            duration=duration,
            scene_index=i,
            fps=25
        )

    del pipe
    if 'unet' in locals():
        del unet
    clean_gpu_memory("SDXL Visual Engine")

def generate_wan_video_assets(
    scenes: List[Any],
    scene_dir: Path,
    durations: List[float]
):
    """
    Generates video clips using Wan2.1-T2V-1.3B-Diffusers.
    """
    import torch
    from diffusers import WanPipeline

    logger.info("Initializing Wan2.1-T2V-1.3B-Diffusers pipeline...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    for i, sc in enumerate(scenes):
        scene_num = sc.scene_number
        clip_path = scene_dir / f"scene_{scene_num:02d}.mp4"
        if clip_path.exists() and clip_path.stat().st_size > 50000:
            logger.info(f"Scene {scene_num} already exists. Skipping.")
            continue

        logger.info(f"Generating Wan2.1 video for Scene {scene_num}...")
        prompt = f"{sc.visual_prompt}, 9:16 vertical orientation, 4k ultra detailed cinematic video"
        output = pipe(
            prompt=prompt,
            height=832,
            width=480,
            num_frames=81,
            guidance_scale=5.0
        ).frames[0]
        
        from diffusers.utils import export_to_video
        export_to_video(output, str(clip_path), fps=16)

    del pipe
    clean_gpu_memory("Wan Visual Engine")

def generate_visual_assets(
    video_plan: Any,
    image_dir: Path,
    scene_dir: Path,
    durations: List[float],
    engine: str = "sdxl"
):
    """
    Master visual generation dispatcher.
    """
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        pass

    if not has_gpu:
        logger.warning("No CUDA GPU detected. Generating procedural cinematic visuals.")
        for i, sc in enumerate(video_plan.scenes):
            scene_num = sc.scene_number
            img_path = image_dir / f"scene_{scene_num:02d}.png"
            clip_path = scene_dir / f"scene_{scene_num:02d}.mp4"
            duration = durations[i] if i < len(durations) else 5.0
            
            if not clip_path.exists() or clip_path.stat().st_size < 10000:
                generate_procedural_scene_image(
                    prompt=sc.visual_prompt,
                    output_image_path=str(img_path),
                    scene_number=scene_num,
                    topic=video_plan.selected_topic
                )
                render_image_to_kenburns_video(
                    image_path=str(img_path),
                    output_mp4=str(clip_path),
                    duration=duration,
                    scene_index=i,
                    fps=25
                )
        return

    if engine.lower() == "wan":
        try:
            generate_wan_video_assets(video_plan.scenes, scene_dir, durations)
            return
        except Exception as e:
            logger.error(f"Wan2.1 generation failed or OOM ({e}). Falling back to SDXL-Lightning.")
            clean_gpu_memory("Wan Recovery")

    generate_sdxl_lightning_assets(
        video_plan.scenes,
        image_dir,
        scene_dir,
        durations,
        video_plan.selected_topic,
        img_width=512,
        img_height=896
    )
