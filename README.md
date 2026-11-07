# 🎬 ZERO-API AI SHORT-FORM VIDEO FACTORY (`local_video_factory_colab.ipynb`)

> **Production-Grade, 100% Free & Zero-API Automated 9:16 Vertical Video Production Pipeline**  
> Designed and optimized for **Google Colab Free Tier (NVIDIA Tesla T4 ~14.56 GB VRAM, Python 3.13.16, PyTorch 2.11.0+cu130)**.

---

## 🌟 Executive Summary

The **Zero-API AI Short-Form Video Factory** is a fully automated video production system that autonomously discovers trending topics, verifies factual context from live reports, generates viral structured scripts, synthesizes natural voiceover narration on CPU via **Piper TTS**, renders cinematic visual assets using **SDXL-Lightning** (512×896 9:16), burns animated subtitles, mixes ducked royalty-free background music, and exports platform-ready 9:16 MP4 videos directly to Google Drive.

### The Non-Negotiable Zero-Cost Guarantee:
- ❌ **No Paid APIs:** Zero OpenAI, Zero ElevenLabs, Zero Runway, Zero Pika, Zero Claude, Zero Gemini.
- ❌ **No Credit Cards / Cloud Billing:** Operates 100% within free open-source ecosystems.
- ✅ **100% Open Weights & Public Protocols:** Hugging Face open weights, public RSS feeds, public news scraping via `ddgs`, and FFmpeg.
- ✅ **Python 3.13 & Colab Free-Tier Optimized:** Zero Kokoro dependencies. Utilizes **Piper TTS** for fast, high-quality, CPU-based speech synthesis.
- ✅ **Sequential GPU Model Scheduling:** Heavy GPU models never share VRAM simultaneously, preventing CUDA OOM crashes on NVIDIA T4 hardware.

---

## 🏛️ System Architecture & Sequential Execution

To operate reliably within the 14.56 GB VRAM and ~12.67 GB Host RAM limitations of Colab Free, the system enforces **single-occupancy sequential GPU model scheduling**:

```
[STAGE 1: HARDWARE DIAGNOSTICS & PREFLIGHT]
      │  Checks Python 3.13, PyTorch, CUDA, VRAM, RAM, FFmpeg, DDGS, Piper
      ▼
[STAGE 2: TREND DISCOVERY ENGINE]
      │  Scrapes Google Trends RSS & DuckDuckGo News (ddgs) with full attribution
      ▼
[STAGE 3: SOURCE-GROUNDED RESEARCH ENGINE]
      │  Extracts documented evidence without fabricating unverified statistics
      ▼
[STAGE 4: SCRIPT ENGINE (GPU)] ───► Load Qwen2.5-7B-Instruct (4-bit NF4)
      │                             Generate 10 scenes, Pydantic validation (~50s total)
      ▼
[STAGE 5: GPU MEMORY PURGE] ──────► del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
      │                             [VRAM Recovered: ~100% available]
      ▼
[STAGE 6: VOICE ENGINE (CPU)] ────► Piper TTS (en_US-lessac-medium) + espeak-ng fallback
      │                             Volume normalization (-14 LUFS) & exact duration tracking
      ▼
[STAGE 7: VISUAL ENGINE (GPU)] ───► Load SDXL-Lightning 4-step UNet (512×896 9:16)
      │                             Generate 10 scenes; render 10-motion dynamic Ken Burns FFmpeg
      ▼
[STAGE 8: GPU MEMORY PURGE] ──────► del pipe; gc.collect(); torch.cuda.empty_cache()
      │                             [VRAM Recovered: ~100% available]
      ▼
[STAGE 9: EDITING & SUBTITLES] ───► Burn animated ASS subtitles (76% height, DejaVuSans-Bold)
      │                             Lossless concatenation with FFmpeg concat demuxer
      ▼
[STAGE 10: AUDIO DUCKING & SEO] ──► Mix background music (-18 dB under voiceover narration)
      │                             Export final_video.mp4, caption.txt, video_metadata.json
      ▼
[STAGE 11: GOOGLE DRIVE PERSIST] ─► Save directly to /content/drive/MyDrive/VideoFactory/
```

---

## 📊 Hardware & VRAM Budget (NVIDIA Tesla T4 14.56 GB)

| Pipeline Stage | Model / Component | Hardware | Peak VRAM | Host RAM | Typical Runtime |
|---|---|---|---|---|---|
| **1. Diagnostics & Preflight** | System checks / DDGS | CPU / Network | **0.0 GB** | ~0.4 GB | ~2–3 sec |
| **2. Trend Discovery** | RSS + DDG News (`ddgs`) | CPU / Network | **0.0 GB** | ~0.5 GB | ~3–5 sec |
| **3. AI Script Generation** | `Qwen2.5-7B-Instruct` (NF4) | GPU | **~5.1 GB** | ~2.5 GB | ~8–12 sec |
| **4. Memory Cleanup** | Garbage Collector + Cache | CPU / GPU | **0.0 GB** | ~1.8 GB | <1 sec |
| **5. Voiceover Synthesis** | `Piper TTS` (en_US-lessac) | CPU | **0.0 GB** | ~1.4 GB | ~4–6 sec (10 scenes) |
| **6. Visual Asset Gen** | `SDXL-Lightning` 4-step (512x896)| GPU | **~5.5 GB** | ~2.8 GB | ~30–45 sec (10 scenes) |
| **7. Memory Cleanup** | Garbage Collector + Cache | CPU / GPU | **0.0 GB** | ~1.9 GB | <1 sec |
| **8. Ken Burns & Subtitles** | FFmpeg (libx264/libass) | CPU | **0.0 GB** | ~1.5 GB | ~15–20 sec |
| **9. Concat, Ducking, SEO** | FFmpeg / Python | CPU | **0.0 GB** | ~1.2 GB | ~5–8 sec |
| **TOTAL (Production Run)** | **Complete 50s Short Video** | **T4 GPU** | **Max 5.5 GB** | **Max 2.8 GB** | **~70–90 seconds** |

---

## 📁 Workspace Directory Structure

All persistent outputs are exported directly into Google Drive (or local runtime storage if unmounted):

```text
VideoFactory/
├── outputs/
│   ├── final_video.mp4       # Final 720x1280 9:16 vertical short-form video (25 FPS)
│   ├── caption.txt           # Multi-platform captions (Shorts, Reels, TikTok)
│   └── video_metadata.json   # Telemetry: timestamps, prompts, models, durations
├── scenes/
│   ├── scene_01.mp4          # 5-second dynamic Ken Burns video clip
│   ├── scene_02.mp4
│   └── ...                   # Up to scene_10.mp4
├── audio/
│   ├── scene_01.wav          # 22.05kHz mono normalized voiceover narration (Piper)
│   └── ...                   # Up to scene_10.wav
├── images/
│   ├── scene_01.png          # 512x896 9:16 vertical source image (SDXL-Lightning)
│   └── ...                   # Up to scene_10.png
├── models/
│   └── piper/
│       ├── en_US-lessac-medium.onnx      # Piper voice model (~63 MB)
│       └── en_US-lessac-medium.onnx.json # Piper voice config
├── scripts/
│   └── video_plan.json       # Pydantic-validated 10-scene structured plan
└── logs/
    └── factory.log           # Chronological execution & memory telemetry log
```

---

## 🚀 How to Run on Google Colab

1. **Upload Notebook:**
   - Open [Google Colab](https://colab.research.google.com/).
   - Click **File > Upload notebook** and select `local_video_factory_colab.ipynb`.

2. **Select GPU Runtime:**
   - Go to **Runtime > Change runtime type**.
   - Select **T4 GPU** (Hardware Accelerator).
   - Click **Save**.

3. **Step 1: Rapid Preflight Smoke-Test (Default in Cell 12):**
   - By default, `SMOKE_TEST_MODE = True` in Cell 12.
   - Run the cells sequentially.
   - The smoke test downloads the ~63 MB Piper voice, runs a live trend test, renders a sample Ken Burns clip, burns subtitles, and validates with `ffprobe` in **~10 seconds without downloading heavy GPU models**.

4. **Step 2: Full Production Run:**
   - In Cell 12, set `SMOKE_TEST_MODE = False`.
   - Run Cell 12 to generate the complete 50-second, 10-scene production video.
   - The final video streams in the embedded HTML5 player and saves to `/content/drive/MyDrive/VideoFactory/outputs/final_video.mp4`.

---

## 🛠️ Cell-by-Cell Function Reference

| Cell | Purpose | Key Technologies / Operations |
|---|---|---|
| **Cell 1** | Drive Mount & Dirs | `pathlib.Path`, `/content/drive/MyDrive/VideoFactory/`, logging |
| **Cell 2** | Dependencies | `piper-tts`, `transformers`, `diffusers`, `bitsandbytes`, `ddgs`, `ffmpeg` |
| **Cell 3** | Diagnostics & Preflight | VRAM query, `clean_gpu_memory`, `run_preflight_system_checks` |
| **Cell 4** | Trend Discovery | Google Trends RSS, DuckDuckGo (`ddgs`), deduplication, URL preservation |
| **Cell 5** | Topic Research | Evidence-grounded research brief, verified sources (no fabricated metrics) |
| **Cell 6** | Script Generation | `Qwen2.5-7B-Instruct` 4-bit NF4, Pydantic validation, immediate VRAM purge |
| **Cell 7** | Voiceover (CPU) | `Piper TTS` (en_US-lessac-medium), FFmpeg normalization, duration tracking |
| **Cell 8** | Visuals (GPU) | `SDXL-Lightning` 4-step (512×896 9:16), 10 Ken Burns motions, VRAM purge |
| **Cell 9** | Subtitles & Video | ASS subtitles (DejaVuSans-Bold, 76% height), FFmpeg concat demuxer |
| **Cell 10**| Audio Mixing | Procedural ambient music, audio ducking (-18 dB under narration) |
| **Cell 11**| SEO Captions & Meta | `caption.txt`, `video_metadata.json` documenting Piper TTS & Qwen 7B |
| **Cell 12**| Master Orchestration | `run_smoke_test()`, `run_video_factory()`, `ffprobe` validation, HTML5 player |

---

## 🎞️ Demo & Outputs

One small sample render is kept in this repository at `sample_output/final_video.mp4` so visitors can see what the pipeline produces. All other generated media — the `temp/` working files, intermediate scene clips/audio, and `VideoFactory/outputs/` — is **not** tracked in git (see `.gitignore`); regenerate it by running the pipeline. Model weights and checkpoints are likewise untracked and are downloaded at runtime.
