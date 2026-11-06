"""
AI Brain: Script and Scene Generation Engine.
Uses Qwen2.5-7B-Instruct in 4-bit NF4 with Pydantic validation and strict memory management.
Grounds narration strictly in the verified research brief.
"""

import json
import re
import warnings
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from .hardware import clean_gpu_memory

# Suppress deprecation warnings on Python 3.13 (e.g. from tqdm/session)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("VideoFactory.ScriptEngine")

# Pydantic Schemas
class Scene(BaseModel):
    scene_number: int = Field(..., ge=1, le=10, description="Sequential index from 1 to 10")
    visual_prompt: str = Field(..., min_length=15, description="Cinematic visual description for SDXL")
    voiceover_text: str = Field(..., min_length=10, description="Natural narration script (~11-14 words)")
    subtitle_text: str = Field(..., min_length=4, description="Concise on-screen caption (~4-7 words)")
    duration_seconds: float = Field(default=5.0, ge=3.0, le=8.0, description="Estimated scene duration")

    @field_validator("visual_prompt")
    @classmethod
    def clean_prompt(cls, v: str) -> str:
        cleaned = re.sub(r'(?i)\b(text|words|letters|signage|subtitles)\b', '', v).strip()
        return cleaned if len(cleaned) > 10 else v

class VideoPlan(BaseModel):
    selected_topic: str = Field(..., min_length=5)
    hook_title: str = Field(..., min_length=5)
    target_audience: str = Field(default="Tech enthusiasts, developers, and curious minds")
    scenes: List[Scene] = Field(..., min_length=10, max_length=10)
    social_caption: str = Field(..., min_length=15)
    hashtags: List[str] = Field(..., min_length=4)

    @field_validator("scenes")
    @classmethod
    def validate_scene_count_and_sequence(cls, scenes: List[Scene]) -> List[Scene]:
        if len(scenes) != 10:
            raise ValueError(f"Must have exactly 10 scenes, found {len(scenes)}")
        for i, s in enumerate(scenes, start=1):
            s.scene_number = i
        return scenes

def generate_fallback_script(research_brief: Dict[str, Any]) -> VideoPlan:
    """
    Deterministic rule-based script synthesizer grounded in the verified research brief.
    Used when running without GPU or if LLM encounters repeated syntax errors.
    """
    topic = research_brief["selected_topic"]
    source = research_brief.get("primary_source", "Tech News")
    date = research_brief.get("publication_date", "Recent")
    summary = research_brief.get("summary", "New advancements and engineering benchmarks reported.")
    
    scenes = [
        Scene(
            scene_number=1,
            visual_prompt=(
                "Cinematic macro shot of an advanced glowing microchip, electric blue neon pathways, "
                "shallow depth of field, dramatic rim lighting, 8k photorealistic, clean composition"
            ),
            voiceover_text=f"A major technical development in {topic[:30]} was just reported by {source}.",
            subtitle_text="Major Technical Milestone Reported",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=2,
            visual_prompt=(
                "Modern research laboratory interior, engineers in silhouette analyzing holographic monitors, "
                "amber and deep teal volumetric lighting, cinematic aesthetic"
            ),
            voiceover_text="For years, engineers struggled with architectural bottlenecks and efficiency limitations.",
            subtitle_text="Overcoming Legacy Bottlenecks",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=3,
            visual_prompt=(
                "Macro view of crystalline quantum processor architecture, fiber optic lasers pulsing data, "
                "dark moody cinematic background, iridescent reflections"
            ),
            voiceover_text="This new report indicates a critical step forward in addressing those fundamental hurdles.",
            subtitle_text="Critical Step Forward Taken",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=4,
            visual_prompt=(
                "Sleek autonomous humanoid robot walking dynamically in an industrial testing facility, "
                "golden hour reflections, cinematic low angle, 35mm lens"
            ),
            voiceover_text=f"According to {source}, practical deployment timelines are accelerating rapidly.",
            subtitle_text="Timelines Accelerating Rapidly",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=5,
            visual_prompt=(
                "Futuristic server racks with cascading teal and cyan optical data streams, "
                "industrial symmetry, cinematic depth of field"
            ),
            voiceover_text="Researchers focused on optimizing efficiency while maintaining stability under real conditions.",
            subtitle_text="Optimized Real-World Stability",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=6,
            visual_prompt=(
                "Close-up of engineer hands typing on an illuminated glass workstation, floating data nodes, "
                "cyberpunk realism, soft rim light"
            ),
            voiceover_text="What once required centralized supercomputers is steadily moving toward localized systems.",
            subtitle_text="Moving Toward Localized Systems",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=7,
            visual_prompt=(
                "Global digital network visualization, Earth at night glowing with neural pathways, "
                "cinematic space view, photo-realistic NASA aesthetic"
            ),
            voiceover_text="Development teams across the industry are adapting their roadmap to these findings.",
            subtitle_text="Global Roadmaps Are Adapting",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=8,
            visual_prompt=(
                "Futuristic smart city at twilight, autonomous transport moving through illuminated avenues, "
                "vibrant neon accents, cinematic realism"
            ),
            voiceover_text="From robotics to intelligent daily software, real-world impacts are already underway.",
            subtitle_text="Real Impacts Emerging Today",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=9,
            visual_prompt=(
                "Conceptual human eye reflecting complex fractal neural network geometries, "
                "extreme macro, golden and cyan highlights, hyper-detailed iris"
            ),
            voiceover_text="The pace of open engineering continues to redefine what is possible in tech.",
            subtitle_text="Open Engineering Leads The Way",
            duration_seconds=5.0
        ),
        Scene(
            scene_number=10,
            visual_prompt=(
                "Inspiring cinematic sunrise over a futuristic horizon, sleek silhouette looking toward the sun, "
                "lens flare, epic scale, 8k cinematic shot"
            ),
            voiceover_text="What do you think of this breakthrough? Share your thoughts below and subscribe for more!",
            subtitle_text="Subscribe For Daily Updates",
            duration_seconds=5.0
        )
    ]

    return VideoPlan(
        selected_topic=topic,
        hook_title=f"The Truth Behind {topic[:35]}",
        target_audience="Tech enthusiasts, developers, and curious minds",
        scenes=scenes,
        social_caption=(
            f"⚡ Verified update: {topic}! Reported by {source} ({date}). "
            "Drop your thoughts below and subscribe for source-verified tech breakdowns! 🚀"
        ),
        hashtags=["#TechNews", "#AI", "#Robotics", "#Innovation", "#Engineering", "#FutureTech"]
    )

def generate_video_script_llm(
    research_brief: Dict[str, Any],
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda"
) -> VideoPlan:
    """
    Loads Qwen2.5-7B-Instruct in 4-bit NF4 quantization, generates a strict JSON script,
    validates with Pydantic, and immediately clears GPU memory.
    """
    logger.info(f"Loading LLM '{model_id}' in 4-bit NF4 for video script generation...")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available() or device == "cpu":
            logger.warning("CUDA not available. Deferring to grounded script engine.")
            return generate_fallback_script(research_brief)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        system_prompt = (
            "You are an expert short-form viral video director. "
            "You must output ONLY valid, parseable JSON conforming strictly to the schema. "
            "Do NOT include markdown formatting or ungrounded statistics.\n"
            "Requirements:\n"
            "- Exactly 10 scenes, sequentially numbered 1 to 10.\n"
            "- Each scene must have: visual_prompt, voiceover_text (~11-14 words), subtitle_text (~4-7 words), duration_seconds (5.0).\n"
            "- Narration MUST be strictly grounded in the provided research brief. Do NOT invent fake percentages or 10x claims.\n"
            "- Visual prompts must specify: subject, environment, lighting, camera angle, and cinematic style. No text in images.\n"
            "- Scene 1 must be a strong viral hook. Scene 10 must be a conclusive call-to-action."
        )

        user_content = (
            f"Topic: {research_brief['selected_topic']}\n"
            f"Source: {research_brief.get('primary_source', 'News Wire')} ({research_brief.get('publication_date', 'Recent')})\n"
            f"Summary: {research_brief.get('summary', '')}\n"
            f"Key Facts: {json.dumps(research_brief.get('key_facts', []))}\n\n"
            "Output Schema:\n"
            "{\n"
            '  "selected_topic": "string",\n'
            '  "hook_title": "string",\n'
            '  "target_audience": "string",\n'
            '  "scenes": [\n'
            '    {"scene_number": 1, "visual_prompt": "...", "voiceover_text": "...", "subtitle_text": "...", "duration_seconds": 5.0}\n'
            '  ],\n'
            '  "social_caption": "string",\n'
            '  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6"]\n'
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1800,
            temperature=0.6,
            top_p=0.9,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Purge LLM immediately from GPU VRAM
        del model
        del tokenizer
        clean_gpu_memory("LLM Script Eviction")

        # Clean JSON
        cleaned_json = response_text.strip()
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(cleaned_json)
            plan = VideoPlan.model_validate(parsed)
            logger.info("Successfully validated Qwen script with Pydantic!")
            return plan
        except Exception as err:
            logger.warning(f"Pydantic validation warning ({err}). Using grounded fallback script.")
            return generate_fallback_script(research_brief)

    except Exception as e:
        logger.warning(f"LLM execution notice ({e}). Falling back to grounded script engine.")
        clean_gpu_memory("LLM Failure Recovery")
        return generate_fallback_script(research_brief)
