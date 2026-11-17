"""Deterministically loaded model profiles, not opportunistic agent skill discovery."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path

REFERENCE_ROLE_POLICY_VERSION = 8
PACK_VERSION = "1.0.2-reference-role-contract"
ROOT = Path(__file__).parent / "prompt_skills/wondful-render-director"
PROFILES = {
    "codex": ("openai-image", "OpenAI ImageGen / Codex"),
    "antigravity": ("gemini-image", "Gemini Image / Antigravity"),
}


def provider_id(value):
    name = str(value or "codex").lower()
    if name not in PROFILES:
        raise ValueError(f"Unsupported image provider: {value}")
    return name


@lru_cache(maxsize=2)
def _profile_cached(value):
    name = provider_id(value)
    slug, label = PROFILES[name]
    paths = [ROOT / "SKILL.md", ROOT / "references/commercial.md", ROOT / f"references/{slug}.md"]
    texts = [p.read_text(encoding="utf-8") for p in paths]
    # Changes to either the common policy or target profile invalidate cached summaries.
    digest = hashlib.sha256((PACK_VERSION + "\n" + "\n".join(texts)).encode()).hexdigest()
    return {"id": slug, "provider": name, "label": label, "pack_version": PACK_VERSION,
            "sha256": digest, "instructions": "\n\n".join(texts), "image_model": None,
            "image_model_source": "CLI built-in tool; not selectable with agent --model"}



def profile(value):
    return dict(_profile_cached(provider_id(value)))

def target_key(value, agent_model=""):
    p = profile(value)
    return hashlib.sha256(json.dumps([p["provider"], agent_model or "default", p["sha256"]], ensure_ascii=False).encode()).hexdigest()


def metadata(value, agent_model=""):
    p = profile(value)
    p.pop("instructions")
    p["agent_model_requested"] = agent_model or "default"
    p["target_key"] = target_key(value, agent_model)
    return p


def needs_target_refresh(props, prefs):
    if not str(getattr(props, "last_ai_prompt", "") or "").strip():
        return False  # A manually authored first prompt remains usable.
    pid = provider_id(getattr(props, "analysis_provider", "CODEX"))
    agent = getattr(prefs, "antigravity_model" if pid == "antigravity" else "codex_model", "")
    return getattr(props, "prompt_target_key", "") != target_key(pid, agent)


def system_prompt(base, value):
    return base.strip() + "\n\n--- Versioned Wondful art direction / target profile ---\n" + profile(value)["instructions"]


def render_brief(text, value):
    if provider_id(value) == "codex":
        return ("EDIT TASK: appearance rerender of the supplied Blender edit canvas.\n"
                "REFERENCE ROLES: use the provided manifest; source photographs are not interchangeable edit targets.\n"
                "APPEARANCE BRIEF:\n" + text.strip() +
                "\nPRESERVE: canvas geometry and requested product identity; change only requested appearance.\n"
                "OUTPUT: one generated bitmap, no annotations or instruction text. Use only actual image-tool parameters.")
    return ("Use the supplied edit canvas and the explicitly labelled ImagePaths to create one coherent commercial image. "
            "Read the requested appearance below as a connected description of how environment, illumination, "
            "reflections and materials work together, not as unrelated keywords.\n" + text.strip() +
            "\nRelight the product in that environment without inheriting the product photograph's lighting. "
            "Keep the canvas geometry and identity intact; return one bitmap without labels or extra copy. "
            "Do not invent API-only controls not exposed by generate_image.")


def style_cache_key(fingerprint, instructions, value, agent_model=""):
    return hashlib.sha256(json.dumps([REFERENCE_ROLE_POLICY_VERSION, fingerprint, list(instructions or []), target_key(value, agent_model)], ensure_ascii=False).encode()).hexdigest()
