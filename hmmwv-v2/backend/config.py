"""
Configuration, constants, and settings persistence for HMMWV Technical Assistant v2.
Extracted from app.py — all logic preserved verbatim.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

# hmmwv-v2/backend/config.py → parent → hmmwv-v2 → parent → worktree root
BASE_DIR             = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_DIR   = BASE_DIR / "knowledge_base"
EXTRACTED_IMAGES_DIR = BASE_DIR / "extracted_images"
CHROMA_PERSIST_DIR   = BASE_DIR / "chroma_db"
SETTINGS_FILE        = BASE_DIR / ".hmmwv_settings.json"
SESSIONS_DIR         = Path(__file__).resolve().parent / "sessions"

for d in [KNOWLEDGE_BASE_DIR, EXTRACTED_IMAGES_DIR, CHROMA_PERSIST_DIR, SESSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER / MODEL DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

OLLAMA_DEFAULT_URL      = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL    = "gpt-oss:latest"
OPENAI_DEFAULT_URL      = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL    = "gpt-4o"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-6"

PROVIDER_OLLAMA    = "Ollama (Local)"
PROVIDER_OPENAI    = "OpenAI-Compatible"
PROVIDER_ANTHROPIC = "Anthropic (Claude)"
ALL_PROVIDERS      = [PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC]

# ═══════════════════════════════════════════════════════════════════════════
# TUNING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

MAX_TOKENS    = 4096
TEMPERATURE   = 0.2
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 8
COLLECTION_NAME = "hmmwv_manuals"
MIN_IMAGE_SIZE  = (100, 100)

# BM25 retrieval parameters
BM25_K1        = 1.5
BM25_B         = 0.75
MIN_SIMILARITY = 0.08

# ═══════════════════════════════════════════════════════════════════════════
# VEHICLE VARIANTS & MAINTENANCE CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════

HMMWV_VARIANTS = [
    "M998 — Cargo/Troop Carrier",
    "M1025 — Armament Carrier",
    "M1035 — Soft Top Ambulance",
    "M1043 — Up-Armored Armament Carrier",
    "M1044 — Up-Armored Armament Carrier w/ Winch",
    "M1097 — Heavy HMMWV",
    "M1113 — Expanded Capacity Vehicle (ECV)",
    "M1114 — Up-Armored HMMWV",
    "M1151 — Enhanced Armament Carrier",
    "M1152 — Enhanced Cargo/Troop Carrier",
    "M1165 — Up-Armored",
    "M1167 — TOW Carrier",
]

MAINTENANCE_CATEGORIES = [
    "Engine & Powertrain",
    "Transmission & Transfer Case",
    "Suspension & Steering",
    "Brake System",
    "Electrical System",
    "Cooling System",
    "Fuel System",
    "Body & Frame",
    "CTIS (Central Tire Inflation System)",
    "HVAC System",
    "Winch System",
    "Weapons Mount / Turret",
    "NBC (Nuclear/Bio/Chem) System",
    "Preventive Maintenance (PMCS)",
]

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the **HMMWV Technical Assistant** — an expert AI system
designed to help military and civilian mechanics perform maintenance, repairs, and
inspections on High Mobility Multipurpose Wheeled Vehicles (HMMWV / Humvee).

## Your Role
- You are an experienced mechanic with over 20 years of hands-on experience maintaining
  and repairing HMMWVs, including extensive field work in combat and battle scenarios
  where improvisation, speed, and precision under pressure were critical.
- You assist mechanics of ALL experience levels — from first-time operators with no
  mechanical background to seasoned maintainers — by adapting your explanations to
  be clear, practical, and easy to follow regardless of the reader's skill level.
- You never assume prior knowledge. Explain every step as if the person has never
  performed the task before, including tool usage, safety posture, and why each
  step matters.
- You reference official TM (Technical Manual) procedures, part numbers, torque specs,
  and safety warnings directly from the knowledge base provided.
- You always prioritize SAFETY — highlight cautions, warnings, and dangerous steps.

## Response Format
When providing maintenance/repair instructions, structure your response as:

1. **Task Overview** — Brief description of what will be accomplished
2. **Safety Warnings** — Any critical cautions BEFORE starting
3. **Tools & Materials Required** — List every tool, part, and consumable needed
4. **Parts Information** — NSN numbers, part numbers where available
5. **Step-by-Step Procedure** — Numbered steps with:
   - Clear action descriptions
   - Torque specifications where applicable
   - References to diagrams (mention figure numbers from the manual)
   - ⚠️ WARNING/CAUTION callouts inline
6. **Quality Checks** — How to verify the work was done correctly
7. **Related Maintenance** — Other tasks the mechanic should consider

## Rules
- ALWAYS cite the source TM or document when referencing procedures.
- If the knowledge base does not contain information for a query, say so clearly
  and suggest where the mechanic might find the information.
- Use military-standard terminology but explain acronyms on first use.
- When multiple HMMWV variants have different procedures, ASK which variant.
- Format torque specs as: XX ft-lbs (XX N·m)
- Provide NSN (National Stock Number) format: XXXX-XX-XXX-XXXX when available.

## Context
You have access to extracted content from HMMWV technical manuals. Use ONLY this
content to answer technical questions. The retrieved context chunks are provided
with each query.
"""

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-AGENT SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

AGENT1_RETRIEVER_PROMPT = """\
You are a document relevance analyst for HMMWV technical manuals.
Given a mechanic's question and retrieved document chunks, select the most relevant chunks.
Return ONLY a JSON array of the chunk indices (0-based) ordered by relevance.
Example: [0, 2, 5, 1]
Be selective — include only chunks that directly answer the question or contain required specs.
"""

AGENT2_PROCEDURE_PROMPT = """\
You are an expert HMMWV maintenance procedure writer with 20+ years of hands-on experience.
Your SOLE task: Write the step-by-step maintenance procedure for the mechanic's question.
Use ONLY information from the provided technical manual context.

Format:
## Step-by-Step Procedure
1. [First step — be specific, include exact values, tool sizes, directions]
2. [Second step]
...

Rules:
- Include exact torque specs as both ft-lbs and N·m
- Include exact measurement values, clearances, pressures
- Note critical sequences and timing where applicable
- Do NOT include: safety warnings, parts lists, general overviews — those are handled by other agents
- If the context lacks sufficient procedure detail, state exactly what is missing
"""

AGENT3_SAFETY_PROMPT = """\
You are a HMMWV maintenance safety officer. Lives depend on you being thorough.
Your SOLE task: Identify ALL safety hazards, warnings, and precautions for the mechanic's question.
Use information from the provided technical manual context and standard HMMWV maintenance practices.

Format:
## ⚠️ Safety Warnings & Precautions

**DANGER / WARNING / CAUTION:**
- ⛔ DANGER: [Life-threatening hazards with consequences]
- ⚠️ WARNING: [Injury hazards]
- 🔔 CAUTION: [Equipment damage hazards]

**Required PPE:**
- [Personal Protective Equipment items]

**Hazard Zones & Conditions:**
- [Specific dangerous areas, voltages, pressures, temperatures]

**Pre-Task Safety Checks:**
- [Confirm vehicle is safe to work on — parking brake, chocks, etc.]

Be thorough. Do not include procedure steps or parts lists.
"""

AGENT4_PARTS_PROMPT = """\
You are a HMMWV parts and supply specialist (MOS 92A/91B trained).
Your SOLE task: List ALL parts, tools, and consumables required for the mechanic's question.
Use ONLY part numbers and NSNs from the provided technical manual context.

Format:
## 🔩 Parts & Materials Required

**Replacement Parts:**
| Part Name | NSN | Part Number | Qty |
|-----------|-----|-------------|-----|
| [name]    | [NSN or TBD] | [P/N or TBD] | [qty] |

**Special Tools Required:**
- [Tool name and specification — do not list common hand tools]

**Consumables & Fluids:**
- [Item: specification, quantity]

**Cross-References / Substitutes (if in context):**
- [Alternative part info]

Mark unknown NSNs/part numbers as "TBD — see TM." Do not include procedure steps or safety warnings.
"""

AGENT5_SIMPLIFIER_PROMPT = """\
You are a plain-language technical writer. Your audience: first-time mechanics with zero prior \
mechanical experience.
Your SOLE task: Rewrite the provided step-by-step procedure using simple, clear language that \
an absolute beginner can follow.

Rules:
- Define every technical term in parentheses on first use:
  e.g., "torque wrench (a special wrench that tightens bolts to a precise force)"
- Split complex steps into smaller sub-steps labeled a., b., c.
- Add "Why this matters:" notes for critical steps
- Add "You'll know it's correct when:" cues after key steps
- NEVER change any number, torque spec, measurement, or part number — copy them exactly
- Use direct language: "Turn clockwise" not "rotate in a dextral direction"
- Mention what the part looks like if helpful: "the round black rubber gasket"
- Output the rewritten procedure ONLY, using the same numbered structure
"""

AGENT6_EDITOR_PROMPT = """\
You are the HMMWV Technical Assistant final editor and synthesizer.
You receive outputs from three specialist agents plus a simplified procedure.
Your task: Synthesize these into ONE complete, coherent, well-structured maintenance guide.

Mandatory output structure:

## Task Overview
[2-3 sentence description of what will be accomplished and why it matters]

## ⚠️ Safety First
[Insert Safety Agent output — condense duplicates, keep all unique warnings]

## 🔩 Tools & Materials
[Insert Parts Agent output — clean up formatting]

## Step-by-Step Procedure
[Insert the SIMPLIFIED procedure — this must be the simplified/beginner-friendly version]

## ✅ Quality Verification
[How to confirm the task was completed correctly — derive from procedure context]

## 📎 Related Maintenance
[2-3 other tasks the mechanic should consider — derive from context]

Guidelines:
- Do NOT repeat information across sections
- If any agent reported missing data, note it clearly with "⚠️ Consult TM directly for [item]"
- Cite source TMs by name where the context provides document names
- Keep headers exactly as shown above — the app parses them for display
"""

# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _default_settings() -> dict:
    return {
        "provider": PROVIDER_OLLAMA,
        "ollama_url": OLLAMA_DEFAULT_URL,
        "ollama_model": OLLAMA_DEFAULT_MODEL,
        "openai_url": OPENAI_DEFAULT_URL,
        "openai_model": OPENAI_DEFAULT_MODEL,
        "openai_api_key": "",
        "anthropic_api_key": "",
        "anthropic_model": ANTHROPIC_DEFAULT_MODEL,
        "youtube_api_key": "",
        "youtube_enabled": True,
        "youtube_max_results": 3,
        "agent_mode": False,
        "agent1_enabled": False,
        "agent1_provider": PROVIDER_OLLAMA,
        "agent1_model": OLLAMA_DEFAULT_MODEL,
        "agent2_provider": PROVIDER_OLLAMA,
        "agent2_model": OLLAMA_DEFAULT_MODEL,
        "agent3_provider": PROVIDER_OLLAMA,
        "agent3_model": OLLAMA_DEFAULT_MODEL,
        "agent4_provider": PROVIDER_OLLAMA,
        "agent4_model": OLLAMA_DEFAULT_MODEL,
        "agent5_enabled": True,
        "agent5_provider": PROVIDER_OLLAMA,
        "agent5_model": OLLAMA_DEFAULT_MODEL,
        "agent6_provider": PROVIDER_OLLAMA,
        "agent6_model": OLLAMA_DEFAULT_MODEL,
    }


def load_settings() -> dict:
    defaults = _default_settings()
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_settings(settings: dict):
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not save settings: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION — HMMWV DOMAIN SYNONYMS
# ═══════════════════════════════════════════════════════════════════════════

_HMMWV_SYNONYMS: dict[str, list[str]] = {
    "won't start":    ["no crank", "fails start", "dead", "starter", "ignition", "cranking"],
    "wont start":     ["no crank", "fails start", "dead", "starter", "ignition"],
    "no start":       ["crank", "starter", "ignition", "battery", "fuel"],
    "hard start":     ["crank", "cold start", "glow plug", "fuel pressure"],
    "crank":          ["starter", "solenoid", "battery", "ignition", "turnover"],
    "starter":        ["solenoid", "crank", "ignition", "battery", "armature"],
    "engine":         ["motor", "powerplant", "diesel", "6.2", "6.5"],
    "overheating":    ["coolant", "thermostat", "radiator", "temperature", "hot"],
    "overheat":       ["coolant", "thermostat", "radiator", "temperature"],
    "knock":          ["ping", "preignition", "bearing", "rod", "piston"],
    "smoke":          ["exhaust", "burning", "blow-by", "rings", "turbo"],
    "idle":           ["rough idle", "rpm", "governor", "injector"],
    "fuel":           ["diesel", "injection", "injector", "pump", "filter", "line"],
    "injector":       ["fuel injection", "nozzle", "spray", "tip"],
    "fuel pump":      ["lift pump", "transfer pump", "injection pump"],
    "filter":         ["filtration", "strainer", "element", "primary", "secondary"],
    "coolant":        ["antifreeze", "water", "radiator", "overflow", "hose"],
    "radiator":       ["coolant", "cooling system", "hose", "cap", "flush"],
    "thermostat":     ["coolant temperature", "cooling", "overheat"],
    "oil":            ["lubricant", "lube", "engine oil", "crankcase", "sump"],
    "oil leak":       ["seal", "gasket", "o-ring", "pan", "drain plug"],
    "oil pressure":   ["pump", "gauge", "sender", "relief valve"],
    "battery":        ["batteries", "charge", "charging", "dead", "voltage", "24 volt"],
    "alternator":     ["charging", "voltage regulator", "belt", "generator"],
    "electrical":     ["wiring", "harness", "fuse", "relay", "circuit", "voltage"],
    "fuse":           ["circuit breaker", "electrical", "relay", "short"],
    "brake":          ["brakes", "braking", "pedal", "caliper", "rotor", "drum", "booster"],
    "brake fluid":    ["dot 3", "hydraulic fluid", "master cylinder", "caliper"],
    "steering":       ["steer", "power steering", "pump", "gear box", "tie rod", "wheel"],
    "wanders":        ["alignment", "tie rod", "steering", "caster", "camber"],
    "transmission":   ["trans", "gearbox", "shifting", "gear", "automatic", "allison"],
    "transfer case":  ["4wd", "four wheel drive", "4x4", "t-case", "selector"],
    "differential":   ["diff", "axle", "gear oil", "spider gear", "ring gear"],
    "driveshaft":     ["u-joint", "universal joint", "propeller shaft", "vibration"],
    "axle":           ["differential", "hub", "spindle", "bearing", "seal"],
    "suspension":     ["shock", "absorber", "spring", "control arm", "ball joint"],
    "vibration":      ["balance", "u-joint", "driveshaft", "wheel", "tire"],
    "tire":           ["tires", "tyre", "wheel", "ctis", "inflation", "pressure", "flat"],
    "ctis":           ["central tire inflation", "tire pressure", "air system"],
    "oil change":     ["drain plug", "filter", "engine oil", "crankcase", "lube"],
    "air filter":     ["air cleaner", "intake", "element", "restriction"],
    "belt":           ["alternator belt", "fan belt", "serpentine", "tension", "pulley"],
    "hose":           ["coolant hose", "clamp", "radiator hose", "heater hose"],
    "tune up":        ["glow plug", "filter", "injector", "timing", "service"],
    "torque":         ["ft-lb", "nm", "specification", "spec", "tighten"],
    "seal":           ["gasket", "o-ring", "leak", "seepage"],
    "pmcs":           ["preventive maintenance", "inspection", "before operation",
                       "after operation", "weekly", "monthly"],
    "check":          ["inspect", "verify", "ensure", "confirm", "examine"],
    "leak":           ["leaking", "seeping", "dripping", "weeping", "fluid loss"],
    "noise":          ["sound", "rattle", "clunk", "squeal", "groan", "whine"],
    "rattle":         ["loose", "bracket", "heat shield", "exhaust", "vibration"],
    "m998":           ["cargo", "basic hmmwv", "1-1/4 ton"],
    "m1025":          ["armament carrier", "weapons mount"],
    "m1114":          ["up-armored", "uah", "add-on armor"],
    "m1151":          ["enhanced armament", "eav"],
}

_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "was", "not", "with", "this", "that",
    "from", "have", "has", "had", "will", "would", "could", "should",
    "may", "can", "its", "it", "be", "been", "is", "in", "of", "to",
    "on", "at", "by", "an", "or", "if", "do", "all", "any", "but",
    "as", "so", "no", "up", "out", "use", "used", "per", "into", "than",
    "then", "when", "where", "which", "who", "also", "each", "only",
})


def _expand_query(query: str) -> str:
    """Expand a user query with HMMWV-domain synonyms and TM terminology."""
    q_lower = query.lower()
    extra: list[str] = []
    for key, synonyms in _HMMWV_SYNONYMS.items():
        if key in q_lower:
            extra.extend(synonyms)
    if not extra:
        return query
    return query + " " + " ".join(extra)
