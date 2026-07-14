"""
Curiosity Scoreboard — multi-dimensional exploration scoring for receptacle prioritization.

Spike 003: Instead of blocking bad intents reactively (002a) or binary searched-markers (001),
we compute a CONTINUOUS multi-dimensional curiosity score for every known receptacle and
present it as a structured table in the Planner's prompt. This makes unexplored locations
actively ATTRACTIVE (curiosity bonus) and visited locations progressively LESS attractive
(habituation penalty via exponential decay).

Combines three signals:
1. Semantic Prior: LLM-derived commonsense "where is X usually found?" (static table)
2. Novelty Score: Exponential decay on visitation count (habituation from neuroscience)
3. Discovery Score: Positive reinforcement — did opening this reveal new objects?

Design inspiration:
- SENSEI (ICML 2025): VLM-derived "interestingness" reward for model-based RL
- Episodic Curiosity / Reachability (Savinov et al., ICLR 2019): novelty via reachability
- FICM (Yang et al., 2019): avoiding catastrophic forgetting of visited states
- Semantic Frontier Exploration (Noda & Tanaka, 2025): LLM-guided frontier prioritization
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Semantic Prior Table — "where is X likely to be found?"
#
# Maps (object_category, receptacle_type) -> prior score [0.0, 1.0]
# Each object in the task is mapped to a category via _classify_target().
# The prior reflects commonsense knowledge about household organization.
# ---------------------------------------------------------------------------

# Broad object categories (mapped from specific object types via _classify_target)
OBJECT_CATEGORIES: dict[str, list[str]] = {
    "food_cold": ["Apple", "Bread", "Egg", "Potato", "Tomato", "Lettuce",
                  "Butter", "Cheese", "Milk", "Yogurt", "Fruit", "Vegetable",
                  "Sandwich", "Burger", "Pizza"],
    "food_dry": ["BreadSliced", "Cereal", "Chips", "Cracker", "Cookie",
                 "Pretzel", "Rice", "Pasta", "Bean", "Nut"],
    "drink": ["Bottle", "Cup", "Mug", "WineBottle", "WaterBottle",
              "JuiceBottle", "SodaCan", "CoffeeMachine"],
    "electronics": ["Laptop", "CellPhone", "Tablet", "Television", "RemoteControl",
                    "AlarmClock", "CD", "CreditCard", "Watch", "KeyChain"],
    "bathroom": ["SoapBottle", "SoapBar", "Towel", "ToiletPaper", "SprayBottle",
                 "Plunger", "ScrubBrush", "TissueBox", "HandTowel", "Cloth"],
    "bedroom": ["Pillow", "Blanket", "Sheet", "Duvet", "Book", "Pen",
                "Pencil", "TeddyBear", "Plush", "AlarmClock"],
    "cleaning": ["SprayBottle", "ScrubBrush", "Plunger", "VacuumCleaner",
                 "Broom", "DustPan", "Rag", "PaperTowelRoll"],
    "utensil": ["ButterKnife", "Fork", "Spoon", "Knife", "Spatula",
                "Ladle", "WineBottle", "Spatula", "Pot", "Pan",
                "Bowl", "Plate", "DishSponge"],
    "container": ["Box", "Basket", "Crate", "Bag", "Suitcase", "Backpack"],
}

# Receptacle types (common across AI2-THOR FloorPlan scenes)
RECEPTACLE_CATEGORIES: dict[str, str] = {
    "Fridge": "cold_storage",
    "Freezer": "cold_storage",
    "Microwave": "heating_appliance",
    "StoveBurner": "heating_appliance",
    "Toaster": "heating_appliance",
    "CoffeeMachine": "heating_appliance",
    "CounterTop": "work_surface",
    "DiningTable": "table",
    "SideTable": "table",
    "CoffeeTable": "table",
    "Desk": "work_surface",
    "Cabinet": "closed_storage",
    "Drawer": "closed_storage",
    "Dresser": "closed_storage",
    "Nightstand": "closed_storage",
    "TVStand": "closed_storage",
    "Shelf": "open_storage",
    "Sink": "wet_area",
    "SinkBasin": "wet_area",
    "Bathtub": "wet_area",
    "BathtubBasin": "wet_area",
    "Toilet": "wet_area",
    "Bed": "furniture",
    "Sofa": "furniture",
    "ArmChair": "furniture",
    "GarbageCan": "disposal",
    "LaundryHamper": "disposal",
    "Box": "container",
    "Basket": "container",
    "Safe": "container",
}

# Semantic prior scores: P(target_category | receptacle_category)
# Representing "how likely is an object of category X to be found in receptacle Y?"
SEMANTIC_PRIORS: dict[tuple[str, str], float] = {
    # Cold storage — food_cold is very likely, others not
    ("food_cold", "cold_storage"): 0.95,
    ("food_dry", "cold_storage"): 0.40,  # sometimes
    ("drink", "cold_storage"): 0.70,
    ("electronics", "cold_storage"): 0.05,
    ("bathroom", "cold_storage"): 0.05,
    ("cleaning", "cold_storage"): 0.05,
    ("bedroom", "cold_storage"): 0.05,
    ("utensil", "cold_storage"): 0.10,
    ("container", "cold_storage"): 0.10,

    # Work surfaces / tables — food and utensils commonly placed here
    ("food_cold", "work_surface"): 0.70,
    ("food_dry", "work_surface"): 0.70,
    ("drink", "work_surface"): 0.80,
    ("electronics", "work_surface"): 0.60,
    ("bathroom", "work_surface"): 0.40,
    ("cleaning", "work_surface"): 0.30,
    ("bedroom", "work_surface"): 0.30,
    ("utensil", "work_surface"): 0.60,
    ("container", "work_surface"): 0.50,

    # Tables — like work surfaces, social gathering spots
    ("food_cold", "table"): 0.60,
    ("food_dry", "table"): 0.50,
    ("drink", "table"): 0.90,
    ("electronics", "table"): 0.40,
    ("bathroom", "table"): 0.10,
    ("cleaning", "table"): 0.10,
    ("bedroom", "table"): 0.20,
    ("utensil", "table"): 0.30,
    ("container", "table"): 0.30,

    # Closed storage — many things in drawers, cabinets
    ("food_cold", "closed_storage"): 0.50,
    ("food_dry", "closed_storage"): 0.80,
    ("drink", "closed_storage"): 0.40,
    ("electronics", "closed_storage"): 0.50,
    ("bathroom", "closed_storage"): 0.60,
    ("cleaning", "closed_storage"): 0.80,
    ("bedroom", "closed_storage"): 0.60,
    ("utensil", "closed_storage"): 0.80,
    ("container", "closed_storage"): 0.20,

    # Open storage — shelves. Electronics, bedroom items common
    ("food_cold", "open_storage"): 0.30,
    ("food_dry", "open_storage"): 0.70,
    ("drink", "open_storage"): 0.30,
    ("electronics", "open_storage"): 0.50,
    ("bathroom", "open_storage"): 0.50,
    ("cleaning", "open_storage"): 0.40,
    ("bedroom", "open_storage"): 0.60,
    ("utensil", "open_storage"): 0.50,
    ("container", "open_storage"): 0.40,

    # Wet areas (sink, toilet area)
    ("food_cold", "wet_area"): 0.15,
    ("food_dry", "wet_area"): 0.10,
    ("drink", "wet_area"): 0.20,
    ("electronics", "wet_area"): 0.05,
    ("bathroom", "wet_area"): 0.80,
    ("cleaning", "wet_area"): 0.70,
    ("bedroom", "wet_area"): 0.10,
    ("utensil", "wet_area"): 0.40,
    ("container", "wet_area"): 0.10,

    # Furniture (bed, sofa)
    ("food_cold", "furniture"): 0.05,
    ("food_dry", "furniture"): 0.10,
    ("drink", "furniture"): 0.10,
    ("electronics", "furniture"): 0.20,
    ("bathroom", "furniture"): 0.30,
    ("cleaning", "furniture"): 0.05,
    ("bedroom", "furniture"): 0.90,
    ("utensil", "furniture"): 0.05,
    ("container", "furniture"): 0.20,

    # Heating appliances
    ("food_cold", "heating_appliance"): 0.50,
    ("food_dry", "heating_appliance"): 0.60,
    ("drink", "heating_appliance"): 0.30,
    ("electronics", "heating_appliance"): 0.05,
    ("bathroom", "heating_appliance"): 0.05,
    ("cleaning", "heating_appliance"): 0.05,
    ("bedroom", "heating_appliance"): 0.05,
    ("utensil", "heating_appliance"): 0.20,
    ("container", "heating_appliance"): 0.05,

    # Disposal
    ("food_cold", "disposal"): 0.05,
    ("food_dry", "disposal"): 0.05,
    ("drink", "disposal"): 0.05,
    ("electronics", "disposal"): 0.01,
    ("bathroom", "disposal"): 0.05,
    ("cleaning", "disposal"): 0.05,
    ("bedroom", "disposal"): 0.05,
    ("utensil", "disposal"): 0.05,
    ("container", "disposal"): 0.05,

    # Container
    ("food_cold", "container"): 0.20,
    ("food_dry", "container"): 0.30,
    ("drink", "container"): 0.20,
    ("electronics", "container"): 0.40,
    ("bathroom", "container"): 0.20,
    ("cleaning", "container"): 0.20,
    ("bedroom", "container"): 0.40,
    ("utensil", "container"): 0.30,
    ("container", "container"): 0.60,
}

# Default prior when mapping is missing — conservative
DEFAULT_SEMANTIC_PRIOR = 0.25

# Scoring weights for combined curiosity score
WEIGHT_SEMANTIC = 0.40   # how likely is the target here?
WEIGHT_NOVELTY = 0.40    # how fresh is this location?
WEIGHT_DISCOVERY = 0.20  # did this location yield discoveries before?

# Thresholds for soft/hard guard
SCORE_SOFT_GUARD = 0.30  # if proposed target < this AND alternatives >= 0.50 exist, warn
SCORE_HARD_GUARD = 0.10  # if proposed target < this, force fallback

# Exponential decay rate for novelty: novelty = exp(-visit_count * DECAY_RATE)
HABITUATION_DECAY = 1.0  # each visit cuts novelty to ~37% of previous


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def _classify_target(object_type: str) -> str:
    """Map a specific objectType to a broad semantic category."""
    for category, members in OBJECT_CATEGORIES.items():
        if object_type in members:
            return category
    # Fuzzy matching for partial matches
    lower = object_type.lower()
    if any(w in lower for w in ["apple", "bread", "egg", "potato", "tomato",
                                  "lettuce", "food", "fruit", "vegetable", "meat",
                                  "sandwich", "burger", "pizza"]):
        return "food_cold"
    if any(w in lower for w in ["mug", "cup", "bottle", "drink", "wine", "juice",
                                  "soda", "water", "coffee"]):
        return "drink"
    if any(w in lower for w in ["laptop", "phone", "tablet", "remote", "clock",
                                  "television", "tv", "creditcard", "watch"]):
        return "electronics"
    if any(w in lower for w in ["soap", "towel", "toilet", "tissue", "cloth",
                                  "spray", "scrub", "plunger", "shampoo"]):
        return "bathroom"
    if any(w in lower for w in ["pillow", "blanket", "sheet", "book", "pen",
                                  "pencil", "teddy", "plush"]):
        return "bedroom"
    if any(w in lower for w in ["knife", "fork", "spoon", "spatula", "ladle",
                                  "pot", "pan", "bowl", "plate", "dish", "sponge"]):
        return "utensil"
    if any(w in lower for w in ["box", "basket", "crate", "bag", "suitcase"]):
        return "container"
    return "food_cold"  # safest default


def get_semantic_prior(target_object_type: str, receptacle_type: str) -> float:
    """Compute semantic prior: P(target | receptacle).

    Uses the static SEMANTIC_PRIORS table, mapping through category hierarchies.
    Returns a score in [0.0, 1.0].
    """
    target_cat = _classify_target(target_object_type)
    recep_cat = RECEPTACLE_CATEGORIES.get(receptacle_type, "work_surface")
    return SEMANTIC_PRIORS.get((target_cat, recep_cat), DEFAULT_SEMANTIC_PRIOR)


def compute_novelty_score(visit_count: int) -> float:
    """Exponential habituation: each visit cuts attractiveness.

    novelty = exp(-visit_count * HABITUATION_DECAY)

    visit_count 0 -> 1.00  (brand new: highest novelty)
    visit_count 1 -> 0.37  (checked once: low novelty)
    visit_count 2 -> 0.14  (checked twice: very low novelty)
    visit_count 3 -> 0.05  (checked 3x: nearly zero novelty)
    """
    return math.exp(-visit_count * HABITUATION_DECAY)


def compute_discovery_score(objects_found: int, times_opened: int) -> float:
    """Objects discovered per interaction. Rewards receptacles that revealed new objects.

    If never opened: assume unknown potential (score = 0.5).
    If opened at least once: score = min(1.0, objects_found / max(1, times_opened)).
    """
    if times_opened == 0:
        return 0.5  # unknown — moderate potential
    return min(1.0, objects_found / max(1, times_opened))


def compute_combined_score(
    target_object_type: str,
    receptacle_type: str,
    visit_count: int,
    objects_found: int,
    times_opened: int,
) -> float:
    """Aggregate into a single curiosity score [0.0, 1.0]."""
    semantic = get_semantic_prior(target_object_type, receptacle_type)
    novelty = compute_novelty_score(visit_count)
    discovery = compute_discovery_score(objects_found, times_opened)
    return (
        WEIGHT_SEMANTIC * semantic
        + WEIGHT_NOVELTY * novelty
        + WEIGHT_DISCOVERY * discovery
    )


# ---------------------------------------------------------------------------
# Scoreboard entry and table formatter
# ---------------------------------------------------------------------------

@dataclass
class ScoreboardEntry:
    """One row in the curiosity scoreboard."""
    receptacle_type: str
    location_hint: str        # "in view ahead, 1.2m" or "remembered, saw 5 steps ago"
    receptacle_category: str  # "cold_storage", "work_surface", etc
    visit_count: int
    times_opened: int
    objects_found: int
    semantic_prior: float     # [0, 1]
    novelty_score: float      # [0, 1]
    discovery_score: float    # [0, 1]
    combined_score: float     # [0, 1]
    signal: str               # "★★★ EXPLORE", "★★ EXPLORE", "★ MAYBE", "✗ AVOID"


def _score_to_signal(score: float, visit_count: int) -> str:
    """Convert a combined score to a human-readable signal."""
    if score >= 0.70:
        return "★★★ EXPLORE"
    elif score >= 0.40:
        return "★★ EXPLORE"
    elif score >= 0.25:
        return "★ MAYBE"
    else:
        tag = f"✗ AVOID"
        if visit_count > 0:
            tag += f" (visited {visit_count}x)"
        return tag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_curiosity_table_text(
    target_object: str,
    memory_entries: list[dict],
    visit_counts: dict[str, int],
    open_counts: dict[str, int],
    objects_found_counts: dict[str, int],
    max_rows: int = 8,
) -> str:
    """Build the CURIOSITY SCOREBOARD as a formatted text block for prompt injection.

    Args:
        target_object: The task target object type (e.g., "Apple", "Pillow").
        memory_entries: List of receptacle entries from EgocentricMemory.
            Each entry is dict with: object_type, status, egocentric_dir, egocentric_dist,
            last_seen_step, current_step.
        visit_counts: Dict mapping receptacle type -> number of times visited.
        open_counts: Dict mapping receptacle type -> number of times opened.
        objects_found_counts: Dict mapping receptacle type -> number of new objects discovered.
        max_rows: Maximum number of rows to show (clamped to top-scoring entries).

    Returns:
        Formatted curiosity scoreboard text, or empty string if no receptacles.
    """
    if not memory_entries:
        return ""

    entries: list[ScoreboardEntry] = []

    for rec in memory_entries:
        otype = rec["object_type"]
        vc = visit_counts.get(otype, 0)
        oc = open_counts.get(otype, 0)
        of = objects_found_counts.get(otype, 0)

        # Location hint
        if rec["status"] == "visible":
            loc = f"visible {rec['egocentric_dir']} {rec['egocentric_dist']:.1f}m"
        else:
            age = rec.get("age_steps", 0)
            loc = f"remembered (~{age} steps ago)"

        sp = get_semantic_prior(target_object, otype)
        nv = compute_novelty_score(vc)
        ds = compute_discovery_score(of, oc)
        combined = compute_combined_score(target_object, otype, vc, of, oc)
        signal = _score_to_signal(combined, vc)

        entries.append(ScoreboardEntry(
            receptacle_type=otype,
            location_hint=loc,
            receptacle_category=RECEPTACLE_CATEGORIES.get(otype, "unknown"),
            visit_count=vc,
            times_opened=oc,
            objects_found=of,
            semantic_prior=sp,
            novelty_score=nv,
            discovery_score=ds,
            combined_score=combined,
            signal=signal,
        ))

    # Sort by combined score descending
    entries.sort(key=lambda e: e.combined_score, reverse=True)
    entries = entries[:max_rows]

    # Build the table
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("CURIOSITY SCOREBOARD — ranked by exploration priority")
    lines.append(f"Target: {target_object} | Higher score = explore first")
    lines.append("=" * 72)
    lines.append("")

    # Header
    header = f" {'#':<2} {'Location':<14} {'Where':<22} {'Visit':>5} {'Sem':>5} {'Nov':>5} {'Disc':>5} {'Score':>6} Signal"
    lines.append(header)
    lines.append("-" * 72)

    for i, e in enumerate(entries, 1):
        row = (
            f" {i:<2} {e.receptacle_type:<14} {e.location_hint:<22} "
            f"{e.visit_count:>5} {e.semantic_prior:>5.2f} {e.novelty_score:>5.2f} "
            f"{e.discovery_score:>5.2f} {e.combined_score:>6.2f} {e.signal}"
        )
        lines.append(row)

    lines.append("-" * 72)
    lines.append(f"Sem=Semantic prior (how likely is {target_object} here?)")
    lines.append(f"Nov=Novelty (1.0=never visited, decays with each visit)")
    lines.append(f"Disc=Discovery (objects found here / times opened)")
    lines.append("")
    lines.append("GUIDANCE: Your proposed intent MUST target a location scoring >= 0.30.")
    lines.append("Locations scoring < 0.30 have been visited multiple times or are")
    lines.append("semantically unlikely. EXPLORE the highest-scoring unvisited location first.")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


def check_proposed_target(
    target_object: str,
    proposed_target: str,
    visit_counts: dict[str, int],
    open_counts: dict[str, int],
    objects_found_counts: dict[str, int],
) -> dict:
    """Post-hoc check: is the proposed target reasonable?

    Returns dict with:
      - blocked: bool — force override?
      - warning: str | None — soft constraint for next prompt?
      - score: float — combined score of proposed target
      - best_alternative: str | None — highest-scoring alternative if blocked/warned
      - best_alt_score: float — score of the best alternative
    """
    result: dict = {
        "blocked": False,
        "warning": None,
        "score": 0.0,
        "best_alternative": None,
        "best_alt_score": 0.0,
    }

    # Score the proposed target
    vc = visit_counts.get(proposed_target, 0)
    oc = open_counts.get(proposed_target, 0)
    of = objects_found_counts.get(proposed_target, 0)
    score = compute_combined_score(target_object, proposed_target, vc, of, oc)
    result["score"] = score

    # Find best alternative (for warning/fallback)
    best_alt = None
    best_alt_score = 0.0
    all_receptacles = set(visit_counts.keys()) | set(open_counts.keys())
    for rec in all_receptacles:
        if rec == proposed_target:
            continue
        avc = visit_counts.get(rec, 0)
        aoc = open_counts.get(rec, 0)
        aof = objects_found_counts.get(rec, 0)
        ascore = compute_combined_score(target_object, rec, avc, aof, aoc)
        if ascore > best_alt_score:
            best_alt_score = ascore
            best_alt = rec
    result["best_alternative"] = best_alt
    result["best_alt_score"] = best_alt_score

    # Hard guard: score < 0.10 → force override
    if score < SCORE_HARD_GUARD and best_alt is not None and best_alt_score >= 0.25:
        result["blocked"] = True
        return result

    # Soft guard: score < 0.30 AND there's a significantly better alternative
    if score < SCORE_SOFT_GUARD and best_alt is not None and best_alt_score >= 0.50:
        result["warning"] = (
            f"LOCATION SCORE WARNING: '{proposed_target}' curiosity score is only "
            f"{score:.2f} (visited {vc} times). The best unexplored alternative is "
            f"'{best_alt}' with score {best_alt_score:.2f}. STRONGLY consider "
            f"exploring '{best_alt}' instead."
        )

    return result


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

@dataclass
class CuriosityStats:
    """Per-episode curiosity scoreboard statistics."""
    scoreboard_shown: int = 0        # times scoreboard was included in prompt
    soft_warnings: int = 0           # soft guard triggered
    hard_blocks: int = 0             # hard guard triggered (force fallback)
    total_locations_scored: int = 0  # unique receptacles scored
    avg_score_proposed: list[float] = field(default_factory=list)  # scores of proposed targets
    blocked_intents: list[dict] = field(default_factory=list)  # {proposed, score, fallback}

    def record_proposal(self, proposed_target: str, score: float):
        self.avg_score_proposed.append(score)

    def record_block(self, proposed_target: str, score: float, fallback: str):
        self.hard_blocks += 1
        self.blocked_intents.append({
            "proposed": proposed_target,
            "score": round(score, 3),
            "fallback": fallback,
        })

    def to_dict(self) -> dict:
        avg = (sum(self.avg_score_proposed) / max(1, len(self.avg_score_proposed)))
        return {
            "scoreboard_shown": self.scoreboard_shown,
            "soft_warnings": self.soft_warnings,
            "hard_blocks": self.hard_blocks,
            "total_locations_scored": self.total_locations_scored,
            "avg_score_of_proposed_targets": round(avg, 3),
            "blocked_intents": self.blocked_intents,
        }
