"""Build evidence figures for the EF-Bench paper from recorded local artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
CASCADE = ROOT / "output_oracle_timed_cascade_20260722"
PILOT = ROOT / "output_quality_eval_20260726" / "trial_T20190907_165608_878138"


def box(ax, xy, text, color, width=2.25, height=0.8, fontsize=7.0):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.4, edgecolor=color, facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1f2933", linespacing=1.25)
    return (x, y, width, height)


def arrow(ax, source, target, text="", dashed=False):
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    start = (sx + sw, sy + sh / 2)
    end = (tx, ty + th / 2)
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
        linestyle="--" if dashed else "-", color="#52616b",
    ))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.16, text,
                ha="center", va="bottom", fontsize=7.3, color="#52616b")


def vertical_arrow(ax, source, target, text="", dashed=False):
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    start = (sx + sw / 2, sy)
    end = (tx + tw / 2, ty + th)
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
        linestyle="--" if dashed else "-", color="#52616b",
    ))
    if text:
        ax.text(start[0] + 0.12, (start[1] + end[1]) / 2, text,
                ha="left", va="center", fontsize=7.1, color="#52616b")


def build_platform():
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    ax.set_xlim(0, 14.3)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    inputs = box(
        ax, (0.10, 1.70), "TASK INPUTS\nGoal / scene / seed\nAgent configuration",
        "#455a64", width=1.75, height=1.55, fontsize=6.4,
    )
    agent = box(
        ax, (2.30, 1.05),
        "AGENT STACK\nPlanner -> executor\nSemantic memory + map\nCritic and safety guards",
        "#1976d2", width=2.65, height=2.85, fontsize=6.5,
    )
    world = box(
        ax, (5.45, 1.05),
        "AI2-THOR + FAILURES\n5 executable mutations\nAction-conditioned triggers\nConfirmed transitions",
        "#c62828", width=2.85, height=2.85, fontsize=6.5,
    )
    record = box(
        ax, (8.80, 1.05),
        "EVENT LOG + LINEAGE\nExact IDs + action traces\nApplied / triggered / recovered\nParent / fork links",
        "#00897b", width=2.65, height=2.85, fontsize=6.35,
    )
    verifier = box(
        ax, (11.95, 1.05),
        "REPLAY + FORK\nReconstruct prefix\nExecute alternative\nTest original goal",
        "#7b1fa2", width=2.15, height=2.85, fontsize=6.5,
    )

    arrow(ax, inputs, agent)
    arrow(ax, agent, world)
    arrow(ax, world, record)
    arrow(ax, record, verifier)
    ax.add_patch(FancyArrowPatch(
        (8.72, 1.32), (4.98, 1.32), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color="#52616b", connectionstyle="arc3,rad=-0.18",
    ))
    ax.text(6.80, 0.42, "observations and environment errors", ha="center",
            fontsize=6.5, color="#52616b")
    ax.text(9.45, 4.58, "Evaluator-controlled", ha="center", va="center",
            fontsize=6.6, color="#c62828")
    ax.plot([5.20, 14.20], [4.34, 4.34], linestyle="--", linewidth=0.9, color="#c62828")
    ax.text(3.62, 4.58, "Agent-visible", ha="center", va="center",
            fontsize=6.6, color="#1976d2")
    fig.tight_layout(pad=0.25)
    fig.savefig(FIGURES / "platform.pdf", bbox_inches="tight")
    plt.close(fig)


def build_scheduler_audit():
    fig, ax = plt.subplots(figsize=(7.15, 2.70))
    ax.set_xlim(0, 14.3)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    ax.text(0.15, 5.05, "(a) Bounded fair scheduler",
            fontsize=8.6, fontweight="bold", color="#1f2933")
    queue_labels = [("MAIN", "#1976d2"), ("FORK", "#7b1fa2"), ("FORK", "#7b1fa2")]
    for index, (label, color) in enumerate(queue_labels):
        x = 0.30 + index * 1.30
        patch = FancyBboxPatch(
            (x, 3.85), 1.05, 0.58, boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=1.2, edgecolor=color, facecolor="#ffffff",
        )
        ax.add_patch(patch)
        ax.text(x + 0.525, 4.14, label, ha="center", va="center", fontsize=7.7,
                fontweight="bold", color=color)
    ax.text(4.35, 4.14, "1:2 rotation while both queues wait", fontsize=6.7,
            va="center", color="#52616b")
    ax.add_patch(FancyArrowPatch(
        (0.30, 3.55), (6.55, 3.55), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color="#52616b",
    ))
    ax.text(3.42, 3.28, "work-conserving dispatch", ha="center", fontsize=7.2,
            color="#52616b")

    nodes = [
        (0.35, "main\nd=0", "#1976d2"),
        (1.85, "fork\nd=1", "#7b1fa2"),
        (3.35, "fork\nd=2", "#7b1fa2"),
        (4.85, "fork\nd=3", "#7b1fa2"),
    ]
    previous = None
    for x, label, color in nodes:
        node = box(ax, (x, 1.10), label, color, width=1.05, height=0.86, fontsize=6.8)
        if previous:
            arrow(ax, previous, node)
        previous = node
    blocked = box(ax, (6.20, 1.10), "d=4\nREJECT", "#c62828", width=0.90, height=0.86, fontsize=6.4)
    arrow(ax, previous, blocked, dashed=True)
    ax.text(0.35, 0.55, "Shallow-first forks; reject invalid lineage before execution.",
            fontsize=6.6, color="#52616b")

    ax.plot([7.35, 7.35], [0.25, 5.10], color="#c8cdd1", linewidth=1.0)
    ax.text(7.65, 5.05, "(b) Auditable evidence funnel",
            fontsize=8.6, fontweight="bold", color="#1f2933")
    ax.text(10.85, 4.52, "Archive: 7,593 records / 8,243 traces / 582,188 actions",
            ha="center", fontsize=6.5, color="#52616b")

    funnel = [
        ("463 mutations applied", 5.70, "#d7ebf7"),
        ("120 failures triggered", 4.80, "#b9dce9"),
        ("99 recoveries marked", 3.90, "#8fc9d3"),
        ("29 retained at depth <= 3", 3.30, "#58aeb4"),
        ("12 strict ECV", 2.55, "#197b83"),
    ]
    center = 10.85
    y = 3.70
    for label, width, color in funnel:
        half = width / 2
        polygon = Polygon(
            [(center - half, y + 0.62), (center + half, y + 0.62),
             (center + half - 0.18, y), (center - half + 0.18, y)],
            closed=True, facecolor=color, edgecolor="#27666b", linewidth=0.9,
        )
        ax.add_patch(polygon)
        ax.text(center, y + 0.31, label, ha="center", va="center", fontsize=6.8,
                color="#ffffff" if color == "#197b83" else "#18373a",
                fontweight="bold" if color == "#197b83" else "normal")
        y -= 0.70
    ax.text(10.85, 0.18, "Each narrowing step is machine-auditable.",
            ha="center", fontsize=6.5, color="#52616b")

    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES / "scheduler-audit.pdf", bbox_inches="tight")
    plt.close(fig)


def build_timed_cascade():
    selected = [
        ("01_apple_visible_in_open_microwave.png", "1. visible"),
        ("03_first_pickup_rejected.png", "2. pickup fails"),
        ("05_first_recovered_apple_visible.png", "3. recover"),
        ("06_second_put_rejected.png", "4. placement fails"),
        ("08_final_put_succeeds.png", "5. finish"),
    ]
    images = [Image.open(CASCADE / name).convert("RGB") for name, _ in selected]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * len(images), height), "white")
    for index, image in enumerate(images):
        canvas.paste(image.resize((width, height)), (index * width, 0))

    fig, ax = plt.subplots(figsize=(7.15, 2.0))
    ax.imshow(canvas)
    ax.axis("off")
    for index, (_, label) in enumerate(selected):
        ax.text(index * width + width / 2, 13, label, ha="center", va="top", fontsize=9,
                color="white", bbox={"boxstyle": "round,pad=0.22", "facecolor": "#17212b", "alpha": 0.88, "edgecolor": "none"})
        if index < len(selected) - 1:
            ax.annotate("", xy=((index + 1) * width - 6, height * 0.52), xytext=(index * width + width - 40, height * 0.52),
                        arrowprops={"arrowstyle": "->", "color": "#eab308", "lw": 1.7})
    fig.tight_layout(pad=0)
    fig.savefig(FIGURES / "timed-cascade.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def build_timed_cascade_column():
    selected = [
        ("03_first_pickup_rejected.png", "1. pickup rejected", "#a83232"),
        ("05_first_recovered_apple_visible.png", "2. open microwave", "#176b52"),
        ("04_second_trap_added_while_first_triggered.png", "3. pickup succeeds", "#176b52"),
        ("06_second_put_rejected.png", "4. placement rejected", "#a83232"),
        ("07_second_recovered_fridge_open.png", "5. open fridge", "#176b52"),
        ("08_final_put_succeeds.png", "6. placement succeeds", "#176b52"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(3.38, 2.30))
    for ax, (name, label, color) in zip(axes.flat, selected):
        ax.imshow(Image.open(CASCADE / name).convert("RGB"))
        ax.axis("off")
        ax.text(
            0.5, 0.97, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color="white",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": color,
                  "alpha": 0.9, "edgecolor": "none"},
        )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.025, hspace=0.03)
    fig.savefig(FIGURES / "timed-cascade-column.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def build_pilot_rollout():
    selected = [
        ("main__s4.png", "1. pickup"),
        ("main__s5.png", "2. blocked"),
        ("main__s6.png", "3. recover"),
        ("main__s10.png", "4. place"),
        ("main__s11.png", "5. complete"),
    ]
    images = [Image.open(PILOT / name).convert("RGB") for name, _ in selected]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * len(images), height), "white")
    for index, image in enumerate(images):
        canvas.paste(image.resize((width, height)), (index * width, 0))

    fig, ax = plt.subplots(figsize=(7.15, 2.0))
    ax.imshow(canvas)
    ax.axis("off")
    for index, (_, label) in enumerate(selected):
        ax.text(index * width + width / 2, 13, label, ha="center", va="top", fontsize=9,
                color="white", bbox={"boxstyle": "round,pad=0.22", "facecolor": "#17212b", "alpha": 0.88, "edgecolor": "none"})
        if index < len(selected) - 1:
            ax.annotate("", xy=((index + 1) * width - 6, height * 0.52), xytext=(index * width + width - 40, height * 0.52),
                        arrowprops={"arrowstyle": "->", "color": "#eab308", "lw": 1.7})
    fig.tight_layout(pad=0)
    fig.savefig(FIGURES / "pilot-rollout.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_platform()
    build_scheduler_audit()
    build_timed_cascade()
    build_timed_cascade_column()
    if PILOT.exists():
        build_pilot_rollout()


if __name__ == "__main__":
    main()
