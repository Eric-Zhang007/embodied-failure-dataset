"""Build evidence figures for the EF-Bench paper from recorded local artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
CASCADE = ROOT / "output_oracle_timed_cascade_20260722"
PILOT = ROOT / "output_quality_eval_20260726" / "trial_T20190907_165608_878138"


def box(ax, xy, text, color, width=2.25, height=0.8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.4, edgecolor=color, facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=8.0, color="#1f2933", wrap=True)
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
    fig, ax = plt.subplots(figsize=(3.38, 4.25))
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    planner = box(ax, (0.45, 5.35), "Planner: intent + diagnosis", "#1976d2", width=3.05, height=0.58)
    executor = box(ax, (0.45, 4.18), "Executor: grounded actions", "#1976d2", width=3.05, height=0.58)
    thor = box(ax, (0.45, 3.01), "AI2-THOR: observed state transition", "#37474f", width=3.05, height=0.58)
    record = box(ax, (0.45, 1.84), "Episode record: images, actions, state", "#00897b", width=3.05, height=0.58)
    verifier = box(ax, (0.45, 0.52), "Evaluator-only:\nOracle validates a replayed alternative", "#7b1fa2", width=3.05, height=0.82)

    vertical_arrow(ax, planner, executor, "intent")
    vertical_arrow(ax, executor, thor, "action")
    vertical_arrow(ax, thor, record, "observation")
    vertical_arrow(ax, record, verifier, "failure context", dashed=True)
    ax.text(0.1, 0.12, "Solid: agent/environment. Dashed: evaluator-only.",
            fontsize=7.2, color="#52616b")
    fig.tight_layout(pad=0.25)
    fig.savefig(FIGURES / "platform.pdf", bbox_inches="tight")
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
    build_timed_cascade()
    if PILOT.exists():
        build_pilot_rollout()


if __name__ == "__main__":
    main()
