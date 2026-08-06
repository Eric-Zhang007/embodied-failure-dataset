"""Composite real AI2-THOR observations into generated paper figure bases."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "output" / "imagegen"
FIGURES = ROOT / "paper" / "figures"
TIMED = ROOT / "output_oracle_timed_cascade_20260722"
DROP = ROOT / "output_oracle_drop_close_cascade_20260722"
FILL = ROOT / "output_oracle_fill_recovery_20260722"
COLLECTOR = ROOT / "output_collector_7lane_collisionguard_20260727"


def font_path(italic: bool = False, bold: bool = False) -> str:
    windows_name = "georgiaz.ttf" if italic and bold else "georgiai.ttf" if italic else "georgiab.ttf" if bold else "georgia.ttf"
    suffix = "-Bold" if bold else ""
    candidates = [
        Path("/mnt/c/Windows/Fonts") / windows_name,
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{suffix}.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("A readable serif font is required")


def font(size: int, *, italic: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(italic=italic, bold=bold), size=size)


def place_image(canvas: Image.Image, source: Path, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    with Image.open(source) as image:
        tile = ImageOps.fit(image.convert("RGB"), (x1 - x0, y1 - y0), method=Image.Resampling.LANCZOS)
    canvas.paste(tile, (x0, y0))
    ImageDraw.Draw(canvas).rectangle(box, outline="#54636d", width=2)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str = "#263238",
    spacing: int = 4,
) -> None:
    draw.multiline_text(center, text, font=text_font, fill=fill, anchor="mm", align="center", spacing=spacing)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#52636d",
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        head = [(ex, ey), (ex - 15 * direction, ey - 9), (ex - 15 * direction, ey + 9)]
    else:
        direction = 1 if ey > sy else -1
        head = [(ex, ey), (ex - 9, ey - 15 * direction), (ex + 9, ey - 15 * direction)]
    draw.polygon(head, fill=color)


def stage_frame(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    color: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline=color, width=3)
    draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y0 + 62), fill="#ffffff")
    draw.line((x0, y0 + 62, x1, y0 + 62), fill=color, width=3)
    centered_text(draw, ((x0 + x1) // 2, y0 + 31), title, font(30, italic=True, bold=True), color)


def build_teaser() -> None:
    canvas = Image.open(GENERATED / "efbench-teaser-base-final.png").convert("RGB")
    canvas = canvas.resize((2048, 1152), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    xs = [(33, 371), (389, 700), (720, 1029), (1051, 1360)]
    ys = [(34, 297), (322, 579), (605, 850), (874, 1117)]
    rows = [
        [
            TIMED / "00_initial.png",
            FILL / "01_empty_bowl.png",
            COLLECTOR / "trial_T20190909_044958_536304" / "main__s3.png",
            COLLECTOR / "trial_T20190907_215326_695860" / "main__s3.png",
        ],
        [
            TIMED / "02_first_trap_microwave_closed.png",
            TIMED / "03_first_pickup_rejected.png",
            FILL / "02_bowl_filled_by_injection.png",
            DROP / "02_first_trap_apple_dropped.png",
        ],
        [
            TIMED / "05_first_recovered_apple_visible.png",
            DROP / "05_first_recovery_apple_held.png",
            FILL / "04_bowl_emptied.png",
            TIMED / "07_second_recovered_fridge_open.png",
        ],
        [
            TIMED / "01_apple_visible_in_open_microwave.png",
            TIMED / "05_first_recovered_apple_visible.png",
            TIMED / "07_second_recovered_fridge_open.png",
            TIMED / "08_final_put_succeeds.png",
        ],
    ]
    colors = ["#238b8e", "#dc5a4b", "#238b8e", "#7450a6"]
    labels = ["observe", "failure", "recover", "verify"]
    for row_index, sources in enumerate(rows):
        for column_index, source in enumerate(sources):
            x0, x1 = xs[column_index]
            y0, y1 = ys[row_index]
            place_image(canvas, source, (x0, y0, x1, y1))
            draw.rectangle((x0, y0, x1, y1), outline=colors[row_index], width=3)
            if column_index == 0:
                draw.rounded_rectangle((x0 + 10, y0 + 10, x0 + 118, y0 + 45), radius=8, fill="#ffffff")
                centered_text(
                    draw,
                    (x0 + 64, y0 + 27),
                    labels[row_index],
                    font(20, italic=True, bold=True),
                    colors[row_index],
                )

    bubble_specs = [
        ((1790, 111), "OBSERVE", '"The apple is visible.\nI will pick it up."', "#238b8e", 25),
        ((1810, 330), "FAILURE", '"Pickup failed: the\nmicrowave is closed."', "#dc5a4b", 24),
        ((1827, 534), "COUNTERFACTUAL", '"Replay, open first,\nthen retry pickup."', "#7450a6", 22),
    ]
    for (cx, cy), heading, body, color, body_size in bubble_specs:
        centered_text(draw, (cx, cy - 24), heading, font(22, italic=True, bold=True), color)
        centered_text(draw, (cx, cy + 23), body, font(body_size, italic=True), "#263238", spacing=6)

    FIGURES.mkdir(parents=True, exist_ok=True)
    canvas.save(FIGURES / "narrative-teaser.png", optimize=True)


def build_pipeline() -> None:
    canvas = Image.new("RGB", (2400, 1280), "#fbfcfd")
    draw = ImageDraw.Draw(canvas)
    navy, teal, red, purple, orange = "#38556b", "#277f7a", "#c95b4d", "#715ca6", "#d28c32"

    input_box = (35, 35, 360, 735)
    agent_box = (410, 35, 850, 735)
    failure_box = (900, 35, 1300, 735)
    event_box = (1350, 35, 1740, 735)
    replay_box = (1790, 35, 2365, 735)
    stage_frame(draw, input_box, "Task / Scene / Seed", navy)
    stage_frame(draw, agent_box, "Agent Stack", navy)
    stage_frame(draw, failure_box, "Failure Programmer", red)
    stage_frame(draw, event_box, "Event Log + Lineage", purple)
    stage_frame(draw, replay_box, "Replay + Verify", teal)

    inputs = [
        (COLLECTOR / "trial_T20190909_044958_536304" / "main__s3.png", "task goal"),
        (COLLECTOR / "trial_T20190907_215326_695860" / "main__s3.png", "scene"),
        (COLLECTOR / "trial_T20190907_174127_043461" / "main__s3.png", "seed"),
    ]
    for index, (source, label) in enumerate(inputs):
        y0 = 105 + index * 198
        place_image(canvas, source, (62, y0, 235, y0 + 156))
        draw.rounded_rectangle((244, y0 + 46, 344, y0 + 110), radius=8, fill="#ffffff", outline=navy, width=2)
        centered_text(draw, (294, y0 + 78), label, font(20, italic=True), navy)
        arrow(draw, (235, y0 + 78), (244, y0 + 78), navy, 3)

    modules = [
        ("Planner", False), ("Executor", False), ("Searched markers", True),
        ("Intent guard", True), ("Critic guard", True), ("Curiosity score", True),
        ("Contrastive proposals", True), ("Progress gate", True), ("Revisit-cost trail", True),
    ]
    for index, (label, switchable) in enumerate(modules):
        y0 = 91 + index * 66
        color = teal if switchable else navy
        draw.rounded_rectangle((445, y0, 815, y0 + 48), radius=7, fill="#ffffff", outline=color, width=2)
        draw.rectangle((445, y0, 454, y0 + 48), fill=color)
        centered_text(draw, (630, y0 + 24), label, font(24, italic=True), "#26343d")
    centered_text(draw, (630, 700), "seven independently switchable controls", font(21, italic=True), teal)

    place_image(canvas, TIMED / "01_apple_visible_in_open_microwave.png", (930, 105, 1270, 350))
    centered_text(draw, (1100, 385), "close Microwave\nbefore PickupObject", font(20, italic=True, bold=True), red, spacing=1)
    arrow(draw, (1100, 411), (1100, 420), red, 4)
    place_image(canvas, TIMED / "02_first_trap_microwave_closed.png", (930, 425, 1270, 670))
    centered_text(draw, (1100, 700), "environment-confirmed state change", font(21, italic=True), red)

    events = [
        ("observe", "01_apple_visible_in_open_microwave.png"),
        ("apply", "02_first_trap_microwave_closed.png"),
        ("trigger", "03_first_pickup_rejected.png"),
        ("recover", "05_first_recovered_apple_visible.png"),
        ("replan", "04_second_trap_added_while_first_triggered.png"),
        ("retry", "06_second_put_rejected.png"),
        ("complete", "08_final_put_succeeds.png"),
    ]
    for index, (label, name) in enumerate(events):
        y0 = 88 + index * 88
        place_image(canvas, TIMED / name, (1378, y0, 1455, y0 + 70))
        draw.rounded_rectangle((1470, y0, 1708, y0 + 70), radius=7, fill="#ffffff", outline=purple, width=2)
        centered_text(draw, (1589, y0 + 35), label, font(24, italic=True), "#26343d")
        if index < len(events) - 1:
            arrow(draw, (1416, y0 + 71), (1416, y0 + 86), purple, 3)

    branch_sets = [
        ("failed", red, ["02_first_trap_microwave_closed.png", "03_first_pickup_rejected.png", "06_second_put_rejected.png"], "x"),
        ("partial", orange, ["00_initial.png", "05_first_recovered_apple_visible.png", "06_second_put_rejected.png"], "o"),
        ("verified", teal, ["05_first_recovered_apple_visible.png", "07_second_recovered_fridge_open.png", "08_final_put_succeeds.png"], "check"),
    ]
    for index, (label, color, names, verdict) in enumerate(branch_sets):
        x0 = 1818 + index * 176
        centered_text(draw, (x0 + 75, 105), label, font(25, italic=True, bold=True), color)
        for row, name in enumerate(names):
            y0 = 135 + row * 145
            place_image(canvas, TIMED / name, (x0, y0, x0 + 150, y0 + 128))
        symbol_y = 620
        draw.rounded_rectangle((x0 + 41, symbol_y, x0 + 109, symbol_y + 62), radius=8, fill="#ffffff", outline=color, width=3)
        if verdict == "x":
            draw.line((x0 + 59, symbol_y + 18, x0 + 91, symbol_y + 44), fill=color, width=5)
            draw.line((x0 + 91, symbol_y + 18, x0 + 59, symbol_y + 44), fill=color, width=5)
        elif verdict == "o":
            draw.ellipse((x0 + 60, symbol_y + 16, x0 + 90, symbol_y + 46), outline=color, width=5)
        else:
            draw.line((x0 + 58, symbol_y + 32, x0 + 70, symbol_y + 44), fill=color, width=5)
            draw.line((x0 + 70, symbol_y + 44, x0 + 94, symbol_y + 17), fill=color, width=5)
    centered_text(draw, (2075, 704), "same prefix, original task predicate", font(22, italic=True), teal)

    for left, right in [(360, 410), (850, 900), (1300, 1350), (1740, 1790)]:
        arrow(draw, (left, 385), (right, 385), navy, 4)

    draw.rounded_rectangle((35, 790, 2365, 1235), radius=14, fill="#ffffff", outline="#8da0ad", width=3)
    centered_text(draw, (1200, 830), "Bounded lineage and replay schedule", font(30, italic=True, bold=True), navy)
    prefix = [
        "00_initial.png", "01_apple_visible_in_open_microwave.png",
        "02_first_trap_microwave_closed.png", "03_first_pickup_rejected.png",
        "05_first_recovered_apple_visible.png", "06_second_put_rejected.png",
        "07_second_recovered_fridge_open.png", "08_final_put_succeeds.png",
    ]
    for index, name in enumerate(prefix):
        x0 = 70 + index * 188
        place_image(canvas, TIMED / name, (x0, 875, x0 + 170, 1030))
        centered_text(draw, (x0 + 85, 1060), str(index), font(20, italic=True), "#52636d")
        if index < len(prefix) - 1:
            arrow(draw, (x0 + 170, 952), (x0 + 184, 952), navy, 3)
    nodes = [(1635, "MAIN", navy), (1810, "FORK d=1", purple), (1990, "FORK d=2", purple), (2170, "FORK d=3", purple)]
    for index, (x0, label, color) in enumerate(nodes):
        draw.rounded_rectangle((x0, 900, x0 + 145, 982), radius=9, fill="#ffffff", outline=color, width=3)
        centered_text(draw, (x0 + 72, 941), label, font(22, italic=True, bold=True), color)
        if index < len(nodes) - 1:
            arrow(draw, (x0 + 145, 941), (nodes[index + 1][0] - 8, 941), color, 3)
    centered_text(draw, (2000, 1065), "1:2 main/fork rotation  |  shallow-first  |  maximum depth 3", font(23, italic=True), "#52636d")
    centered_text(draw, (1200, 1170), "exact object IDs  /  action results  /  confirmed mutations  /  parent-fork links", font(25, italic=True), "#26343d")

    FIGURES.mkdir(parents=True, exist_ok=True)
    canvas.save(FIGURES / "generated-pipeline.png", optimize=True)


def main() -> None:
    build_teaser()
    build_pipeline()


if __name__ == "__main__":
    main()
