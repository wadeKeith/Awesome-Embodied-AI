#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Assets" / "cover.png"
OUTPUT = ROOT / "Assets" / "social-preview.png"

WIDTH = 1280
HEIGHT = 640


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts") / name,
        Path("/System/Library/Fonts/Supplemental") / name,
        Path("/Library/Fonts") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font("Avenir Next Condensed.ttc", 84)
SUBTITLE_FONT = load_font("NewYork.ttf", 28)
BODY_FONT = load_font("SFNS.ttf", 22)
MONO_FONT = load_font("SFNSMono.ttf", 18)
PILL_FONT = load_font("SFNSMono.ttf", 18)
CARD_TITLE_FONT = load_font("Avenir Next.ttc", 30)
CARD_BODY_FONT = load_font("SFNS.ttf", 19)


def crop_to_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    dst_ratio = width / height

    if src_ratio > dst_ratio:
        crop_w = int(src_h * dst_ratio)
        left = (src_w - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, src_h))
    else:
        crop_h = int(src_w / dst_ratio)
        top = (src_h - crop_h) // 2
        image = image.crop((0, top, src_w, top + crop_h))

    return image.resize((width, height), Image.Resampling.LANCZOS)


def add_multistop_overlay(base: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(8, 14, 22, 96))
    draw.ellipse((760, -40, 1270, 470), fill=(67, 196, 255, 62))
    draw.ellipse((-120, 320, 520, 820), fill=(255, 118, 79, 52))
    draw.rectangle((0, 0, 710, HEIGHT), fill=(6, 12, 24, 152))
    return Image.alpha_composite(base, overlay)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    x: int,
    y: int,
    max_width: int,
    line_gap: int,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    cursor_y = y
    for line in lines:
        draw.text((x, cursor_y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, cursor_y), line, font=font)
        cursor_y = bbox[3] + line_gap
    return cursor_y


def draw_pill(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: int,
    y: int,
    *,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    text_fill: tuple[int, int, int, int],
) -> int:
    bbox = draw.textbbox((0, 0), label, font=PILL_FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    width = text_w + 26
    height = text_h + 14
    draw.rounded_rectangle((x, y, x + width, y + height), radius=height // 2, fill=fill, outline=outline, width=1)
    draw.text((x + 13, y + 7), label, font=PILL_FONT, fill=text_fill)
    return width


def add_panel(base: Image.Image) -> None:
    draw = ImageDraw.Draw(base)

    panel_box = (56, 50, 690, 586)
    panel = Image.new("RGBA", base.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle(panel_box, radius=32, fill=(7, 16, 28, 182), outline=(116, 219, 255, 92), width=2)
    panel_draw.line((88, 118, 246, 118), fill=(121, 221, 255, 255), width=4)
    panel_draw.line((88, 124, 196, 124), fill=(255, 145, 103, 255), width=2)
    base.alpha_composite(panel)

    draw = ImageDraw.Draw(base)
    draw.text((88, 144), "AWESOME", font=TITLE_FONT, fill=(237, 247, 255, 255))
    draw.text((88, 228), "EMBODIED AI", font=TITLE_FONT, fill=(237, 247, 255, 255))

    subtitle = "A curated map of embodied AI research, benchmarks, simulators, humanoids, VLA models, and safety resources."
    end_y = draw_wrapped_text(
        draw,
        subtitle,
        SUBTITLE_FONT,
        (203, 223, 239, 255),
        90,
        334,
        520,
        8,
    )

    labels = [
        "surveys",
        "vla models",
        "datasets",
        "simulators",
        "humanoids",
        "safety",
    ]

    cursor_x = 90
    cursor_y = end_y + 26
    for label in labels:
        pill_width = draw_pill(
            draw,
            label,
            cursor_x,
            cursor_y,
            fill=(18, 38, 64, 204),
            outline=(98, 182, 217, 128),
            text_fill=(229, 243, 255, 255),
        )
        cursor_x += pill_width + 10
        if cursor_x > 560:
            cursor_x = 90
            cursor_y += 48

    footer_y = 520
    draw.text((90, footer_y), "github.com/wadeKeith/Awesome-Embodied-AI", font=MONO_FONT, fill=(140, 231, 255, 255))
    draw.text((90, footer_y + 34), "Community-maintained  •  Practical links  •  Updated and verified", font=BODY_FONT, fill=(224, 233, 241, 232))


def add_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, accent: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=26, fill=(8, 18, 30, 176), outline=(255, 255, 255, 38), width=1)
    draw.rounded_rectangle((x1 + 18, y1 + 20, x1 + 32, y1 + 68), radius=7, fill=accent)
    draw.text((x1 + 52, y1 + 20), title, font=CARD_TITLE_FONT, fill=(241, 248, 255, 255))
    draw_wrapped_text(draw, body, CARD_BODY_FONT, (208, 224, 238, 235), x1 + 52, y1 + 60, x2 - x1 - 78, 6)


def add_right_cards(base: Image.Image) -> None:
    draw = ImageDraw.Draw(base)
    add_card(
        draw,
        (760, 82, 1188, 198),
        "VLA Models",
        "OpenVLA, Octo, RT-2, pi0, diffusion policies, and robot reasoning systems.",
        (103, 214, 255, 255),
    )
    add_card(
        draw,
        (800, 230, 1220, 346),
        "Datasets",
        "Open X-Embodiment, DROID, VLABench, RoboMIND, ALOHA, and dexterous data.",
        (255, 154, 112, 255),
    )
    add_card(
        draw,
        (760, 378, 1188, 494),
        "Simulators + Safety",
        "Habitat, ManiSkill, OmniGibson, Genesis, embodied safety benchmarks, and attack-defense surveys.",
        (142, 255, 204, 255),
    )


def apply_finish(base: Image.Image) -> Image.Image:
    sharpened = base.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    vignette = Image.new("L", base.size, 0)
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.ellipse((-120, -80, WIDTH + 120, HEIGHT + 160), fill=220)
    vignette = ImageChops.invert(vignette).filter(ImageFilter.GaussianBlur(80))
    mask = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask.putalpha(vignette)
    return Image.alpha_composite(sharpened, mask)


def main() -> None:
    background = Image.open(INPUT).convert("RGBA")
    background = crop_to_canvas(background, WIDTH, HEIGHT)
    background = background.filter(ImageFilter.GaussianBlur(2.1))
    background = add_multistop_overlay(background)
    add_panel(background)
    add_right_cards(background)
    final = apply_finish(background)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUTPUT, optimize=True)


if __name__ == "__main__":
    main()
