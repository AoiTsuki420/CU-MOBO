from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent.parent / "results"
OUT_DIR = HERE / "english"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _center(draw: ImageDraw.ImageDraw, xy, text: str, font, fill="black") -> None:
    x0, y0, x1, y1 = xy
    box = draw.textbbox((0, 0), text, font=font)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, font=font, fill=fill)


def _rotated_label(img: Image.Image, xy, text: str, font, fill="black") -> None:
    x0, y0, x1, y1 = xy
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    box = draw.textbbox((0, 0), text, font=font)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    text_layer = Image.new("RGBA", (text_w + 10, text_h + 10), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.text((5, 5), text, font=font, fill=fill)
    rotated = text_layer.rotate(90, expand=True)
    layer.alpha_composite(
        rotated,
        ((layer.width - rotated.width) // 2, (layer.height - rotated.height) // 2),
    )
    img.alpha_composite(layer, (x0, y0))


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int, rows) -> None:
    font = _font(15)
    row_h = 24
    width = max(draw.textbbox((0, 0), text, font=font)[2] for _, _, text in rows) + 72
    height = row_h * len(rows) + 16
    draw.rounded_rectangle((x, y, x + width, y + height), radius=3, fill="white", outline="#cccccc")
    for i, (color, style, text) in enumerate(rows):
        yy = y + 15 + i * row_h
        if style == "dashed":
            for xx in range(x + 12, x + 48, 9):
                draw.line((xx, yy + 5, xx + 5, yy + 5), fill=color, width=2)
        elif style == "marker":
            draw.line((x + 12, yy + 5, x + 48, yy + 5), fill=color, width=3)
            draw.ellipse((x + 27, yy, x + 37, yy + 10), fill=color)
        else:
            draw.line((x + 12, yy + 5, x + 48, yy + 5), fill=color, width=3)
        draw.text((x + 58, yy - 4), text, font=font, fill="black")


def overlay_convergence() -> None:
    img = Image.open(RESULTS_DIR / "oracle_gap_convergence.png").convert("RGBA")
    draw = ImageDraw.Draw(img)
    title = _font(19)
    label = _font(17)

    draw.rectangle((0, 0, img.width, 44), fill="white")
    _center(draw, (0, 0, img.width, 42), "PMO sample efficiency: true activity vs budget", title)

    draw.rectangle((0, 80, 46, img.height - 45), fill="white")
    _rotated_label(img, (0, 105, 44, img.height - 82), "top-K true activity by mu", label)

    draw.rectangle((245, img.height - 55, img.width - 165, img.height), fill="white")
    _center(draw, (230, img.height - 42, img.width - 150, img.height - 5), "True oracle evaluations (oracle budget)", label)

    img.convert("RGB").save(OUT_DIR / "oracle_gap_convergence.png")


def overlay_calibration() -> None:
    img = Image.open(RESULTS_DIR / "calibration.png").convert("RGBA")
    draw = ImageDraw.Draw(img)
    title = _font(19)
    subtitle = _font(18)
    label = _font(16)

    draw.rectangle((0, 0, img.width, 45), fill="white")
    _center(
        draw,
        (0, 0, img.width, 42),
        "CU-MOBO sigma calibration: OOF-calibrated sigma vs RF tree-variance sigma",
        title,
    )

    draw.rectangle((130, 54, 675, 88), fill="white")
    _center(draw, (120, 54, 690, 86), "Reliability diagram (sigma^2 as train_Yvar)", subtitle)
    draw.rectangle((795, 54, 1375, 88), fill="white")
    _center(draw, (780, 54, 1390, 86), "Error calibration curve (sigma objective; rho=Spearman)", subtitle)

    draw.rectangle((92, 98, 365, 174), fill="white")
    _legend(
        draw,
        98,
        104,
        [
            ("black", "dashed", "perfect calibration"),
            ("gray", "line", "tree-var (miscal=0.017)"),
            ("#ff7f0e", "line", "OOF-calibrated (miscal=0.021)"),
        ],
    )
    draw.rectangle((800, 98, 1138, 174), fill="white")
    _legend(
        draw,
        806,
        104,
        [
            ("black", "dashed", "perfect"),
            ("gray", "marker", "tree-var (ENCE=0.258, rho=0.21)"),
            ("#ff7f0e", "marker", "OOF-calibrated (ENCE=0.412, rho=0.18)"),
        ],
    )

    draw.rectangle((0, 185, 45, 400), fill="white")
    _rotated_label(img, (0, 195, 44, 395), "Observed coverage", label)
    draw.rectangle((712, 180, 750, 405), fill="white")
    _rotated_label(img, (712, 190, 748, 395), "Observed RMSE per bin", label)
    draw.line((706, 91, 706, 491), fill="black", width=2)

    draw.rectangle((220, 522, 600, img.height), fill="white")
    _center(draw, (220, 522, 600, img.height - 3), "Nominal coverage p", label)
    draw.rectangle((885, 522, 1285, img.height), fill="white")
    _center(
        draw,
        (855, 522, 1320, img.height - 3),
        "Predicted RMV = sqrt(mean sigma^2) per bin",
        label,
    )

    img.convert("RGB").save(OUT_DIR / "calibration.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_convergence()
    overlay_calibration()
    print(f"Wrote overlay English figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
