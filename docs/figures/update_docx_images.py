from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


DEFAULT_MAPPING = {
    "word/media/rId12.png": "main_figure-1.png",
    "word/media/rId20.png": "oracle_gap.png",
    "word/media/rId24.png": "oracle_gap_convergence.png",
    "word/media/rId28.png": "retrospective_hardened.png",
    "word/media/rId32.png": "calibration.png",
    "word/media/rId36.png": "diversity.png",
    "word/media/rId40.png": "tdc_anchor.png",
    "word/media/rId45.png": "overoptimization.png",
}


def replace_docx_images(docx: Path, image_dir: Path, backup: Path | None) -> None:
    docx = docx.resolve()
    image_dir = image_dir.resolve()
    if backup and not backup.exists():
        shutil.copy2(docx, backup)

    tmp = docx.with_suffix(".tmp_600dpi.docx")
    replaced = set()
    with zipfile.ZipFile(docx, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in DEFAULT_MAPPING:
                source = image_dir / DEFAULT_MAPPING[item.filename]
                data = source.read_bytes()
                replaced.add(item.filename)
            zout.writestr(item, data)

    missing = set(DEFAULT_MAPPING) - replaced
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Missing media entries in docx: {sorted(missing)}")

    shutil.move(str(tmp), str(docx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, default=Path("paper_jcim_clean.docx"))
    parser.add_argument("--image-dir", type=Path, default=Path("figures/english_600dpi"))
    parser.add_argument("--backup", type=Path, default=Path("paper_jcim_clean.before_600dpi.docx"))
    args = parser.parse_args()
    replace_docx_images(args.docx, args.image_dir, args.backup)
    print(f"Updated {args.docx} with images from {args.image_dir}")


if __name__ == "__main__":
    main()
