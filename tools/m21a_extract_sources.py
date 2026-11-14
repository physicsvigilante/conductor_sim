"""M21a source-evidence capture for selected JEE-style PDF anchors.

Run in an ordinary repository checkout. Do not run OCR; scanned pages are
reviewed as images and searchable pages retain extracted text.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import fitz

SOURCES = (
    ("02_conductors_half_size.pdf",
     "9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d",
     17, tuple(range(1, 18))),
    ("POTENTIAL_CONDUCTORS1.pdf",
     "8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5",
     71, (13,20,25,26,27,29,61,68,69,70)),
)


def extract(repo: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for filename, pin, expected_count, target_pages in SOURCES:
        path = repo / filename
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != pin:
            raise ValueError(f"{filename}: SHA-256 mismatch: {digest}")
        doc = fitz.open(str(path))
        if len(doc) != expected_count:
            raise ValueError(f"{filename}: page count changed: {len(doc)}")
        folder = out / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for one_based in target_pages:
            page = doc[one_based - 1]
            # Full source page, not cropped/selectively recontextualised.
            # Render about 120dpi so charge labels and spatial geometry remain legible.
            pix = page.get_pixmap(matrix=fitz.Matrix(120/72, 120/72),
                                  alpha=False)
            img = folder / f"page-{one_based:02d}.jpg"
            pix.save(str(img), jpg_quality=72)
            txt = page.get_text(sort=True)
            (folder / f"page-{one_based:02d}.txt").write_text(
                txt, encoding="utf-8")
            records.append(dict(pdf_page=one_based,
                                full_page_image=img.relative_to(out).as_posix(),
                                extracted_text=(folder / f"page-{one_based:02d}.txt").relative_to(out).as_posix(),
                                image_width=pix.width, image_height=pix.height,
                                page_text_length=len(txt)))
        meta.append(dict(filename=filename, sha256=digest,
                         pages=len(doc), size_bytes=path.stat().st_size,
                         reviewed_pages=records))
        doc.close()
    result=dict(schema="conductor_sim.m21a_source_evidence.v1",
                method="Original pinned full PDF page JPEG renders at 120 dpi, quality 72; PDF-native text where available; no OCR",
                sources=meta)
    (out / "source-evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("review-artifacts/m21a-source"))
    args = ap.parse_args()
    data = extract(args.repo, args.out)
    for s in data["sources"]:
        print(s["filename"], "sha", s["sha256"], "pages",
              s["pages"], "rendered", len(s["reviewed_pages"]))
        print("TEXT LENGTHS", [(p["pdf_page"], p["page_text_length"])
                                for p in s["reviewed_pages"]])


if __name__ == "__main__":
    main()
