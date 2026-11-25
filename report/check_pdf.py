"""Check a PDF: page count, and whether every embedded image sits inside the margins."""

import sys
from pathlib import Path

import pymupdf

path = Path(sys.argv[1] if len(sys.argv) > 1 else "Project_Report_new.pdf").resolve()
doc = pymupdf.open(str(path))
print(f"{path.name}: {doc.page_count} pages")
for i, page in enumerate(doc, 1):
    w, h = page.rect.width, page.rect.height
    images = page.get_image_info()
    text_len = len(page.get_text().strip())
    print(f"  p{i}: {text_len:4d} chars, {len(images)} image(s)")
    for img in images:
        x0, y0, x1, y1 = img["bbox"]
        bad = not (x0 >= 40 and y0 >= 40 and x1 <= w - 40 and y1 <= h - 40)
        note = "   <-- OUT OF BOUNDS" if bad else ""
        print(f"      {x1 - x0:6.1f} x {y1 - y0:6.1f} pt at ({x0:5.1f}, {y0:5.1f}){note}")
