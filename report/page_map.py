"""Print the page each paragraph starts on, to locate whitespace gaps."""

import sys
from pathlib import Path

import win32com.client as win32

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = Path(sys.argv[1] if len(sys.argv) > 1 else "Project_Report_new.docx").resolve()
word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(str(path), ReadOnly=True, AddToRecentFiles=False)
try:
    doc.Repaginate()
    total = doc.ComputeStatistics(2)
    last = 0
    for para in doc.Paragraphs:
        rng = para.Range
        page = rng.Information(3)  # wdActiveEndPageNumber
        text = rng.Text.replace("\r", " ").replace("\x07", " ").strip()
        if not text:
            continue
        marker = "*" if para.Range.InlineShapes.Count else " "
        if page != last:
            print(f"--- page {page} ---")
            last = page
        print(f"{marker} p{page} {text[:95]}")
    print(f"TOTAL {total} pages")
finally:
    doc.Close(SaveChanges=0)
    word.Quit()
