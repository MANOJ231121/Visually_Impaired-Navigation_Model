"""Export a DOCX to PDF with Word."""

import sys
from pathlib import Path

import win32com.client as win32

src = Path(sys.argv[1] if len(sys.argv) > 1 else "Project_Report_new.docx").resolve()
pdf = src.with_suffix(".pdf")

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(str(src), ReadOnly=True, AddToRecentFiles=False)
try:
    doc.ExportAsFixedFormat(str(pdf), 17)  # wdExportFormatPDF
finally:
    doc.Close(SaveChanges=0)
    word.Quit()

print(f"exported {pdf}")
