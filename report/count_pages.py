"""Open a DOCX in Word, report page and word counts, and close it again.

Read-only: the document is never saved, so any open copy is left untouched.
"""

import sys
from pathlib import Path

import win32com.client as win32

path = Path(sys.argv[1] if len(sys.argv) > 1 else "Project_Report.docx").resolve()

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(str(path), ReadOnly=True, AddToRecentFiles=False)
try:
    doc.Repaginate()
    pages = doc.ComputeStatistics(2)  # wdStatisticPages
    words = doc.ComputeStatistics(0)  # wdStatisticWords
    chars = doc.ComputeStatistics(3)
    print(f"{path.name}: {pages} pages, {words} words, {chars} characters")
finally:
    doc.Close(SaveChanges=0)
    word.Quit()
