"""Build a styled pandoc reference document (A4, Times New Roman, page numbers)."""

import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
REF = HERE / "reference.docx"

subprocess.run(
    ["pandoc", "-o", str(REF), "--print-default-data-file", "reference.docx"],
    check=True,
    stdout=subprocess.DEVNULL,
)

doc = Document(str(REF))

# ---- page setup -----------------------------------------------------------
for section in doc.sections:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)


def set_font(style, name="Times New Roman", size=None, bold=None, italic=None):
    font = style.font
    font.name = name
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if italic is not None:
        font.italic = italic
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def set_spacing(style, before=0, after=4, line=1.15):
    """Line spacing is intentionally *single* (multiple 1.0), never "exactly".

    Any exact line spacing makes Word clip an inserted picture to the line height,
    so the image overlaps the text and can even be pushed off the page.
    """
    pf = style.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line


styles = doc.styles
for name, size, bold in (
    ("Normal", 11, False),
    ("Body Text", 11, False),
    ("First Paragraph", 11, False),
    ("Compact", 9.5, False),
    ("Block Text", 10, False),
):
    if name in [s.name for s in styles]:
        st = styles[name]
        set_font(st, size=size, bold=bold)
        # "Compact" wraps every table cell and list item: no space after keeps
        # long tables (Appendix A) from growing a third of a page taller.
        set_spacing(st, after=0 if name == "Compact" else 2, line=1.0)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size, before, after in (
    ("Heading 1", 14, 8, 2),
    ("Heading 2", 12, 5, 1),
    ("Heading 3", 11, 6, 2),
):
    if name in [s.name for s in styles]:
        st = styles[name]
        set_font(st, size=size, bold=True, italic=False)
        set_spacing(st, before=before, after=after, line=1.0)
        st.paragraph_format.keep_with_next = True

for name, size, bold, italic in (
    ("Title", 22, True, False),
    ("Subtitle", 12, False, True),
    ("Author", 12, False, False),
    ("Date", 11, False, False),
    ("Image Caption", 9, False, True),
    ("Table Caption", 9, False, True),
    ("Table", 9, False, False),
    ("Caption", 9, False, True),
):
    if name in [s.name for s in styles]:
        st = styles[name]
        set_font(st, size=size, bold=bold, italic=italic)
        set_spacing(st, before=2, after=3, line=1.0)

if "Table" in [s.name for s in styles]:
    styles["Table"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

# ---- footer with page number ---------------------------------------------
footer = doc.sections[0].footer
para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
para.alignment = WD_ALIGN_PARAGRAPH.CENTER

run = para.add_run()
for kind, text in (("begin", None), ("instr", " PAGE "), ("end", None)):
    if kind == "instr":
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = text
        run._r.append(instr)
    else:
        fld = OxmlElement("w:fldChar")
        fld.set(qn("w:fldCharType"), kind)
        run._r.append(fld)

doc.save(str(REF))
print(f"reference written: {REF}")
