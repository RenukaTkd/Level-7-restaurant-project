"""Build docs/FINAL_REPORT.docx from docs/FINAL_REPORT.md.

    pip install python-docx
    python scripts/make_report_docx.py            # defaults below
    python scripts/make_report_docx.py IN.md OUT.docx
"""
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm, Twips

SRC, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("docs/FINAL_REPORT.md", "docs/FINAL_REPORT.docx")
BLUE = RGBColor(0x1C, 0x5C, 0xAB)
DARK = RGBColor(0x1F, 0x29, 0x37)
GREY = RGBColor(0x55, 0x55, 0x55)
FONT = "Arial"
TEXT_WIDTH = 9026  # A4 minus 2 x 1in margins, in twips

doc = Document()

# ---- page setup: A4, 1in margins ---------------------------------------------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Twips(11906), Twips(16838)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Twips(1440))

# ---- styles ----------------------------------------------------------------------------
st = doc.styles
normal = st["Normal"]
normal.font.name, normal.font.size = FONT, Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15
for lvl, size, before, after, color in [(1, 18, 18, 8, BLUE), (2, 14, 14, 6, BLUE), (3, 12, 10, 4, DARK)]:
    h = st[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = FONT, Pt(size), True, color
    rpr = h.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    pf = h.paragraph_format
    pf.space_before, pf.space_after, pf.keep_with_next = Pt(before), Pt(after), True


def shade(el, fill):
    pr = el.get_or_add_tcPr() if el.tag.endswith("tc") else el.get_or_add_pPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill)
    pr.append(s)


def para_border(p, side="left", color="1C5CAB", sz=18, space=8):
    ppr = p._p.get_or_add_pPr()
    b = ppr.find(qn("w:pBdr"))
    if b is None:
        b = OxmlElement("w:pBdr"); ppr.append(b)
    e = OxmlElement(f"w:{side}")
    for k, v in (("val", "single"), ("sz", str(sz)), ("space", str(space)), ("color", color)):
        e.set(qn(f"w:{k}"), v)
    b.append(e)


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_inline(p, text, bold=False, italic=False, size=None, color=None):
    for part in INLINE.split(text):
        if not part:
            continue
        b, i, code = bold, italic, False
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, code = part[1:-1], True
        elif part.startswith("[") and "](" in part:
            part = part[1:part.index("](")]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, i = part[1:-1], True
        r = p.add_run(part)
        r.bold, r.italic = b, i
        if code:
            r.font.name = "Consolas"; r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(0xB4, 0x23, 0x18)
        if size:
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color
    return p


def add_field(p, instr):
    r = p.add_run()
    for kind, txt in (("begin", None), (None, instr), ("separate", None), (None, "Right-click → Update Field"), ("end", None)):
        if kind:
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), kind); r._r.append(f)
        elif txt == instr:
            t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = instr; r._r.append(t)
        else:
            t = OxmlElement("w:t"); t.text = txt; r._r.append(t)


def table(rows):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    clean = lambda x: re.sub(r"[*`]", "", x)
    lens = [max(len(clean(r[c])) if c < len(r) else 0 for r in rows) for c in range(ncol)]
    words = [max((len(w) for r in rows if c < len(r) for w in clean(r[c]).split()), default=4) for c in range(ncol)]
    lens = [min(max(l, 4), 60) for l in lens]
    lens = [max(l, w * 1.35 + 2) for l, w in zip(lens, words)]   # never split a word
    widths = [int(TEXT_WIDTH * l / sum(lens)) for l in lens]
    widths[-1] += TEXT_WIDTH - sum(widths)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "C9CED6")
        borders.append(e)
    tblPr.append(borders)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(w))
    for ri, row in enumerate(rows):
        for ci in range(ncol):
            cell = t.cell(ri, ci)
            cell.width = Twips(widths[ci])
            txt = row[ci] if ci < len(row) else ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            add_inline(p, txt, bold=(ri == 0), size=9.5, color=(RGBColor(0xFF, 0xFF, 0xFF) if ri == 0 else None))
            if ri == 0:
                shade(cell._tc, "1C5CAB")
            elif ri % 2 == 0:
                shade(cell._tc, "F3F6FA")
    hdr = t.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); hdr.append(h)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ---- parse markdown -------------------------------------------------------------------
lines = open(SRC, encoding="utf-8").read().splitlines()
i = 0
title = lines[0].lstrip("# ").strip()
subtitle = lines[2].lstrip("# ").strip() if lines[2].startswith("###") else ""
i = 3

# title page
for _ in range(6):
    doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
add_inline(p, title, bold=True, size=30, color=BLUE)
para_border(p, "bottom", "1C5CAB", 12, 6)
p = doc.add_paragraph(); add_inline(p, subtitle, size=16, color=GREY)
doc.add_paragraph()
# header info table (first markdown table)
while not lines[i].startswith("|"):
    i += 1
info = []
while i < len(lines) and lines[i].startswith("|"):
    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    if not set("".join(cells)) <= set("-: ") and any(cells):
        info.append(cells)
    i += 1
for k, v in info:
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(4)
    add_inline(p, re.sub(r"\*", "", k) + ":  ", bold=True, size=11, color=DARK)
    add_inline(p, v, size=11)
p = doc.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE)

# table of contents (Word field, updated on open)
doc.add_paragraph("Contents", style="Heading 1")
add_field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
p = doc.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE)

# skip the markdown "Contents" section
while i < len(lines) and not lines[i].startswith("## 1."):
    i += 1

first_h1 = True
while i < len(lines):
    ln = lines[i]
    s = ln.strip()
    if s == "---" or s == "":
        i += 1; continue
    if s.startswith("```"):
        block = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            block.append(lines[i]); i += 1
        i += 1
        for bl in block:
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_after, pf.space_before, pf.line_spacing = Pt(0), Pt(0), 1.0
            pf.left_indent = Cm(0.3)
            r = p.add_run(bl if bl else " ")
            r.font.name, r.font.size = "Consolas", Pt(8.5)
            r._r.get_or_add_rPr().append(OxmlElement("w:noProof"))
            shade(p._p, "F3F4F6")
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        continue
    if s.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
            i += 1
        table(rows); continue
    if s.startswith("#"):
        level = len(s) - len(s.lstrip("#"))
        text = s.lstrip("#").strip()
        if level == 2:
            if not first_h1:
                p = doc.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE)
                p.paragraph_format.space_after = Pt(0)
            first_h1 = False
            doc.add_paragraph(text, style="Heading 1")
        else:
            doc.add_paragraph(text, style="Heading 2" if level == 3 else "Heading 3")
        i += 1; continue
    if s.startswith(">"):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        para_border(p, "left")
        add_inline(p, s.lstrip("> ").strip(), italic=True, color=DARK)
        i += 1; continue
    m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
    if m:
        indent = len(m.group(1)) // 2
        marker, text = m.group(2), m.group(3)
        # join wrapped continuation lines
        i += 1
        while i < len(lines) and lines[i].startswith(" " * (len(m.group(1)) + 2)) and lines[i].strip() \
                and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i]):
            text += " " + lines[i].strip(); i += 1
        if marker in "-*":
            p = doc.add_paragraph(style="List Bullet" if indent == 0 else "List Bullet 2")
        else:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75 + 0.75 * indent)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            p.add_run(marker + " ")
        p.paragraph_format.space_after = Pt(3)
        add_inline(p, text)
        continue
    # normal paragraph: join wrapped lines
    text = s
    i += 1
    while i < len(lines) and lines[i].strip() and not re.match(r"^\s*(#|\||```|>|---|[-*]\s|\d+\.\s)", lines[i]):
        text += " " + lines[i].strip(); i += 1
    p = doc.add_paragraph()
    italic = text.startswith("*") and text.endswith("*") and not text.startswith("**")
    add_inline(p, text[1:-1] if italic else text, italic=italic, color=GREY if italic else None)

# ---- header/footer with page numbers ---------------------------------------------------
sec.different_first_page_header_footer = True
hp = sec.header.paragraphs[0]
add_inline(hp, "AI Restaurant Sales Forecasting · Final report", size=8.5, color=GREY)
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Page "); r.font.size = Pt(9); r.font.color.rgb = GREY
add_field(fp, "PAGE")
for r in fp.runs:
    r.font.size = Pt(9); r.font.color.rgb = GREY

# ask Word to refresh the TOC/page fields when the file is opened
settings = doc.settings.element
u = OxmlElement("w:updateFields"); u.set(qn("w:val"), "true"); settings.append(u)

doc.core_properties.title = title
doc.core_properties.author = "Renuka"
doc.save(OUT)
print("saved", OUT)
