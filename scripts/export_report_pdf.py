"""Render PHASE2_REPORT.md to a clean, properly-formatted PDF using reportlab.
Persian/Farsi substrings (used for a few original-spec section citations) are
reshaped + bidi-reordered and rendered in Apple's GeezaPro system font (font 0 of
the .ttc, which has full Arabic-presentation-form glyph coverage unlike SF Arabic),
since reportlab's default fonts have no Arabic-script glyphs and cannot shape RTL
text on their own.

Usage:  python3 scripts/export_report_pdf.py PHASE2_REPORT.md PHASE2_REPORT.pdf

Requires: pip install reportlab markdown beautifulsoup4 arabic-reshaper python-bidi
macOS-specific: assumes /System/Library/Fonts/GeezaPro.ttc exists (ships with every
macOS install). On another OS, swap in any TTF/TTC with full Arabic Presentation
Forms-A/B coverage (U+FB50-FDFF, U+FE70-FEFF), e.g. Noto Sans Arabic.
"""
import re
import sys

import arabic_reshaper
import markdown
from bidi.algorithm import get_display
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SRC = sys.argv[1]
OUT = sys.argv[2]

pdfmetrics.registerFont(TTFont("SFArabic", "/System/Library/Fonts/GeezaPro.ttc", subfontIndex=0))

with open(SRC, encoding="utf-8") as f:
    md_text = f.read()

# --- Shape Persian/Arabic runs and wrap them in raw <font> HTML before markdown parsing ---
PERSIAN_RUN_RE = re.compile(r"[؀-ۿ‌][؀-ۿ‌ ]*[؀-ۿ‌]"
                             r"|[؀-ۿ‌]")


def _shape(match: re.Match) -> str:
    raw = match.group(0)
    shaped = get_display(arabic_reshaper.reshape(raw))
    return f'<font face="SFArabic">{shaped}</font>'


md_text = PERSIAN_RUN_RE.sub(_shape, md_text)

html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
soup = BeautifulSoup(html, "html.parser")

styles = getSampleStyleSheet()
BODY = ParagraphStyle("BodyText2", parent=styles["Normal"], fontSize=9.5, leading=13.5,
                       spaceAfter=6, alignment=TA_LEFT)
H1 = ParagraphStyle("H1c", parent=styles["Heading1"], fontSize=17, leading=21, spaceBefore=14,
                     spaceAfter=10, textColor=colors.HexColor("#1a2b4a"))
H2 = ParagraphStyle("H2c", parent=styles["Heading2"], fontSize=13.5, leading=17, spaceBefore=14,
                     spaceAfter=8, textColor=colors.HexColor("#1a2b4a"))
H3 = ParagraphStyle("H3c", parent=styles["Heading3"], fontSize=11.5, leading=15, spaceBefore=10,
                     spaceAfter=6, textColor=colors.HexColor("#2c3e63"))
H4 = ParagraphStyle("H4c", parent=styles["Heading4"], fontSize=10.5, leading=14, spaceBefore=8,
                     spaceAfter=5, textColor=colors.HexColor("#2c3e63"))
CODE = ParagraphStyle("Code2", parent=styles["Code"], fontSize=7.6, leading=9.6,
                       backColor=colors.HexColor("#f4f4f6"), leftIndent=6, spaceAfter=8)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8, leading=10.5, spaceAfter=0)
CELL_HEAD = ParagraphStyle("CellHead", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
QUOTE = ParagraphStyle("Quote", parent=BODY, leftIndent=12, textColor=colors.HexColor("#333333"),
                        backColor=colors.HexColor("#eef2f8"), borderPadding=8, spaceAfter=8)


def inline_html(node) -> str:
    """Render inline content of a tag to reportlab-compatible mini-HTML."""
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        elif isinstance(child, Tag):
            if child.name in ("strong", "b"):
                parts.append(f"<b>{inline_html(child)}</b>")
            elif child.name in ("em", "i"):
                parts.append(f"<i>{inline_html(child)}</i>")
            elif child.name == "code":
                parts.append(f'<font face="Courier" size="8" color="#a12d2d">{inline_html(child)}</font>')
            elif child.name == "font":
                face = child.get("face", "Helvetica")
                parts.append(f'<font face="{face}">{inline_html(child)}</font>')
            elif child.name == "a":
                parts.append(f'<font color="#2255aa">{inline_html(child)}</font>')
            elif child.name == "br":
                parts.append("<br/>")
            else:
                parts.append(inline_html(child))
    return "".join(parts)


def render_table(tag):
    rows = []
    thead = tag.find("thead")
    tbody = tag.find("tbody")
    if thead:
        for tr in thead.find_all("tr"):
            rows.append([Paragraph(inline_html(td), CELL_HEAD) for td in tr.find_all(["th", "td"])])
    body_rows = tbody.find_all("tr") if tbody else tag.find_all("tr")
    n_header_rows = len(rows)
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        rows.append([Paragraph(inline_html(c), CELL) for c in cells])

    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append(Paragraph("", CELL))
    avail_width = 7.3 * inch
    col_width = avail_width / ncols
    table = Table(rows, colWidths=[col_width] * ncols, repeatRows=n_header_rows)
    style = [
        ("BACKGROUND", (0, 0), (-1, n_header_rows - 1 if n_header_rows else 0),
         colors.HexColor("#2c3e63")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd6e4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, n_header_rows), (-1, -1),
         [colors.white, colors.HexColor("#f6f8fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    table.setStyle(TableStyle(style))
    return table


def render_list(tag, ordered=False, level=0):
    items = []
    for li in tag.find_all("li", recursive=False):
        sub_lists = li.find_all(["ul", "ol"], recursive=False)
        for sl in sub_lists:
            sl.extract()
        text = inline_html(li)
        items.append(ListItem(Paragraph(text, BODY), leftIndent=12 * (level + 1)))
        for sl in sub_lists:
            nested = render_list(sl, ordered=(sl.name == "ol"), level=level + 1)
            items.append(nested)
    # NOTE: passing start= alongside bulletType="bullet" forces numeric bullets in this
    # reportlab version regardless of bulletType, so start= is only passed for ordered lists.
    kwargs = dict(bulletType="1" if ordered else "bullet", leftIndent=14 * (level + 1),
                  bulletFontSize=9, bulletFontName="Helvetica")
    if ordered:
        kwargs["start"] = "1"
    return ListFlowable(items, **kwargs)


flowables = []
title_done = False

for el in soup.find_all(recursive=False):
    if el.name == "h1":
        if not title_done:
            flowables.append(Paragraph(inline_html(el), H1))
            title_done = True
        else:
            flowables.append(PageBreak())
            flowables.append(Paragraph(inline_html(el), H1))
    elif el.name == "h2":
        flowables.append(Paragraph(inline_html(el), H2))
    elif el.name == "h3":
        flowables.append(Paragraph(inline_html(el), H3))
    elif el.name == "h4":
        flowables.append(Paragraph(inline_html(el), H4))
    elif el.name == "p":
        flowables.append(Paragraph(inline_html(el), BODY))
    elif el.name == "table":
        t = render_table(el)
        if t:
            flowables.append(t)
            flowables.append(Spacer(1, 10))
    elif el.name in ("ul", "ol"):
        flowables.append(render_list(el, ordered=(el.name == "ol")))
        flowables.append(Spacer(1, 6))
    elif el.name == "pre":
        code_text = el.get_text()
        flowables.append(Preformatted(code_text, CODE))
    elif el.name == "blockquote":
        for child in el.find_all(["p"], recursive=False):
            flowables.append(Paragraph(inline_html(child), QUOTE))
    elif el.name == "hr":
        flowables.append(Spacer(1, 4))
        flowables.append(HRFlowable(width="100%", color=colors.HexColor("#cfd6e4"), thickness=0.7))
        flowables.append(Spacer(1, 8))

doc = SimpleDocTemplate(
    OUT, pagesize=LETTER,
    leftMargin=0.65 * inch, rightMargin=0.65 * inch,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    title="Phase 2 Report — DAG Execution-Time Quantile Prediction",
    author="DAG Predictor Project",
)


def add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawRightString(LETTER[0] - 0.65 * inch, 0.45 * inch, f"Page {doc_.page}")
    canvas.drawString(0.65 * inch, 0.45 * inch,
                       "Phase 2 Report — DAG Execution-Time Quantile Prediction on Heterogeneous Multicore + DVFS")
    canvas.restoreState()


doc.build(flowables, onFirstPage=add_page_number, onLaterPages=add_page_number)
print("wrote", OUT)
