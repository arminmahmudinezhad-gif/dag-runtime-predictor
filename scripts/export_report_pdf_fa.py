"""
Render a fully-Persian markdown report (PHASE2_REPORT_FA.md) to PDF with correct
right-to-left typesetting.

Why this needs its own pipeline (not just export_report_pdf.py with a Persian font):
reportlab does not implement the Unicode Bidirectional Algorithm, and critically
does not know how to LINE-WRAP right-to-left text either. This script handles both
problems explicitly:

  1. GLYPHS + READING ORDER (per line): Persian letters are reshaped into their
     correct joined presentation forms (arabic_reshaper), then the full Unicode
     bidi algorithm (python-bidi, base direction = RTL) reorders each line so
     embedded English words/numbers/file paths land in their correct position
     relative to surrounding Persian text.
  2. LINE WRAPPING (the subtle bug this script exists to avoid): if you bidi-reorder
     an ENTIRE multi-line paragraph as one string and then hand it to reportlab,
     reportlab's line-breaker -- which has no idea the text is already in reordered
     RTL display order -- chops it at arbitrary points sized to fit the column
     width, scrambling logical flow ACROSS line breaks (verified by rendering and
     visually inspecting a real multi-line paragraph: words from the middle/end of
     a sentence ended up on an earlier visual line than words from the start).
     The fix: wrap the PLAIN LOGICAL text into lines FIRST, using real font-metric
     width measurement (mixed Persian/Latin, since GeezaPro and Helvetica have
     disjoint glyph coverage and different character widths), and only THEN run
     reshape+bidi independently on each already-correctly-sized line. Lines are
     joined into one Paragraph with explicit <br/> tags so reportlab never
     re-wraps them.
  3. FONT COVERAGE: GeezaPro (font 0 of the .ttc) has zero Latin/digit glyphs;
     Helvetica has zero Arabic-script glyphs. Each reordered line is split into
     runs by actual glyph coverage and wrapped in the matching <font face=...> tag
     so nothing renders as a black box (verified by rendering every unique
     character in the source and checking for .notdef glyphs).

Deliberate simplification (documented, not accidental): inline **bold**/*italic*
markup WITHIN body paragraphs, list items, and table cells is not preserved.
Tracking emphasis spans through both the line-wrap step and the bidi reorder step
compounds the same class of edge-case risk (bidi-neutral/mirrored characters
drifting across a span boundary) that this script exists to eliminate. Headings
remain visually bold via paragraph style (not inline markup). Fenced code blocks
are rendered completely unprocessed (pure LTR, monospace, never bidi-reordered)
since they are literal shell/code syntax -- the source markdown keeps their
comments in English for exactly this reason.

Usage:  python3 scripts/export_report_pdf_fa.py PHASE2_REPORT_FA.md PHASE2_REPORT_FA.pdf
Requires: pip install reportlab markdown beautifulsoup4 arabic-reshaper python-bidi
macOS-specific: assumes /System/Library/Fonts/GeezaPro.ttc (ships with every macOS
install; font 0 of the collection has full Arabic-script glyph coverage, unlike
SF Arabic which is missing several presentation-form codepoints).
"""

import sys

import arabic_reshaper
import markdown
from bidi.algorithm import get_display
from bs4 import BeautifulSoup, Tag
from fontTools.ttLib import TTCollection
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
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

GEEZA_PATH = "/System/Library/Fonts/GeezaPro.ttc"
pdfmetrics.registerFont(TTFont("GeezaPro", GEEZA_PATH, subfontIndex=0))
pdfmetrics.registerFont(TTFont("GeezaProBold", GEEZA_PATH, subfontIndex=1))

_coll = TTCollection(GEEZA_PATH)
GEEZA_CMAP = set(_coll.fonts[0].getBestCmap().keys())

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN = 0.65 * inch
FULL_WIDTH = PAGE_WIDTH - 2 * MARGIN


def is_fa_char(ch: str) -> bool:
    return ord(ch) in GEEZA_CMAP


def char_width(ch: str, font_size: float) -> float:
    font = "GeezaPro" if is_fa_char(ch) else "Helvetica"
    try:
        return pdfmetrics.stringWidth(ch, font, font_size)
    except Exception:
        return font_size * 0.55


def measure(text: str, font_size: float) -> float:
    return sum(char_width(ch, font_size) for ch in text)


def wrap_to_lines(text: str, avail_width: float, font_size: float) -> list[str]:
    """Greedy word-wrap of PLAIN LOGICAL text (pre-shaping, pre-bidi) into lines
    that fit avail_width, using real per-character font-metric widths."""
    words = text.split(" ")
    space_w = char_width(" ", font_size) or font_size * 0.3
    lines, cur, cur_w = [], [], 0.0
    for word in words:
        if word == "":
            continue
        w = measure(word, font_size)
        add_w = w if not cur else w + space_w
        if cur and cur_w + add_w > avail_width:
            lines.append(" ".join(cur))
            cur, cur_w = [word], w
        else:
            cur.append(word)
            cur_w += add_w
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def shape_line(text: str) -> str:
    """Reshape + bidi-reorder ONE LINE (already known to fit its column), then
    split into <font face=...> runs by actual glyph coverage. Safe to call only
    on text that is already a single visual line -- see module docstring."""
    if not text.strip():
        return text
    reshaped = arabic_reshaper.reshape(text)
    display = get_display(reshaped, base_dir="R")

    runs = []
    cur_cls, cur_chars = None, []
    for ch in display:
        cls = "FA" if is_fa_char(ch) else "LT"
        if cls != cur_cls and cur_chars:
            runs.append((cur_cls, "".join(cur_chars)))
            cur_chars = []
        cur_cls = cls
        cur_chars.append(ch)
    if cur_chars:
        runs.append((cur_cls, "".join(cur_chars)))

    parts = []
    for cls, chunk in runs:
        esc = chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        font = "GeezaPro" if cls == "FA" else "Helvetica"
        parts.append(f'<font face="{font}">{esc}</font>')
    return "".join(parts)


def shaped_multiline(text: str, avail_width: float, font_size: float) -> str:
    """Full pipeline for text that may need to wrap: logical-order word-wrap
    first, then per-line reshape+bidi, joined with explicit <br/> so reportlab
    never re-wraps the already-reordered result."""
    lines = wrap_to_lines(text, avail_width, font_size)
    return "<br/>".join(shape_line(line) for line in lines)


with open(SRC, encoding="utf-8") as f:
    md_text = f.read()

html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
soup = BeautifulSoup(html, "html.parser")

styles = getSampleStyleSheet()
BODY_SIZE = 10.5
BODY = ParagraphStyle("BodyFa", fontName="GeezaPro", fontSize=BODY_SIZE, leading=18,
                       spaceAfter=8, alignment=TA_RIGHT)
H1 = ParagraphStyle("H1Fa", fontName="GeezaProBold", fontSize=17, leading=24, spaceBefore=14,
                     spaceAfter=10, alignment=TA_RIGHT, textColor=colors.HexColor("#1a2b4a"))
H2 = ParagraphStyle("H2Fa", fontName="GeezaProBold", fontSize=13.5, leading=20, spaceBefore=14,
                     spaceAfter=8, alignment=TA_RIGHT, textColor=colors.HexColor("#1a2b4a"))
H3 = ParagraphStyle("H3Fa", fontName="GeezaProBold", fontSize=11.5, leading=18, spaceBefore=10,
                     spaceAfter=6, alignment=TA_RIGHT, textColor=colors.HexColor("#2c3e63"))
H4 = ParagraphStyle("H4Fa", fontName="GeezaProBold", fontSize=10.5, leading=16, spaceBefore=8,
                     spaceAfter=5, alignment=TA_RIGHT, textColor=colors.HexColor("#2c3e63"))
CODE = ParagraphStyle("CodeFa", parent=styles["Code"], fontSize=7.4, leading=9.4,
                       backColor=colors.HexColor("#f4f4f6"), leftIndent=6, spaceAfter=8,
                       alignment=TA_RIGHT)
CELL_SIZE = 8.3
CELL = ParagraphStyle("CellFa", fontName="GeezaPro", fontSize=CELL_SIZE, leading=13,
                       spaceAfter=0, alignment=TA_RIGHT)
CELL_HEAD = ParagraphStyle("CellHeadFa", parent=CELL, fontName="GeezaProBold",
                            textColor=colors.white)
QUOTE = ParagraphStyle("QuoteFa", parent=BODY, rightIndent=12, textColor=colors.HexColor("#333333"),
                        backColor=colors.HexColor("#eef2f8"), borderPadding=8, spaceAfter=8)

CELL_PAD = 8.0  # LEFTPADDING + RIGHTPADDING in render_table


def para(text: str, style: ParagraphStyle, avail_width: float) -> Paragraph:
    shaped = shaped_multiline(text, avail_width, style.fontSize)
    return Paragraph(shaped, style)


def render_table(tag):
    header_texts, body_rows_text = [], []
    thead = tag.find("thead")
    tbody = tag.find("tbody")
    if thead:
        for tr in thead.find_all("tr"):
            header_texts.append([td.get_text() for td in tr.find_all(["th", "td"])])
    body_trs = tbody.find_all("tr") if tbody else tag.find_all("tr")
    for tr in body_trs:
        cells = tr.find_all(["td", "th"])
        if cells:
            body_rows_text.append([c.get_text() for c in cells])

    all_rows_text = header_texts + body_rows_text
    if not all_rows_text:
        return None
    ncols = max(len(r) for r in all_rows_text)
    for r in all_rows_text:
        while len(r) < ncols:
            r.append("")

    col_width = FULL_WIDTH / ncols
    cell_avail = col_width - CELL_PAD - 2

    n_header_rows = len(header_texts)
    rows = []
    for i, row_text in enumerate(all_rows_text):
        style = CELL_HEAD if i < n_header_rows else CELL
        rows.append([para(cell_text, style, cell_avail) for cell_text in row_text])

    table = Table(rows, colWidths=[col_width] * ncols, repeatRows=n_header_rows)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, n_header_rows - 1 if n_header_rows else 0),
         colors.HexColor("#2c3e63")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd6e4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, n_header_rows), (-1, -1),
         [colors.white, colors.HexColor("#f6f8fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def render_list(tag, ordered=False, level=0):
    """Returns a flat list of Paragraph flowables. The bullet/number marker is
    prepended to the LOGICAL text before wrapping+shaping, so it lands at the
    correct (right/start-of-reading) edge via the normal pipeline -- reportlab's
    own ListFlowable bullet positioning has no working RTL support in this
    version (bullets stayed stuck on the left regardless of bulletDir)."""
    out = []
    counter = 1
    right_indent = 14 * (level + 1)
    avail_width = FULL_WIDTH - right_indent
    for li in tag.find_all("li", recursive=False):
        sub_lists = li.find_all(["ul", "ol"], recursive=False)
        for sl in sub_lists:
            sl.extract()
        marker = f"{str(counter).translate(PERSIAN_DIGITS)}." if ordered else "•"
        text = f"{marker} {li.get_text()}"
        style = ParagraphStyle(f"ListFa{level}", parent=BODY, rightIndent=right_indent)
        out.append(para(text, style, avail_width))
        counter += 1
        for sl in sub_lists:
            out.extend(render_list(sl, ordered=(sl.name == "ol"), level=level + 1))
    return out


flowables = []
title_done = False

for el in soup.find_all(recursive=False):
    if el.name == "h1":
        if not title_done:
            flowables.append(para(el.get_text(), H1, FULL_WIDTH))
            title_done = True
        else:
            flowables.append(PageBreak())
            flowables.append(para(el.get_text(), H1, FULL_WIDTH))
    elif el.name == "h2":
        flowables.append(para(el.get_text(), H2, FULL_WIDTH))
    elif el.name == "h3":
        flowables.append(para(el.get_text(), H3, FULL_WIDTH))
    elif el.name == "h4":
        flowables.append(para(el.get_text(), H4, FULL_WIDTH))
    elif el.name == "p":
        flowables.append(para(el.get_text(), BODY, FULL_WIDTH))
    elif el.name == "table":
        t = render_table(el)
        if t:
            flowables.append(t)
            flowables.append(Spacer(1, 10))
    elif el.name in ("ul", "ol"):
        flowables.extend(render_list(el, ordered=(el.name == "ol")))
        flowables.append(Spacer(1, 6))
    elif el.name == "pre":
        code_text = el.get_text()
        flowables.append(Preformatted(code_text, CODE))
    elif el.name == "blockquote":
        for child in el.find_all(["p"], recursive=False):
            flowables.append(para(child.get_text(), QUOTE, FULL_WIDTH - 12))
    elif el.name == "hr":
        flowables.append(Spacer(1, 4))
        flowables.append(HRFlowable(width="100%", color=colors.HexColor("#cfd6e4"), thickness=0.7))
        flowables.append(Spacer(1, 8))

doc = SimpleDocTemplate(
    OUT, pagesize=LETTER,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    title="گزارش فاز ۲ — پیش‌بینی چندکی زمان اجرای DAG",
    author="DAG Predictor Project",
)

def add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(MARGIN, 0.45 * inch, f"Page {doc_.page}")
    footer_para = para("گزارش فاز ۲ — پیش‌بینی چندکی زمان اجرای DAG",
                        ParagraphStyle("Footer", fontName="GeezaPro", fontSize=8,
                                       textColor=colors.HexColor("#888888"), alignment=TA_RIGHT),
                        FULL_WIDTH)
    w, h = footer_para.wrap(FULL_WIDTH, 20)
    footer_para.drawOn(canvas, MARGIN, 0.45 * inch - 2)
    canvas.restoreState()


doc.build(flowables, onFirstPage=add_page_number, onLaterPages=add_page_number)
print("wrote", OUT)
