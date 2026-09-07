"""
report_generator.py
=====================
Builds a professional, downloadable PDF report for a single water-quality
analysis using ReportLab. Includes: header/branding, user inputs, prediction
badge, confidence, full Explainable AI breakdown (reasons + contribution
percentages), probable causes, recommendations, and a timestamp/footer.
"""
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(BASE_DIR, "static", "images", "icon.png")

PREDICTION_COLORS = {
    "Safe": colors.HexColor("#1fae6b"),
    "Caution": colors.HexColor("#e0a419"),
    "Unsafe": colors.HexColor("#e0453f"),
}

BRAND_TEAL = colors.HexColor("#0e7c8c")
BRAND_DARK = colors.HexColor("#0b2035")
MUTED = colors.HexColor("#5b7688")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="HSTitle", fontSize=20, leading=24, textColor=BRAND_DARK,
                           fontName="Helvetica-Bold", spaceAfter=2))
    ss.add(ParagraphStyle(name="HSSubtitle", fontSize=10, textColor=MUTED, fontName="Helvetica"))
    ss.add(ParagraphStyle(name="HSSection", fontSize=13, leading=16, textColor=BRAND_TEAL,
                           fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle(name="HSBody", fontSize=10, leading=15, textColor=BRAND_DARK,
                           fontName="Helvetica"))
    ss.add(ParagraphStyle(name="HSBodyMuted", fontSize=9, leading=13, textColor=MUTED,
                           fontName="Helvetica"))
    ss.add(ParagraphStyle(name="HSPredBig", fontSize=26, leading=30, fontName="Helvetica-Bold",
                           alignment=TA_CENTER))
    ss.add(ParagraphStyle(name="HSPredSub", fontSize=11, leading=14, alignment=TA_CENTER,
                           textColor=colors.white, fontName="Helvetica"))
    return ss


def _param_table(params: dict, statuses: dict, styles):
    from ml.domain_knowledge import FEATURE_LABELS, FEATURE_UNITS

    header = ["Parameter", "Measured Value", "Status"]
    rows = [header]
    for f in ["ph", "chlorine", "hardness", "nitrate"]:
        unit = FEATURE_UNITS[f]
        val = f"{params[f]:.2f} {unit}".strip()
        rows.append([FEATURE_LABELS[f], val, statuses[f]])

    t = Table(rows, colWidths=[55 * mm, 55 * mm, 55 * mm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8e4e9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, f in enumerate(["ph", "chlorine", "hardness", "nitrate"], start=1):
        c = PREDICTION_COLORS.get(statuses[f], MUTED) if statuses[f] != "Safe" else colors.HexColor("#1fae6b")
        style.append(("TEXTCOLOR", (2, i), (2, i), c))
        style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def _contribution_table(contribution: dict, styles):
    from ml.domain_knowledge import FEATURE_LABELS

    ordered = sorted(contribution.items(), key=lambda kv: -kv[1])
    max_bar_width = 70 * mm

    rows = [["Parameter", "Contribution", "Visual"]]
    for f, pct in ordered:
        bar_w = max(2, (pct / 100.0) * max_bar_width)
        bar = Table([[""]], colWidths=[bar_w], rowHeights=[6])
        bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BRAND_TEAL)]))
        rows.append([FEATURE_LABELS[f], f"{pct:.1f}%", bar])

    t = Table(rows, colWidths=[40 * mm, 25 * mm, max_bar_width + 5 * mm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8e4e9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    t.setStyle(TableStyle(style))
    return t


def build_report(analysis: dict, output_path: str) -> str:
    """
    analysis: dict shaped like the Explainer output PLUS 'params' (raw
              ph/chlorine/hardness/nitrate) and optional 'sample_label',
              'analysis_id', 'created_at'.
    Writes a PDF to output_path and returns that path.
    """
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
        title="HydroScan AI Water Quality Report",
    )
    story = []

    # ---------------- Header ----------------
    header_cells = []
    if os.path.exists(LOGO_PATH):
        try:
            header_cells.append(Image(LOGO_PATH, width=16 * mm, height=16 * mm))
        except Exception:
            header_cells.append("")
    title_block = [
        Paragraph("HydroScan AI", styles["HSTitle"]),
        Paragraph("Explainable Water Quality Analysis Report", styles["HSSubtitle"]),
    ]
    if header_cells:
        header_tbl = Table([[header_cells[0], title_block]], colWidths=[20 * mm, 150 * mm])
        header_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(header_tbl)
    else:
        story.extend(title_block)

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d8e4e9")))
    story.append(Spacer(1, 10))

    meta_line = (f"Report ID: HS-{analysis.get('analysis_id', '—')}   |   "
                 f"Generated: {analysis.get('created_at', datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))}")
    if analysis.get("sample_label"):
        meta_line += f"   |   Sample: {analysis['sample_label']}"
    story.append(Paragraph(meta_line, styles["HSBodyMuted"]))
    story.append(Spacer(1, 12))

    # ---------------- Prediction banner ----------------
    pred = analysis["prediction"]
    pred_color = PREDICTION_COLORS.get(pred, BRAND_TEAL)
    banner_data = [[
        Paragraph(pred.upper(), ParagraphStyle(name="p", parent=styles["HSPredBig"], textColor=colors.white)),
    ], [
        Paragraph(f"Confidence: {analysis['confidence']:.1f}%", styles["HSPredSub"]),
    ]]
    banner = Table(banner_data, colWidths=[174 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pred_color),
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("BOTTOMPADDING", (0, 1), (0, 1), 12),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 14))

    story.append(Paragraph(analysis["plain_summary"], styles["HSBody"]))
    story.append(Spacer(1, 10))

    # ---------------- Measured parameters ----------------
    story.append(Paragraph("Measured Parameters", styles["HSSection"]))
    story.append(_param_table(analysis["params"], analysis["parameter_statuses"], styles))
    story.append(Spacer(1, 6))

    # ---------------- XAI: reasons ----------------
    story.append(Paragraph("Why This Prediction Was Made", styles["HSSection"]))
    reason_items = [ListItem(Paragraph(r, styles["HSBody"]), bulletColor=BRAND_TEAL) for r in analysis["reasons"]]
    story.append(ListFlowable(reason_items, bulletType="bullet", start="circle", leftIndent=12))
    story.append(Spacer(1, 6))

    # ---------------- XAI: contribution ----------------
    story.append(Paragraph(
        f"Parameter Contribution to Result  (Top factor: {analysis['top_influential_parameter']})",
        styles["HSSection"]))
    story.append(_contribution_table(analysis["contribution_percent"], styles))
    story.append(Spacer(1, 6))

    # ---------------- Probable causes ----------------
    if analysis.get("causes"):
        story.append(Paragraph("Probable Causes", styles["HSSection"]))
        cause_items = [ListItem(Paragraph(c, styles["HSBody"]), bulletColor=BRAND_TEAL) for c in analysis["causes"]]
        story.append(ListFlowable(cause_items, bulletType="bullet", start="circle", leftIndent=12))
        story.append(Spacer(1, 6))

    # ---------------- Recommendations ----------------
    story.append(Paragraph("Recommendations", styles["HSSection"]))
    rec_items = [ListItem(Paragraph(r, styles["HSBody"]), bulletColor=BRAND_TEAL) for r in analysis["recommendations"]]
    story.append(ListFlowable(rec_items, bulletType="bullet", start="circle", leftIndent=12))
    story.append(Spacer(1, 16))

    # ---------------- Footer / disclaimer ----------------
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#d8e4e9")))
    story.append(Spacer(1, 6))
    disclaimer = (
        "Disclaimer: HydroScan AI provides an educational, AI-assisted estimate based on the "
        "parameters provided or detected from a test-strip photo. It is not a certified laboratory "
        "analysis. For regulatory, medical, or legal decisions, confirm results with an accredited "
        "water-testing laboratory."
    )
    story.append(Paragraph(disclaimer, styles["HSBodyMuted"]))

    doc.build(story)
    return output_path
