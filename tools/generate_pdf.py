"""
generate_pdf.py — NeuroGuide 仿真报告生成模块
竖向顺序排列，紧凑间距，A4单页
"""

import os, io, time, tempfile, glob, re
from pathlib import Path
import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
USABLE_W = PAGE_W - 2 * MARGIN   # 180 mm
USABLE_H = PAGE_H - 2 * MARGIN   # 267 mm

_CJK_FONT_NAME = "Helvetica"
TITLE_BLUE = DIVIDER = HIGHLIGHT = BLACK = WHITE = GRAY = DARK_BG = LIGHT_BG = None


def _init_all():
    global _CJK_FONT_NAME, TITLE_BLUE, DIVIDER, HIGHLIGHT, BLACK, WHITE, GRAY, DARK_BG, LIGHT_BG
    if not REPORTLAB_AVAILABLE:
        return
    for _name, _path in [("SimHei","C:/Windows/Fonts/simhei.ttf"),
                         ("Microsoft YaHei","C:/Windows/Fonts/msyh.ttc"),
                         ("Microsoft YaHei Bold","C:/Windows/Fonts/msyhbd.ttc")]:
        if os.path.exists(_path):
            try:
                pdfmetrics.registerFont(TTFont(_name, _path))
                _CJK_FONT_NAME = _name
                break
            except Exception:
                continue
    TITLE_BLUE = HexColor("#1A365D")
    DIVIDER    = HexColor("#E2E8F0")
    HIGHLIGHT  = HexColor("#E53E3E")
    BLACK      = HexColor("#1A202C")
    WHITE      = HexColor("#FFFFFF")
    GRAY       = HexColor("#718096")
    DARK_BG    = HexColor("#2D3748")
    LIGHT_BG   = HexColor("#F7FAFC")


# ═══ 工具函数 ═══

def _hline():
    return Table([[""]], colWidths=[USABLE_W], rowHeights=[1 * mm],
                 style=TableStyle([
                     ("LINEBELOW", (0, 0), (-1, 0), 0.5, DIVIDER),
                     ("TOPPADDING", (0, 0), (-1, -1), 0),
                     ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                 ]))


def _fmt(v, d="—"):
    return v or d


def _fig_to_png(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(buf.read())
        return tmp.name


def _png_size(path):
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            return im.size[0] * 0.75, im.size[1] * 0.75
    except Exception:
        return 100, 100


def _coord_str(affine, sh):
    if affine is None:
        return f"({sh[0]//2}, {sh[1]//2}, {sh[2]//2}) voxel"
    m = affine @ np.array([sh[0] // 2, sh[1] // 2, sh[2] // 2, 1])
    return f"({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f}) mm"


def _clean_name(raw_name):
    """去掉括号及括号内内容，只保留英文/数字/下划线/短横"""
    name = re.sub(r'\s*\(.*?\)\s*', ' ', raw_name)  # 去掉括号
    name = re.sub(r'[^\w\-\s]', '', name)           # 只保留英文/数字/空格/下划线/短横
    return ' '.join(name.split()).strip() or raw_name


# ═══ 模块构建 ═══

def _build_header(story):
    fn = _CJK_FONT_NAME
    tbl = Table([[
        Paragraph("NeuroGuide 神经调控电场仿真诊断报告",
                  ParagraphStyle("H", fontName=fn, fontSize=13,
                                 textColor=TITLE_BLUE, alignment=TA_CENTER)),
    ]], colWidths=[USABLE_W])
    tbl.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(tbl)
    story.append(Paragraph(
        f"报告日期：{time.strftime('%Y-%m-%d')}　|　版本：Ver 3.5",
        ParagraphStyle("D", fontName=fn, fontSize=7, textColor=GRAY, alignment=TA_RIGHT)))
    story.append(Spacer(1, 3 * mm))
    story.append(_hline())
    story.append(Spacer(1, 3 * mm))


def _build_patient_info(story, patient_info, disease):
    fn = _CJK_FONT_NAME
    vs = ParagraphStyle("V", fontName=fn, fontSize=8.5, textColor=BLACK, leading=12)
    fields = [
        ("患者姓名", patient_info.get("name", "")),
        ("性　　别", patient_info.get("gender", "")),
        ("年　　龄", patient_info.get("age", "")),
        ("头　　围", patient_info.get("head_circumference", "")),
        ("患病类型", patient_info.get("disease_type", "") or disease),
        ("报告编号", f"NG-{time.strftime('%Y%m%d%H%M%S')}"),
    ]
    cells = [Paragraph(f"{l}：<b>{_fmt(v)}</b>", vs) for l, v in fields]
    cw = USABLE_W / 2 - 1 * mm
    rows = []
    for i in range(0, len(cells), 2):
        r = cells[i:i + 2]
        if len(r) == 1:
            r.append(Paragraph("", vs))
        rows.append(r)
    t = Table(rows, colWidths=[cw, cw])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 3 * mm))
    story.append(_hline())
    story.append(Spacer(1, 3 * mm))


def _build_key_params(story, patient_info, target, disease):
    fn = _CJK_FONT_NAME
    kd = [
        ["仿真关键参数", "参数值"],
        ["靶区 (Target)", _fmt(target)],
        ["刺激电流", _fmt(patient_info.get("current"))],
        ["患病类型", _fmt(patient_info.get("disease_type") or disease)],
    ]
    ks = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), fn),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1A365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#FFF5F5")),
        ("TEXTCOLOR", (0, 1), (-1, -1), BLACK),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#E53E3E")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ])
    t = Table(kd, colWidths=[USABLE_W * 0.3, USABLE_W * 0.7])
    t.setStyle(ks)
    story.append(t)
    story.append(Spacer(1, 3 * mm))
    story.append(_hline())
    story.append(Spacer(1, 3 * mm))


def _build_overlay_table(story, engine):
    fn = _CJK_FONT_NAME
    story.append(Paragraph("叠加层电场分布",
        ParagraphStyle("OT", fontName=fn, fontSize=10, textColor=TITLE_BLUE,
                       leading=13, spaceAfter=1 * mm)))

    if engine.overlays:
        ov_rows = [["叠加层名称", "类型", "强度范围 (min ~ max)"]]
        for ov in engine.overlays:
            raw = ov.get("name", "—")
            nm = _clean_name(raw)[:28]
            tp = "图谱" if ov.get("is_atlas") else "电场"
            mi = ov.get("min", 0) or 0
            ma = ov.get("max", 0) or 0
            ov_rows.append([nm, tp, f"{mi:.4f} ~ {ma:.4f}"])
        ot = Table(ov_rows, colWidths=[USABLE_W * 0.38, USABLE_W * 0.15, USABLE_W * 0.47])
        ot.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), fn),
            ("BACKGROUND", (0, 0), (-1, 0), DARK_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("BACKGROUND", (0, 1), (-1, -1), WHITE),
            ("TEXTCOLOR", (0, 1), (-1, -1), BLACK),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, DIVIDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ot)
    else:
        story.append(Paragraph("（无叠加层数据）",
            ParagraphStyle("N", fontName=fn, fontSize=8, textColor=GRAY)))
    story.append(Spacer(1, 3 * mm))
    story.append(_hline())
    story.append(Spacer(1, 3 * mm))


def _build_coord_and_electrode(story, engine, patient_info):
    fn = _CJK_FONT_NAME
    sh = engine.data.shape

    coord = _coord_str(engine.affine, sh)
    story.append(Paragraph(
        f"📍 当前视点坐标：X={sh[0]//2}　Y={sh[1]//2}　Z={sh[2]//2}　　{coord}",
        ParagraphStyle("C", fontName=fn, fontSize=8, textColor=TITLE_BLUE, leading=12,
                       backColor=HexColor("#EBF4FF"), borderPadding=4, borderRadius=2)))
    story.append(Spacer(1, 3 * mm))

    ep = patient_info.get("electrode_map_path")
    if ep and os.path.exists(ep):
        try:
            ew, eh = _png_size(ep)
            sc = min(45 * mm / ew, 45 * mm / eh, 1.0)
            story.append(Paragraph("10-10 电极位置图",
                ParagraphStyle("EpT", fontName=fn, fontSize=8, textColor=GRAY, leading=10)))
            story.append(RLImage(ep, width=ew * sc, height=eh * sc))
        except Exception:
            pass
    story.append(Spacer(1, 1 * mm))
    story.append(_hline())
    story.append(Spacer(1, 1 * mm))


def _build_three_views(story, canvases):
    fn = _CJK_FONT_NAME
    captions = {"sagittal": "矢状面 Sagittal", "coronal": "冠状面 Coronal", "axial": "轴状面 Axial"}

    temp_files = []
    fig_paths = {}
    for key in ("sagittal", "coronal", "axial"):
        c = canvases.get(key) if canvases else None
        if c is None or c.fig is None:
            continue
        p = _fig_to_png(c.fig, dpi=100)
        temp_files.append(p)
        fig_paths[key] = p

    if not fig_paths:
        return []

    story.append(Paragraph("三平面视图",
        ParagraphStyle("3T", fontName=fn, fontSize=10, textColor=TITLE_BLUE,
                       leading=13, spaceAfter=1 * mm)))

    n = len(fig_paths)
    gap = 3 * mm
    each_w = (USABLE_W - (n - 1) * gap) / n
    # 给三视图更大的高度，使其靠近底部
    each_h = USABLE_H * 0.45

    sizes = {k: _png_size(p) for k, p in fig_paths.items()}
    sc = 1.0
    for pw, ph in sizes.values():
        sc = min(sc, each_w / pw, each_h / ph)

    cells = []
    for key in ("sagittal", "coronal", "axial"):
        if key not in fig_paths:
            cells.append(Paragraph("—", ParagraphStyle("X", fontName=fn, fontSize=7)))
            continue
        pw, ph = sizes[key]
        col = [
            Paragraph(captions[key],
                ParagraphStyle("Cap", fontName=fn, fontSize=7.5,
                               textColor=TITLE_BLUE, alignment=TA_CENTER, leading=10)),
            Spacer(1, 1 * mm),
            RLImage(fig_paths[key], width=pw * sc, height=ph * sc),
        ]
        cell = Table([[col]], colWidths=[each_w])
        cell.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 1),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ]))
        cells.append(cell)

    row = Table([cells], colWidths=[each_w] * n)
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(row)
    story.append(Spacer(1, 3 * mm))
    story.append(_hline())
    story.append(Spacer(1, 3 * mm))
    return temp_files


def _build_footer(story):
    fn = _CJK_FONT_NAME
    t = Table([[
        Paragraph(f"报告编号：NG-{time.strftime('%Y%m%d%H%M%S')}　|　CONFIDENTIAL",
                  ParagraphStyle("F", fontName=fn, fontSize=6.5, textColor=GRAY)),
        Paragraph("— 第 1 页 / 共 1 页 —",
                  ParagraphStyle("G", fontName=fn, fontSize=6.5, textColor=GRAY,
                                 alignment=TA_RIGHT)),
    ]], colWidths=[USABLE_W * 0.5, USABLE_W * 0.5])
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, DIVIDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)


# ═══ 主入口 ═══

def generate_report(engine, base_path, disease, target,
                    output_path=None, canvases=None, patient_info=None):
    if not REPORTLAB_AVAILABLE:
        return False, "缺少 reportlab 库。pip install reportlab"
    if engine.data is None:
        return False, "无加载底图。"

    _init_all()
    if patient_info is None:
        patient_info = {}

    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        pn = patient_info.get("name", "") or "Patient"
        output_path = str(Path.home() / "Desktop" / f"NeuroGuide_{pn}_{ts}.pdf")

    temp_files = []
    try:
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
        )
        story = []

        _build_header(story)
        _build_patient_info(story, patient_info, disease)
        _build_key_params(story, patient_info, target, disease)
        _build_overlay_table(story, engine)
        _build_coord_and_electrode(story, engine, patient_info)
        tf = _build_three_views(story, canvases)
        if tf:
            temp_files.extend(tf)
        _build_footer(story)

        doc.build(story)

        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
        for f in glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.png")):
            try:
                os.unlink(f)
            except Exception:
                pass
        return True, output_path

    except Exception as e:
        import traceback
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
        return False, f"生成报告失败: {e}\n\n详细错误:\n{traceback.format_exc()}"