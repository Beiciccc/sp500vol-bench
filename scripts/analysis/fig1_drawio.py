#!/usr/bin/env python3
"""Figure 1 v2 — the identity audit as a concept diagram (draw.io source).

Replaces the four-text-list layout: panel 3 becomes the figure's protagonist,
drawing the paper's actual argument — (a) a filing reaches the forecast through
TWO channels, content and firm identity, and a naive reference credits both;
(b) strengthening references collapses Holm survivors 38->8->9->0 of 69, drawn
as a bar ladder (lengths carry the data); (c) two mechanism probes shown as
micro-diagrams: entity masking (identity share 0.51) and matched-firm swap
(kills 84-93%). Panels 1/2/4 compress to feeds and verdict.

Authored as native draw.io XML (editable in the app), exported via the desktop
CLI to PDF, then ghostscript-outlined into figures/figure1_vnext.pdf in place.

Typography floor: every fontSize >= 25 model units; the canvas is 1800 wide and
prints at \\textwidth (504 pt), so 25 px -> 7.0 pt. Hero numerals 44-56.
Palette: the audited panel tints; red(#C0392B)/vermillion reserved for the
naive-trap semantics, green (#1E7A4E) for verdict checks, all else slate/ink.
"""
import base64
import io
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "writing/paper/figures"
ICONS = FIG / "icons"

INK, SUB, SLATE = "#132238", "#5F6F82", "#2F3B4C"
RED, GREEN, VERM = "#C0392B", "#1E7A4E", "#C85800"
P1 = dict(x=10, w=392, tint="#F4F8FE", stroke="#9DBFE8", tc="#2F6FB2",
          t="Point-in-time benchmark", sub="survivorship-free")
P2 = dict(x=440, w=336, tint="#F2FBF8", stroke="#9FD8CA", tc="#0D806A",
          t="Task &amp; challengers", sub="same splits &amp; losses")
P3 = dict(x=814, w=600, tint="#FFF8EF", stroke="#F0BE83", tc="#AC5E03",
          t="The identity audit", sub="sever &#8216;who filed&#8217;, keep the content")
P4 = dict(x=1452, w=336, tint="#F8F6FF", stroke="#C4B7E8", tc="#6D55B7",
          t="Gates &amp; verdict", sub="pre-declared, powered")

# Vertical rhythm. Helvetica at BODY px needs ~1.25x line pitch to read
# uncrowded; the previous revision ran 25-28 units of pitch on a 26px face
# (measured: 40+ sub-2-unit gaps), which is what read as "cramped".
BODY = 26            # floor: 26 units x 0.7207 pt/unit x (503.75/1284) = 7.35pt
PITCH = 32           # 1.23 x BODY
HEAD = 27
W, H, PY, PH = 1800, 588, 34, 546

_cells: list[str] = []
_n = 0


def _id() -> str:
    global _n
    _n += 1
    return f"c{_n}"


def V(x, y, w, h, style):
    _cells.append(f'<mxCell id="{_id()}" style="{style}" vertex="1" parent="1">'
                  f'<mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" as="geometry"/></mxCell>')


def T(x, y, w, h, text, size=25, color=INK, bold=False, align="left",
      valign="top", style_extra=""):
    st = (f"text;html=1;fontFamily=Helvetica;fontSize={size};fontColor={color};"
          f"align={align};verticalAlign={valign};"
          + ("fontStyle=1;" if bold else "") + style_extra)
    _cells.append(f'<mxCell id="{_id()}" value="{text}" style="{st}" vertex="1" parent="1">'
                  f'<mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" as="geometry"/></mxCell>')


def E(x1, y1, x2, y2, color=SLATE, width=2.5, dashed=False, curved=False,
      points=None, end="block"):
    st = (f"edgeStyle=none;strokeColor={color};strokeWidth={width};"
          f"endArrow={end};endFill=1;html=1;"
          + ("dashed=1;dashPattern=6 5;" if dashed else "")
          + ("curved=1;" if curved else ""))
    pts = ""
    if points:
        inner = "".join(f'<mxPoint x="{px:.0f}" y="{py:.0f}"/>' for px, py in points)
        pts = f'<Array as="points">{inner}</Array>'
    _cells.append(f'<mxCell id="{_id()}" style="{st}" edge="1" parent="1">'
                  f'<mxGeometry relative="1" as="geometry">'
                  f'<mxPoint x="{x1:.0f}" y="{y1:.0f}" as="sourcePoint"/>'
                  f'<mxPoint x="{x2:.0f}" y="{y2:.0f}" as="targetPoint"/>{pts}'
                  f'</mxGeometry></mxCell>')


_B64 = {}


def IC(name, x, y, s):
    if name not in _B64:
        from PIL import Image
        svg = ICONS / f"{name}.svg"
        if svg.exists():
            import subprocess as _sp
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                _sp.run(["rsvg-convert", "-w", "280", "-h", "280", "-o",
                         tf.name, str(svg)], check=True)
                im = Image.open(tf.name).convert("RGBA")
        else:
            im = Image.open(ICONS / f"{name}.png").convert("RGBA")
        bbox = im.getbbox()
        im = im.crop(bbox) if bbox else im
        side = max(im.size)
        pad = int(side * 0.04)
        canvas = Image.new("RGBA", (side + 2 * pad,) * 2, (0, 0, 0, 0))
        canvas.paste(im, (pad + (side - im.width) // 2, pad + (side - im.height) // 2))
        canvas = canvas.resize((140, 140), Image.LANCZOS)
        buf = io.BytesIO()
        canvas.save(buf, "PNG", optimize=True)
        _B64[name] = base64.b64encode(buf.getvalue()).decode()
    V(x, y, s, s, f"image;imageAspect=0;image=data:image/png,{_B64[name]};")


def panel(p, num):
    V(p["x"], PY, p["w"], PH,
      f"rounded=1;absoluteArcSize=1;arcSize=16;fillColor={p['tint']};"
      f"strokeColor={p['stroke']};strokeWidth=1.7;")
    V(p["x"] + 16, PY - 20, 40, 40,
      f"ellipse;fillColor={p['tc']};strokeColor=none;")
    T(p["x"] + 16, PY - 17, 40, 34, str(num), 27, "#FFFFFF", True, "center", "middle")
    T(p["x"] + 52, PY + 6, p["w"] - 60, 34, p["t"], HEAD, p["tc"], True)
    T(p["x"] + 52, PY + 40, p["w"] - 60, 30, p["sub"], BODY, SUB)


CHIP = ("rounded=1;absoluteArcSize=1;arcSize=10;fillColor=#FFFFFF;"
        "strokeColor=#C7D3E0;strokeWidth=1.3;")


def build():
    for p, n in ((P1, 1), (P2, 2), (P3, 3), (P4, 4)):
        panel(p, n)
    for xa in (P1['x'] + P1['w'], P2['x'] + P2['w'], P3['x'] + P3['w']):
        E(xa + 4, 320, xa + 34, 320, SLATE, 4.0)

    # ---------------- P1: benchmark ----------------
    x, iw = P1["x"] + 20, P1["w"] - 40
    cw = (iw - 14) // 2
    for dx, icon, t1, t2 in ((0, "doc_stack", "SEC filings", "10-K/Q &#183; 8-K"),
                             (cw + 14, "db_prices", "CRSP prices", "daily returns")):
        V(x + dx, 116, cw, 134, CHIP)
        IC(icon, x + dx + (cw - 52) // 2, 122, 52)
        T(x + dx, 180, cw, 28, t1, BODY, INK, True, "center")
        T(x + dx, 210, cw, 28, t2, BODY, SUB, False, "center")
    IC("link_chain", x + 2, 272, 30)
    T(x + 40, 268, iw - 40, 28, "PERMNO&#8594;CIK", BODY, INK, True)
    T(x + 40, 268 + PITCH, iw - 40, 28, "time-varying", BODY, SUB)
    hw = (iw - 10) // 2
    T(x, 348, hw, 46, "144,129", 42, P1["tc"], True, "center")
    T(x, 396, hw, 28, "filings", BODY, SUB, False, "center")
    T(x + hw + 10, 348, hw, 46, "431,245", 42, P1["tc"], True, "center")
    T(x + hw + 10, 396, hw, 28, "aligned rows", BODY, SUB, False, "center")
    E(x + 58, 450, x + iw - 58, 450, SLATE, 2.2, end="none")
    V(x + 54, 445, 10, 10, f"ellipse;fillColor={SLATE};strokeColor=none;")
    V(x + iw - 64, 445, 10, 10, f"ellipse;fillColor={SLATE};strokeColor=none;")
    T(x - 6, 437, 58, 28, "2010", BODY, INK, True)
    T(x + iw - 52, 437, 58, 28, "2025", BODY, INK, True, "right")
    T(x, 470, iw, 28, "features end before filing", BODY, SUB, False, "center")
    V(x, 512, iw, 42, "rounded=1;absoluteArcSize=1;arcSize=10;fillColor=#E9F5EE;strokeColor=#9CCDB2;")
    T(x, 519, iw, 30, "&#10003; 0 look-ahead violations", BODY, GREEN, True, "center")

    # ---------------- P2: task & challengers ----------------
    x, iw = P2["x"] + 16, P2["w"] - 32
    IC("doc_stack", x + 4, 118, 32)
    T(x + 38, 122, 20, 28, "+", HEAD, SLATE, True, "center")
    IC("chart_price", x + 60, 118, 32)
    E(x + 100, 134, x + 128, 134, SLATE, 2.2)
    V(x + 134, 112, iw - 134, 46, CHIP)
    T(x + 134, 115, iw - 134, 40, "forecast &#963;", BODY, INK, True, "center", "middle")
    T(x, 176, iw, 28, "h = 5 / 10 / 20 days", BODY, INK, False, "center")
    T(x, 214, iw, 30, "TEXT challengers", BODY, P2["tc"], True)
    chips = (("BoW &#183; dictionaries", None), ("fine-tuned FinBERT", "flame"),
             ("frozen 7&#8211;8B embed.", "snowflake"), ("prompted 32B LLM", "robot_prompt"))
    for i, (label, icon) in enumerate(chips):
        cy = 248 + i * 44
        V(x, cy, iw, 40, CHIP)
        if icon:
            IC(icon, x + 8, cy + 8, 24)
        T(x + 40, cy + 3, iw - 48, 34, label, BODY, INK, False, "left", "middle")
    T(x, 434, iw, 30, "PRICE references", BODY, P2["tc"], True)
    V(x, 468, iw, 98, CHIP)
    for i, line in enumerate(("HAR-RV &#183; SHAR", "GARCH &#183; EGARCH",
                              "ARIMA &#183; HAR-X")):
        T(x, 476 + i * 30, iw, 28, line, BODY, INK, False, "center")

    # ---------------- P3: the identity audit ----------------
    x, iw = P3["x"] + 16, P3["w"] - 32
    V(x, 118, 182, 58, CHIP)
    IC("doc_stack", x + 10, 128, 38)
    T(x + 52, 121, 124, 52, "one filing", BODY, INK, True, "center", "middle")
    trap_x = x + iw - 224
    V(trap_x, 112, 224, 82,
      "rounded=1;absoluteArcSize=1;arcSize=10;fillColor=#FDF3EC;"
      f"strokeColor={VERM};strokeWidth=1.6;dashed=1;dashPattern=7 5;")
    IC("flag_trap", trap_x + 10, 136, 28)
    T(trap_x + 44, 122, 172, 28, "apparent gain", BODY, VERM, True, "center")
    T(trap_x + 44, 122 + PITCH, 172, 28, "up to +5.9%", BODY, VERM, False, "center")
    E(x + 188, 147, trap_x - 6, 147, SLATE, 2.8)
    T(x + 182, 108, trap_x - x - 182, 30, "what it says", BODY, SLATE, True, "center")
    E(x + 188, 179, trap_x - 6, 179, VERM, 2.8, dashed=True)
    T(x + 182, 188, trap_x - x - 182, 30, "who filed it", BODY, VERM, True, "center")
    V(x, 226, 240, 38, "rounded=1;absoluteArcSize=1;arcSize=10;fillColor=#EFF2F6;strokeColor=#C7D3E0;")
    T(x, 232, 240, 30, "standalone 0 / 180", BODY, SUB, False, "center")

    T(x, 282, iw, 32, "reference ladder &#8594; Holm survivors of 69", HEAD, P3["tc"], True)
    bars = (("recalibrated HAR", 38, "38"), ("firm identity", 8, "8"),
            ("maximal pool", 9, "9"), ("full conjunction", 0, "0 / 69"))
    lw, bx = 222, x + 230
    for i, (label, v, tag) in enumerate(bars):
        by = 324 + i * 33
        T(x, by - 3, lw, 30, label, BODY, INK, i == 3, "right")
        blen = max(6, v / 38 * 216)
        fill = P3["tc"] if v else INK
        V(bx, by, blen, 22, f"rounded=0;fillColor={fill};strokeColor=none;opacity={90 - i * 12};")
        T(bx + blen + 10, by - 4, 116, 30, tag, HEAD, INK, True)
    T(bx + 108, 420, iw - 338, 28, "disjoint sets", BODY, SUB)

    ccw = (iw - 12) // 2
    for dx, icon, title, l1, l2 in (
            (0, "mask", "ANONYMISE", "&#8216;Apple &#183; AAPL&#8217;", "identity share &#8804; 0.51"),
            (ccw + 12, None, "MATCHED SWAP", "A &#8646; B, matched RV", "residual &#8722;84&#8211;93%")):
        cx = x + dx
        V(cx, 458, ccw, 100, CHIP)
        if icon:
            IC(icon, cx + 10, 488, 30)
        else:
            E(cx + 12, 496, cx + 38, 496, SLATE, 2.4)
            E(cx + 38, 516, cx + 12, 516, SLATE, 2.4)
        T(cx + 46, 466, ccw - 52, 30, title, BODY, P3["tc"], True)
        T(cx + 46, 496, ccw - 52, 30, l1, BODY, INK)
        T(cx + 46, 526, ccw - 52, 30, l2, BODY, INK, True)
    V(cx - ccw - 12 + 232, 500, 36, 22, f"rounded=0;fillColor={INK};strokeColor=none;")

    # ---------------- P4: gates & verdict ----------------
    x, iw = P4["x"] + 16, P4["w"] - 32
    gates = (("scale", "clustered DM + Holm"), ("shuffle", "label-shuffle placebo"),
             ("syringe_gauge", "injection &#183; MDE"), ("fingerprint", "identity battery &#215;4"))
    for i, (icon, label) in enumerate(gates):
        gy = 118 + i * 44
        IC(icon, x, gy + 8, 22)
        T(x + 30, gy + 2, iw - 54, 34, label, BODY, INK, False, "left", "middle")
        T(x + iw - 24, gy + 2, 24, 34, "&#10003;", BODY, GREEN, True, "center", "middle")
    E(x, 306, x + iw, 306, "#C4B7E8", 1.6, end="none")
    T(x, 316, iw, 58, "0 / 69", 54, P4["tc"], True, "center")
    T(x, 376, iw, 30, "survive full conjunction", BODY, INK, False, "center")
    V(x, 416, iw, 146, f"rounded=1;absoluteArcSize=1;arcSize=10;fillColor=#FFFFFF;"
                       f"strokeColor={GREEN};strokeWidth=2;")
    for i, (s, col, bold) in enumerate((("yet: 8-K residual", GREEN, True),
                                        ("+0.21/+0.18/+0.20 %", INK, True),
                                        ("mask-robust at h=5", INK, False),
                                        ("bounded &#183; &#916;Sharpe &#8776; 0", SUB, False))):
        T(x + 14, 426 + i * PITCH, iw - 28, 30, s, BODY, col, bold)


def main():
    build()
    xml = ('<mxfile><diagram name="fig1"><mxGraphModel dx="0" dy="0" grid="0" '
           'page="0" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + "".join(_cells) + "</root></mxGraphModel></diagram></mxfile>")
    out = FIG / "figure1_vnext.drawio"
    out.write_text(xml)
    print(f"wrote {out} ({len(xml)//1024}KB, {_n} cells)")
    cli = "/Applications/draw.io.app/Contents/MacOS/draw.io"
    subprocess.run([cli, "-x", "-f", "pdf", "--crop", "-o",
                    str(FIG / "figure1_vnext_raw.pdf"), str(out)], check=True,
                   capture_output=True)
    subprocess.run([cli, "-x", "-f", "png", "-s", "1.2", "-o",
                    str(FIG / "figure1_vnext_preview.png"), str(out)], check=True,
                   capture_output=True)
    print("exported pdf + png preview")


if __name__ == "__main__":
    main()
