#!/usr/bin/env python3
"""Figure 1 vNext skeleton — playbook-spec rebuild (four tinted panels, numbered
tabs, two-layer captioning, labeled arrows, the LADDER as hero object, NAIVE
off-ladder trap). Icon slots render as dashed placeholders until assets land in
figures/icons/ (then rerun: slots auto-fill, PNG or SVG).
Targets: <=130 in-figure words; fonts >=25px (7.04pt at \textwidth); viewBox
1800x531 (~2.08in print) — verified: body still ends on p7, References open p8.
"""
import os, re, subprocess, tempfile, base64

FIG = "writing/paper/figures"
ICO = f"{FIG}/icons"
INK, SUB, SLATE = "#132238", "#5F6F82", "#2F3B4C"
RED = "#C0392B"; GREEN = "#1E7A4E"   # ✓ renders as TEXT: 4.97:1 on the violet
                                     # tint (was #248A5A = 4.04:1 < 4.5 floor)
PANELS = [
    dict(t="Point-in-time benchmark", sub="survivorship-free construction",
         tint="#F4F8FE", stroke="#9DBFE8", tc="#2F6FB2"),
    dict(t="Task & model spectrum", sub="one model per horizon",
         tint="#F2FBF8", stroke="#9FD8CA", tc="#0D806A"),
    dict(t="Reference ladder", sub="credit text only beyond controls",
         tint="#FFF8EF", stroke="#F0BE83", tc="#AC5E03"),
    dict(t="Inference & validity gates", sub="pre-registered, power-calibrated",
         tint="#F8F6FF", stroke="#C4B7E8", tc="#6D55B7"),
]
W,H = 1800, 531
PW,PH = 424, 471; GAP=20; MX=(W-4*PW-3*GAP)/2; PY=34
FT_T, FT_S, FT_B, FT_F = 28, 25, 25, 25   # floor 25px = 7.04pt at textwidth (1px=0.2816pt)
LH = 33
E=[]
def txt(x,y,s,size=FT_B,fill=INK,w="normal",anchor="start",extra=""):
    s=str(s).replace("&","&amp;").replace("<","&lt;")
    E.append(f'<text x="{x:.0f}" y="{y:.0f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{extra}>{s}</text>')

COLORFUL_ICONS = True   # author direction 2026-07-20: rich full-color icons
                        # (exemplar style); geometry-only normalisation, colors kept.

def _norm_png(path, name):
    """Build-time icon normalisation (originals untouched): crop to content bbox,
    square-pad 6%, downsample to 256px (raw 1024px x 1.4MB each would bloat the
    PDF ~20MB). When COLORFUL_ICONS is False, additionally flatten to the family
    slate (#2F3B4C; crimson for flag_trap) — legacy line-art mode."""
    from PIL import Image
    import io
    im=Image.open(path).convert("RGBA")
    a=im.split()[3]
    if not COLORFUL_ICONS:
        tgt=(192,57,43) if name=="flag_trap" else (47,59,76)
        solid=Image.new("RGBA", im.size, tgt+(255,))
        solid.putalpha(a)
        im=solid
    bb=a.getbbox() or (0,0,im.size[0],im.size[1])
    im=im.crop(bb)
    side=int(max(im.size)*1.12)
    sq=Image.new("RGBA",(side,side),(0,0,0,0))
    sq.paste(im,((side-im.size[0])//2,(side-im.size[1])//2),im)
    sq=sq.resize((256,256),Image.LANCZOS)
    buf=io.BytesIO(); sq.save(buf,"PNG",optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def icon(name,x,y,s):
    """Embed icons/<name>.(svg|png) at (x,y) box s x s, else dashed slot."""
    for ext in ("svg","png"):
        p=f"{ICO}/{name}.{ext}"
        if os.path.exists(p):
            if ext=="png":
                b=_norm_png(p,name)
                E.append(f'<image x="{x:.0f}" y="{y:.0f}" width="{s}" height="{s}" href="data:image/png;base64,{b}"/>')
            else:
                inner=open(p).read()
                inner=re.sub(r'<\?xml[^>]*\?>','',inner)
                m=re.search(r'viewBox="([\d.\s-]+)"', inner)
                vb=m.group(1) if m else "0 0 1024 1024"
                body=re.sub(r'</?svg[^>]*>','',inner)
                E.append(f'<svg x="{x:.0f}" y="{y:.0f}" width="{s}" height="{s}" viewBox="{vb}">{body}</svg>')
            return
    E.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{s}" height="{s}" rx="6" fill="none" stroke="{SUB}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    E.append(f'<text x="{x+s/2:.0f}" y="{y+s/2+6:.0f}" font-family="Arial" font-size="15" fill="{SUB}" text-anchor="middle">{name}</text>')

# ---------------- panels, tabs ----------------
# Crowding-audit redesign (2026-07-20, 4x visual audit + geometry quant):
# one redundant line deleted per panel funds a two-tier rhythm (tight within
# pairs, >=0.8 line-height between blocks); italic footers dropped (read as an
# extra list line and sat on the border); no height increase needed.
for i,p in enumerate(PANELS):
    px=MX+i*(PW+GAP)
    E.append(f'<rect x="{px:.0f}" y="{PY}" width="{PW}" height="{PH}" rx="16" fill="{p["tint"]}" stroke="{p["stroke"]}" stroke-width="1.7"/>')
    # header fully inside the frame (author decision 2026-07-20 A/B): border
    # stays closed — panels read as four intact cards, no knockout stubs
    E.append(f'<rect x="{px+16:.0f}" y="{PY+12}" width="36" height="30" rx="9" fill="{p["tc"]}"/>')
    txt(px+34, PY+34, str(i+1), 25, "#FFFFFF", "bold", "middle")
    txt(px+60, PY+35, p["t"], FT_T, p["tc"], "bold")
    txt(px+22, PY+68, p["sub"], FT_S, SUB)
    if i<3:
        # flow arrows: enlarged + darkened (exemplar audit: faint gutter
        # triangles nearly vanish at print; gutter is 20 wide, arrow spans 18)
        ax=px+PW+GAP/2
        E.append(f'<path d="M {ax-9:.0f} {PY+PH/2-20} L {ax+7:.0f} {PY+PH/2} L {ax-9:.0f} {PY+PH/2+20} Z" fill="{SLATE}"/>')

# ---------------- Panel 1: benchmark ----------------
# ("features end strictly before filing" deleted: mechanism carried by the
# timeline+lock and the no-look-ahead pill; stat block cleared of the db icon)
px=MX
icon("doc_stack", px+24, PY+108, 50); txt(px+86, PY+132, "SEC filings 10-K/10-Q/8-K")
icon("db_prices", px+24, PY+168, 50); txt(px+86, PY+192, "CRSP daily prices")
y=PY+264
txt(px+24, y, "144,129", 38, PANELS[0]["tc"], "bold"); txt(px+24, y+30, "filings", 25, SUB)
txt(px+230, y, "431,245", 38, PANELS[0]["tc"], "bold"); txt(px+230, y+30, "aligned rows", 25, SUB)
y=PY+340
E.append(f'<line x1="{px+86}" y1="{y}" x2="{px+302}" y2="{y}" stroke="#5A8FD0" stroke-width="4" stroke-linecap="round"/>')
for cx in (px+86, px+158, px+230, px+302):
    E.append(f'<circle cx="{cx}" cy="{y}" r="5" fill="#2F6FB2"/>')
txt(px+74, y+8, "2010", 25, SUB, "600", "end"); txt(px+314, y+8, "2025", 25, SUB, "600")
icon("lock_clock", px+380, y-18, 34)
icon("link_chain", px+24, PY+372, 34); txt(px+68, PY+396, "PERMNO→CIK, time-varying", FT_B, INK)
E.append(f'<rect x="{px+20}" y="{PY+PH-50}" width="{PW-40}" height="36" rx="10" fill="#E9F2FD" stroke="#9DBFE8"/>')
txt(px+PW/2, PY+PH-23, "✓ no-look-ahead: 0 violations", 25, PANELS[0]["tc"], "bold", "middle")

# ---------------- Panel 2: task & models ----------------
# (top dead band compressed; two-tier rhythm: rows at 40, >=0.9em before each
# green header; footer deleted)
px=MX+(PW+GAP); y=PY+106
E.append(f'<path d="M {px+28} {y+30} L {px+62} {y+12} L {px+92} {y+36} L {px+122} {y+4} L {px+150} {y+26}" fill="none" stroke="{SLATE}" stroke-width="3" stroke-linecap="round"/>')
txt(px+170, y+26, "h ∈ {5, 10, 20} days", FT_B, INK)
y=PY+180
txt(px+24, y, "TEXT", 25, PANELS[1]["tc"], "bold", "start", ' letter-spacing="1"')
# inline word-list glyph fills the BoW row's icon slot (no PNG asset left)
for k,wl in enumerate((18,24,14)):
    E.append(f'<rect x="{px+31}" y="{y+22+k*8}" width="{wl}" height="4" rx="2" fill="{SLATE}"/>')
txt(px+68, y+38, "BoW · dictionaries", FT_B)
icon("flame", px+26, y+56, 32);       txt(px+68, y+78, "fine-tuned encoders", FT_B)
icon("snowflake", px+26, y+96, 32);   txt(px+68, y+118, "frozen 7–8B embeddings", FT_B)
icon("robot_prompt", px+23, y+132, 38); txt(px+68, y+158, "prompted 32B LLM", FT_B)
y=PY+386
txt(px+24, y, "PRICE", 25, PANELS[1]["tc"], "bold", "start", ' letter-spacing="1"')
icon("chart_price", px+26, y+14, 32)
txt(px+68, y+38, "HAR-RV · SHAR · GARCH", FT_B)
txt(px+68, y+68, "EGARCH · ARIMA · VIX", FT_B)

# ---------------- Panel 3: the ladder (hero) ----------------
# ("disjoint sets" intro line deleted — stated in Results and the Fig.2 caption;
# rung label/count pairs opened to 28; bars start clear of the text and sit
# 4 below the count baseline; NAIVE box taller with real interior padding,
# dashed arrow dropped — vertical adjacency already implies the link)
px=MX+2*(PW+GAP)
LC=PANELS[2]["tc"]
# no "+" prefixes: the rungs are SEPARATE references (disjoint survivor sets),
# not cumulative nesting — R25 cold read flagged the "+" as implying the latter
RUNGS=[("recalibrated HAR","38/69"),("firm identity","8/69"),
       ("maximal price pool","9/69"),("full conjunction","0/69")]
bx,by=px+254, PY+366; sw,sh=30,68
for n,(lab,cnt) in enumerate(RUNGS):
    rx,ry=bx+n*sw, by-n*sh
    E.append(f'<path d="M {rx} {ry+4} L {rx+sw+30} {ry+4}" stroke="{LC}" stroke-width="5" stroke-linecap="round"/>')
    if n<3:
        E.append(f'<path d="M {rx+sw+30} {ry+4} L {rx+sw+30} {ry+4-sh}" stroke="{LC}" stroke-width="3" stroke-linecap="round"/>')
    txt(rx-14, ry-30, lab, 25, INK, "bold" if n==3 else "normal", "end")
    txt(rx-14, ry-2, cnt+" survive", 25, SUB, "normal", "end")
E.append(f'<rect x="{px+20}" y="{PY+PH-86}" width="316" height="72" rx="10" fill="#F4F4F4" stroke="{SUB}" stroke-width="1.4" stroke-dasharray="7 5"/>')
icon("flag_trap", px+30, PY+PH-71, 30)
txt(px+68, PY+PH-59, "NAIVE: raw baseline", 25, RED, "bold")
txt(px+68, PY+PH-31, "→ “apparent text gain”", 25, RED)

# ---------------- Panel 4: gates ----------------
# (checkmark rail pulled off the border; rows at even 54 pitch, list re-centred
# in the freed band; pill kept as the P1-symmetric verdict badge, footer deleted)
px=MX+3*(PW+GAP); y=PY+118
GATES=[("scale","day-clustered DM + Holm"),("shuffle","label-shuffle placebo"),
       ("syringe_gauge","injection + per-cell MDE"),("mask","anonymisation · swap"),
       ("fingerprint","identity share 0.51")]
for n,(ic,lab) in enumerate(GATES):
    gy=y+n*54
    icon(ic, px+24, gy-2, 38)
    txt(px+74, gy+24, lab, FT_B, INK)
    txt(px+PW-38, gy+24, "✓", 26, GREEN, "bold")
E.append(f'<rect x="{px+20}" y="{PY+PH-50}" width="{PW-40}" height="36" rx="10" fill="#EEEAFB" stroke="#C4B7E8"/>')
txt(px+PW/2, PY+PH-23, "what survives: 8-K residual", 25, PANELS[3]["tc"], "bold", "middle")

DEFS='<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#5F6F82"/></marker></defs>'
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{DEFS}'+"".join(E)+'</svg>'
open(f"{FIG}/figure1_vnext.svg","w").write(svg)
tmp=tempfile.mktemp(suffix=".pdf")
subprocess.run(["rsvg-convert","-f","pdf","-o",tmp,f"{FIG}/figure1_vnext.svg"],check=True)
GS="/usr/local/bin/gs" if os.path.exists("/usr/local/bin/gs") else "gs"
subprocess.run([GS,"-o",f"{FIG}/figure1_vnext.pdf","-sDEVICE=pdfwrite","-dNoOutputFonts","-dQUIET","-dBATCH","-dNOPAUSE",tmp],check=True)
os.unlink(tmp)
subprocess.run(["rsvg-convert","-w","2400",f"{FIG}/figure1_vnext.svg","-o","/tmp/vnext.png"],check=True)
print("vnext skeleton written")
