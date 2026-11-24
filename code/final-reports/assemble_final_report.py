"""Fills FINAL_REPORT_TEMPLATE.md -> FINAL_REPORT.md (+ PDF with the diagram embedded).
usage: python assemble_final_report.py fills.json results_report.md diagram.png out_dir"""
import base64, json, re, sys
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright
here = Path(__file__).parent
fills = json.loads(Path(sys.argv[1]).read_text())
res = Path(sys.argv[2]).read_text()
table = res.split("## 1. All models", 1)[1].split("## 2.", 1)[0].strip()
rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in table.splitlines()[2:] if l.startswith("|")]
GROUPS = [("Classical, structured and semantic baselines", lambda g: g in ("Classical", "Structured", "Semantic")),
          ("BERT-Tiny + Knowledge v1 (the plan's main line)", lambda g: g.startswith("BERT-Tiny")),
          ("DistilBERT (M5 plain, M5 + Knowledge v1 through the plan, M8 hybrid)", lambda g: g.startswith("DistilBERT"))]
parts = ["All numbers on validation; mean ± SD over seeds (one seed has no SD). Full class-wise metrics and confusion matrices are in FINAL_RESULTS_REPORT.md.", ""]
for title, test in GROUPS:
    parts += [f"**{title}**", "", "| Model | Labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |", "|---|---|---|---|---|---|---|---|---|"]
    parts += ["| " + " | ".join(r[1:]) + " |" for r in rows if test(r[0])] + [""]
fills["RESULTS_TABLE"] = "\n".join(parts)
s = (here / "FINAL_REPORT_TEMPLATE.md").read_text()
for k, v in fills.items():
    assert "{{" + k + "}}" in s, k
    s = s.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z_]+\}\}", s); assert not left, left
out = Path(sys.argv[4]); (out / "FINAL_REPORT.md").write_text(s)
import io
from PIL import Image
_buf = io.BytesIO(); Image.open(sys.argv[3]).save(_buf, "PNG")
img = base64.b64encode(_buf.getvalue()).decode()
html_md = s.replace("![HireMind pipeline](HireMind_pipeline_diagram.png)", "<div class=fig><img src=\"data:image/png;base64," + img + "\"><div class=cap>Figure 1 (appendix). The HireMind pipeline (advice only): inputs, extraction, matching, classifier, HR review. Also in 00_overview/HireMind_pipeline_diagram.pdf.</div></div>")
css = ("body{font:14.5px/1.6 Arial,Helvetica,sans-serif;color:#1a1a1a;margin:0}"
       "h1{font-size:27px;line-height:1.25;margin:0 0 10px}"
       "h1+p{color:#555;font-size:14px;margin-bottom:22px}"
       "h2{font-size:20px;margin:30px 0 10px;padding-bottom:5px;border-bottom:2px solid #d0d7de;break-after:avoid}"
       "h3{font-size:17px;margin:22px 0 8px;break-after:avoid}"
       "p{margin:8px 0 12px}ul{margin:6px 0 14px;padding-left:24px}li{margin:6px 0}"
       "table{border-collapse:collapse;margin:12px 0 18px;font-size:12.5px;line-height:1.4;width:100%}"
       "td,th{border:1px solid #c4c4c4;padding:6px 8px;vertical-align:top;text-align:left}th{background:#eef2f6}"
       "tr{break-inside:avoid}tr:nth-child(even) td{background:#fafbfc}"
       "img{max-width:100%;display:block;margin:16px auto}strong{color:#000}"
       "table.num td{white-space:nowrap}table.num td:first-child{white-space:normal;min-width:120px}table.num th{white-space:normal}table.num{font-size:12px}"
       ".sd{display:block;color:#666;font-size:11px}"
       "table{break-inside:avoid}p:has(+table){break-after:avoid}"
       ".fig{break-before:page;text-align:center}.fig img{max-height:238mm;max-width:100%}.cap{font-size:13px;color:#444;margin-top:8px}")
body = markdown.markdown(html_md, extensions=["tables"])
body = re.sub(r"<table>(\s*<thead>\s*<tr>(?:(?!</tr>).)*?Accuracy)", r"<table class=num>\1", body, flags=re.S)
body = re.sub(r"<table class=num>.*?</table>", lambda m: re.sub(r"(\d) ± (\d+\.\d+)", r"\1<span class=sd>± \2</span>", m.group(0)), body, flags=re.S)
_fig = re.search(r"<p><div class=fig>.*?</div></div></p>|<div class=fig>.*?</div></div>", body, flags=re.S)
if _fig:
    body = body.replace(_fig.group(0), "<p><em>The full pipeline is shown in Figure 1 on the last page.</em></p>")
    body = body + _fig.group(0).replace("break-after:page", "")
html = "<!doctype html><meta charset=utf-8><style>" + css + "</style>" + body
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.set_content(html)
    pg.pdf(path=str(out / "FINAL_REPORT.pdf"), format="A4", margin={"top": "18mm", "bottom": "17mm", "left": "20mm", "right": "20mm"},
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='font-size:9px;width:100%;text-align:center;color:#888'>HireMind final report · <span class=pageNumber></span>/<span class=totalPages></span></div>")
    b.close()
print("written", out / "FINAL_REPORT.md", out / "FINAL_REPORT.pdf")
