"""md -> pdf with the same simple style as the earlier model card. usage: python md2pdf.py in.md out.pdf"""
import sys, markdown
from playwright.sync_api import sync_playwright
css = "body{font:11px/1.45 Arial;margin:24px}h1{font-size:17px}h2{font-size:13.5px;margin:13px 0 4px}h3{font-size:12px}table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #bbb;padding:3px 6px;vertical-align:top}th{background:#f2f2f2}li{margin:2px 0}code{font-size:10px}"
src = open(sys.argv[1]).read()
if src.startswith("---"):
    head, src = src[3:].split("\n---", 1)
    t = [l.split(":", 1)[1].strip().strip('"') for l in head.splitlines() if l.startswith("title:")]
    src = ("# " + t[0] + "\n" if t else "") + src
html = "<!doctype html><meta charset=utf-8><style>" + css + "</style>" + markdown.markdown(src, extensions=["tables"])
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.set_content(html); pg.pdf(path=sys.argv[2], format="A4", margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"}); b.close()
print("pdf", sys.argv[2])
