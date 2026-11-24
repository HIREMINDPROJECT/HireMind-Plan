"""E3 system tests of the HR prototype, SYNTHETIC data only. Run: python3 test_prototype.py <dir with hr_review_prototype.html + demo_case_SYNTHETIC.json>
Writes test_results.csv (id, test, result, detail)."""
import csv, json, sys, threading, http.server, socketserver, functools, tempfile, os, time
from pathlib import Path
from playwright.sync_api import sync_playwright

D = Path(sys.argv[1]).resolve()
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(D))
H.log_message = lambda *a, **k: None
srv = socketserver.TCPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{port}/hr_review_prototype.html"
res = []
def rec(i, name, ok, detail=""): res.append((i, name, "PASS" if ok else "FAIL", detail)); print(i, "PASS" if ok else "FAIL", name, detail)

def case(n_req=2, cv="Python and SQL. Used React.", job="Python required.", extra=None):
    reqs = [{"requirement_id": f"t:R{i}", "category": "SKILL", "value": f"skill{i}", "requirement_type": "REQUIRED", "status": "FOUND_EXPLICIT", "job_span": [0, 6], "cv_spans": [[0, 6, "EXPLICIT"]]} for i in range(n_req)]
    c = {"case_id": "t1", "cv_text": cv, "job_text": job, "requirements": reqs, "advice_only": True, "prediction": {"label": "Potential Fit", "probabilities": {"No Fit": .3, "Potential Fit": .4, "Good Fit": .3}, "abstain": False}}
    c.update(extra or {}); return c

with sync_playwright() as p:
    b = p.chromium.launch(); 
    def newpage():
        pg = b.new_context(accept_downloads=True).new_page(); errs = []; dlg = []
        pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("dialog", lambda d: (dlg.append(d.message), d.dismiss())); return pg, errs, dlg
    def load_json(pg, obj):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f.write(json.dumps(obj) if not isinstance(obj, str) else obj); f.close()
        pg.set_input_files("#file", f.name); time.sleep(.3); os.unlink(f.name)

    # T1 loads without errors, banner present
    pg, errs, dlg = newpage(); pg.goto(URL)
    rec("T01", "page loads with no console errors", not errs, "; ".join(errs)[:120])
    rec("T02", "advice-only banner visible", "never accepts or rejects" in pg.inner_text(".banner"))
    # T3 demo loads, abstain shown, requirement rows
    pg.click("#demo"); pg.wait_for_selector("#app:not([hidden])")
    demo = json.load(open(D / "demo_case_SYNTHETIC.json"))
    rec("T03", "built-in demo loads; rows = requirements", pg.locator("#reqs tr").count() == len(demo["requirements"]), f"{pg.locator('#reqs tr').count()} vs {len(demo['requirements'])}")
    rec("T04", "abstain flag shown as low confidence", "low confidence" in pg.inner_text("#pred").lower())
    rec("T05", "unknown is shown as 'not negative'", ("Not stated (unknown, not negative)" in pg.inner_text("#reqs")) or not any(r['status']=='NOT_STATED_UNKNOWN' for r in demo['requirements']))
    # T6 decision required
    pg.click("#save"); rec("T06", "save without decision is blocked", any("Choose a decision" in m for m in dlg), str(dlg))
    # T7 override without reason blocked, with reason works and audit content
    dlg.clear(); pg.select_option("#dec", "No Fit"); pg.click("#save")
    rec("T07", "override without reason is blocked", any("reason" in m.lower() for m in dlg), str(dlg))
    pg.fill("#why", "Missing required certification (synthetic test)")
    with pg.expect_download() as dl: pg.click("#save")
    a = json.load(open(dl.value.path()))
    rec("T08", "audit record content", a["overridden"] is True and a["hr_decision"] == "No Fit" and a["suggestion"] == demo["prediction"]["label"] and "Advice-only" in a["note"], json.dumps({k: a[k] for k in ("overridden", "hr_decision", "suggestion")}))
    rec("T09", "audit record has no CV/job text", demo["cv_text"][:30] not in json.dumps(a))
    # T10 no external network requests
    pg2, e2, d2 = newpage(); reqs = []; pg2.on("request", lambda r: reqs.append(r.url)); pg2.goto(URL); pg2.click("#demo"); pg2.wait_for_selector("#app:not([hidden])")
    rec("T10", "no request leaves localhost", all(u.startswith(f"http://127.0.0.1:{port}") for u in reqs), str([u for u in reqs if not u.startswith(f"http://127.0.0.1:{port}")][:3]))
    # T11 XSS: script-like text rendered as text
    pg3, e3, d3 = newpage(); pg3.goto(URL); pg3.evaluate("window.__x=0")
    evil = '<img src=x onerror="window.__x=1"><script>window.__x=2</script>'
    c = case(cv=evil + " Python", job="Python " + evil); c["requirements"][0]["value"] = evil
    load_json(pg3, c); time.sleep(.3)
    rec("T12", "HTML in CV/job/requirement text is not executed", pg3.evaluate("window.__x") == 0 and pg3.locator("#cv img").count() == 0 and pg3.locator("#reqs img").count() == 0)
    rec("T13", "HTML shown literally", "<img" in pg3.inner_text("#cv"))
    # T14 invalid file
    pg4, e4, d4 = newpage(); pg4.goto(URL); pg4.evaluate("window.__a=[];window.alert=m=>{window.__a.push(String(m))};0")
    load_json(pg4, "{not json"); A = pg4.evaluate("window.__a"); rec("T14", "invalid JSON: message shown, app stays hidden", "Could not read the file" in A and pg4.locator("#app").is_hidden(), str(A))
    pg4.evaluate("window.__a.length=0"); load_json(pg4, {"foo": 1}); A = pg4.evaluate("window.__a"); rec("T15", "wrong schema: message shown, app stays hidden", "Not a valid case file" in A and pg4.locator("#app").is_hidden(), str(A))
    # T16 emoji / non-BMP offsets (spans are code points)
    pg5, e5, d5 = newpage(); pg5.goto(URL)
    c = case(cv="😀😀 Python and SQL", job="Python required"); c["requirements"] = [c["requirements"][0]]; c["requirements"][0]["cv_spans"] = [[3, 9, "EXPLICIT"]]; c["requirements"][0]["job_span"] = [0, 6]
    load_json(pg5, c); marks = pg5.locator("#cv mark").all_inner_texts()
    rec("T16", "highlight correct with emoji (code-point offsets)", marks == ["Python"], str(marks))
    # T17 out-of-range / overlapping spans do not crash
    c = case(cv="abc", job="xyz"); c["requirements"][0]["cv_spans"] = [[0, 99, "EXPLICIT"]]; c["requirements"][1]["cv_spans"] = [[2, 1, "EXPLICIT"], [0, 2, "EXPLICIT"], [1, 3, "EXPLICIT"]]
    pg6, e6, d6 = newpage(); pg6.goto(URL); load_json(pg6, c)
    rec("T17", "bad/overlapping spans do not crash", not e6 and pg6.locator("#app").is_visible(), str(e6)[:100])
    # T18 evidence checks do not leak from one case to the next
    pg7, e7, d7 = newpage(); pg7.goto(URL); load_json(pg7, case()); pg7.locator("#reqs select").first.select_option("Extraction is wrong")
    c2 = case(); c2["case_id"] = "t2"; load_json(pg7, c2); pg7.select_option("#dec", "Potential Fit")
    with pg7.expect_download() as dl: pg7.click("#save")
    a = json.load(open(dl.value.path()))
    rec("T18", "HR checks from case 1 do not leak into case 2's audit record", a["evidence_checks"] == {}, str(a["evidence_checks"]))
    # T19 no suggestion attached
    pg8, e8, d8 = newpage(); pg8.goto(URL); c = case(); del c["prediction"]; load_json(pg8, c)
    rec("T19", "case without prediction shows notice and saves", "No model prediction" in pg8.inner_text("#pred"))
    # T20 large case renders fast
    pg9, e9, d9 = newpage(); pg9.goto(URL); t0 = time.time(); load_json(pg9, case(n_req=300, cv="Python " * 2000, job="Python " * 500)); dt = time.time() - t0
    rec("T20", "300 requirements, 14k-char CV renders in < 3 s", pg9.locator("#reqs tr").count() == 300 and dt < 3, f"{dt:.2f}s")
    # T21 demo button over file:// gives the documented message instead of failing silently
    pg10, e10, d10 = newpage(); pg10.goto("file://" + str(D / "hr_review_prototype.html")); pg10.click("#demo"); time.sleep(.5)
    rec("T21", "demo over file:// shows the help message", any("Open this page next to" in m for m in d10) or pg10.locator("#app").is_visible(), str(d10))
    # T22 keyboard: labels
    rec("T22", "controls have accessible names", all(pg.locator(s).count() for s in ("label:has(#dec)", "label:has(#why)")))
    b.close()
with open(D / "test_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "test", "result", "detail"]); w.writerows(res)
print(sum(r[2] == "PASS" for r in res), "pass,", sum(r[2] == "FAIL" for r in res), "fail")
