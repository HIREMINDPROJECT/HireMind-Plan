"""E3 system tests for HR prototype v2 (SYNTHETIC data only). Run: python3 test_prototype_v2.py <dir>. Writes test_results_v2.csv."""
import csv, json, sys, threading, http.server, socketserver, functools, tempfile, os, time
from pathlib import Path
from playwright.sync_api import sync_playwright
D = Path(sys.argv[1]).resolve()
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(D)); H.log_message = lambda *a, **k: None
srv = socketserver.TCPServer(("127.0.0.1", 0), H); port = srv.server_address[1]; threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{port}/hr_review_prototype.html"; res = []
def rec(i, name, ok, detail=""): res.append((i, name, "PASS" if ok else "FAIL", detail)); print(i, "PASS" if ok else "FAIL", name, detail)
def case(n_req=2, cv="Python and SQL. Used React.", job="Python required.", cid="t1"):
    reqs = [{"requirement_id": f"{cid}:R{i}", "category": "SKILL", "value": f"skill{i}", "requirement_type": "REQUIRED", "status": "FOUND_EXPLICIT" if i % 2 == 0 else "NOT_STATED_UNKNOWN", "job_span": [0, 6], "cv_spans": [[0, 6, "EXPLICIT"]] if i % 2 == 0 else []} for i in range(n_req)]
    return {"case_id": cid, "cv_text": cv, "job_text": job, "requirements": reqs, "advice_only": True, "prediction": {"label": "Potential Fit", "probabilities": {"No Fit": .3, "Potential Fit": .4, "Good Fit": .3}, "abstain": False}}
with sync_playwright() as p:
    b = p.chromium.launch()
    def newpage():
        pg = b.new_context(accept_downloads=True).new_page(); errs = []; dlg = []
        pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("dialog", lambda d: (dlg.append(d.type), d.accept())); return pg, errs, dlg
    def load_json(pg, obj):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f.write(obj if isinstance(obj, str) else json.dumps(obj)); f.close()
        pg.set_input_files("#file", f.name); time.sleep(.3); os.unlink(f.name)
    msg = lambda pg: pg.inner_text("#msg")
    pg, errs, dlg = newpage(); pg.goto(URL)
    rec("V01", "page loads with no console errors", not errs, "; ".join(errs)[:120])
    rec("V02", "advice-only banner; wording 'not evidence' of absence", "never accepts or rejects" in pg.inner_text(".banner") and "not</b>" not in pg.inner_text(".banner") and "not evidence" in pg.inner_text(".banner"))
    pg.click("#demo"); pg.wait_for_selector("#app:not([hidden])"); demo = json.load(open(D / "demo_case_SYNTHETIC.json"))
    rec("V03", "demo loads; rows = requirements", pg.locator("#reqs tr").count() == len(demo["requirements"]))
    rec("V04", "abstain shown as low confidence", "low confidence" in pg.inner_text("#pred").lower())
    pg.click("#save"); rec("V05", "UX-01 no decision: inline message, no browser dialog", "Choose a decision" in msg(pg) and not dlg and pg.get_attribute("#msg", "aria-live") == "polite", msg(pg))
    pg.select_option("#dec", "No Fit"); pg.click("#save"); rec("V06", "UX-01 override without reason: inline message, focus on reason", "reason" in msg(pg).lower() and pg.evaluate("document.activeElement.id") == "why" and not dlg)
    pg.fill("#why", "synthetic test reason")
    with pg.expect_download() as dl: pg.click("#save")
    a = json.load(open(dl.value.path())); rec("V07", "audit record content and no CV text", a["overridden"] and a["hr_decision"] == "No Fit" and demo["cv_text"][:30] not in json.dumps(a))
    rec("V08", "save confirmation shown", "saved" in msg(pg).lower())
    pg2, e2, d2 = newpage(); reqs = []; pg2.on("request", lambda r: reqs.append(r.url)); pg2.goto(URL); pg2.click("#demo"); pg2.wait_for_selector("#app:not([hidden])")
    rec("V09", "no request leaves localhost", all(u.startswith(f"http://127.0.0.1:{port}") for u in reqs))
    pg3, e3, d3 = newpage(); pg3.goto(URL); pg3.evaluate("window.__x=0")
    evil = '<img src=x onerror="window.__x=1"><script>window.__x=2</script>'; c = case(cv=evil + " Python", job="Python " + evil); c["requirements"][0]["value"] = evil
    load_json(pg3, c); rec("V10", "HTML in texts not executed, shown literally", pg3.evaluate("window.__x") == 0 and pg3.locator("#cv img,#reqs img").count() == 0 and "<img" in pg3.inner_text("#cv"))
    pg4, e4, d4 = newpage(); pg4.goto(URL); load_json(pg4, "{not json"); rec("V11", "UX-01 invalid JSON: inline error, app hidden, no dialog", "Could not read" in msg(pg4) and pg4.locator("#app").is_hidden() and not d4, msg(pg4))
    load_json(pg4, {"foo": 1}); rec("V12", "wrong schema: inline error", "Not a valid case file" in msg(pg4) and pg4.locator("#app").is_hidden())
    pg5, e5, d5 = newpage(); pg5.goto(URL); c = case(cv="😀😀 Python and SQL", job="Python required", n_req=1); c["requirements"][0]["cv_spans"] = [[3, 9, "EXPLICIT"]]
    load_json(pg5, c); rec("V13", "emoji offsets highlight correctly", pg5.locator("#cv mark").all_inner_texts() == ["Python"])
    # UX-02 overlapping spans: both passages covered
    pg6, e6, d6 = newpage(); pg6.goto(URL); c = case(cv="Senior Python developer", job="Python", n_req=2)
    c["requirements"][0]["cv_spans"] = [[0, 13, "EXPLICIT"]]; c["requirements"][1]["status"] = "FOUND_EXPLICIT"; c["requirements"][1]["cv_spans"] = [[7, 23, "EXPLICIT"]]
    load_json(pg6, c); mk = pg6.locator("#cv mark").all_inner_texts()
    rec("V14", "UX-02 overlapping spans merged: both passages highlighted", mk == ["Senior Python developer"] and not e6, str(mk))
    rec("V15", "UX-02/08 evidence list shows each excerpt per requirement", pg6.locator("#reqs .ev button").all_inner_texts() == ["CV: “Senior Python”", "CV: “Python developer”"], str(pg6.locator("#reqs .ev button").all_inner_texts()))
    pg6.locator("#reqs .ev button").nth(1).click(); rec("V16", "UX-08 jump-to-source focuses the highlight", pg6.evaluate("document.activeElement.tagName") == "MARK" and pg6.locator("mark.focus").count() == 1)
    # UX-07 filter
    pg7, e7, d7 = newpage(); pg7.goto(URL); load_json(pg7, case(n_req=300, cv="Python " * 2000, job="Python " * 500))
    pg7.select_option("#flt", "NOT_STATED_UNKNOWN"); vis = pg7.locator("#reqs tr:not([hidden])").count()
    rec("V17", "UX-07 filter by evidence status (300 rows)", vis == 150 and "150 of 300" in pg7.inner_text("#cnt"), f"{vis} visible")
    t0 = time.time(); load_json(pg7, case(n_req=300, cv="Python " * 2000, job="Python " * 500, cid="t9")); rec("V18", "300-requirement case renders in < 3 s", time.time() - t0 < 3, f"{time.time()-t0:.2f}s")
    # UX-03 unsaved-change protection
    pg8, e8, d8 = newpage(); pg8.goto(URL); load_json(pg8, case()); pg8.select_option("#dec", "Good Fit")
    load_json(pg8, case(cid="t2")); blocked = "unsaved" in msg(pg8).lower() and "t1" in pg8.inner_text("#reqs").lower() + "t1"
    rec("V19", "UX-03 loading another case with an unsaved decision is stopped once", "unsaved" in msg(pg8).lower())
    load_json(pg8, case(cid="t2")); rec("V20", "second load discards and opens the new case", "t2 loaded" in msg(pg8).lower())
    pg8.click("#why"); pg8.keyboard.type("draft reason")
    try: pg8.goto(URL + "?leave", timeout=5000)
    except Exception: pass
    time.sleep(.5)
    rec("V21", "UX-03 leaving the page with unsaved input (after a real user gesture) triggers the browser warning", "beforeunload" in d8, str(d8))
    pg9, e9, d9 = newpage(); pg9.goto(URL); load_json(pg9, case()); pg9.locator("#reqs select").first.select_option("Extraction is wrong")
    c2 = case(cid="t2"); load_json(pg9, c2); load_json(pg9, c2); pg9.select_option("#dec", "Potential Fit")
    with pg9.expect_download() as dl: pg9.click("#save")
    rec("V22", "HR checks do not leak between cases", json.load(open(dl.value.path()))["evidence_checks"] == {})
    rec("V23", "controls have accessible names", all(pg.locator(s).count() for s in ("label:has(#dec)", "label:has(#why)", "label:has(#flt)")))
    rec("V24", "no browser alert() used anywhere", "alert(" not in open(D / "hr_review_prototype.html").read())
    b.close()
with open(D / "test_results_v2.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(["id", "test", "result", "detail"]); w.writerows(res)
print(sum(r[2] == "PASS" for r in res), "pass,", sum(r[2] == "FAIL" for r in res), "fail")
