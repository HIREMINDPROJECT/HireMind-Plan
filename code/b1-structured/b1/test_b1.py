"""Unit tests on SYNTHETIC text only (no dataset text). Run: python3 test_b1.py"""
from structured_extractor import structure_pair, extract_cv, cv_lines, detect_heading, sha

def recs(res, src, cat=None):
    return [r for r in res["records"] if r["source_type"] == src and (cat is None or r["evidence_category"] == cat)]

def check(res, text_cv, text_jd):
    for r in res["records"]:
        t = text_cv if r["source_type"] == "CV" else text_jd
        assert sha(t[r["source_start_char"]:r["source_end_char"]]) == r["span_sha256"], r
    return True

cv = ("SummaryDetail-oriented analyst with 11+ years experience.\n"
      "HighlightsSelf-starterMicrosoft ExcelSpreadsheet developmentPython\n"
      "ExperienceAnalyst,2011to2012Acme Corp.No Java experience.Never used Tableau.Interested in learning Docker.Built dashboards in SQL.\n"
      "EducationBachelor of Science in Accounting, 2010\nSkillsCommunication, Teamwork, Oracle")
jd = ("Job Title: AnalystLocation: X\nQualifications:Bachelors degree in accounting required.5+ years of reporting experience.Python and SQL required."
      "Tableau preferred.Knowledge of Docker is a plus.Excel is required.Perks:Free lunch with Java and Kotlin parties.")
res = structure_pair("t1", cv, jd)
check(res, cv, jd)
link = {(l["normalized_value"]): l for l in res["requirement_links"]}
assert detect_heading("SummaryDetail-oriented")[0] == "summary" and detect_heading("Skillset foo") is None
secs = [s for _, _, s in cv_lines(cv)]
assert secs == ["summary", "skills", "experience", "education", "skills"], secs
# glued item: Excel found in skills section
cvs = {r["normalized_value"]: r for r in recs(res, "CV") if r["evidence_category"] in ("SKILL", "TOOL_TECHNOLOGY")}
assert cvs["excel"]["evidence_status"] == "EXPLICIT" and cvs["python"]["evidence_status"] == "EXPLICIT"
assert cvs["java"]["evidence_status"] == "EXPLICIT_NEGATIVE"          # explicit lack
assert cvs["docker"]["evidence_status"] == "UNCLEAR"                    # interest only
assert cvs["sql"]["evidence_status"] == "EXPLICIT"
assert cvs["oracle"]["confidence"] == "MEDIUM"                              # ambiguous alias, skills section only
# requirement typing
assert link["python"]["requirement_type"] == "REQUIRED" and link["tableau"]["requirement_type"] == "PREFERRED"
assert link["docker"]["requirement_type"] == "PREFERRED"
# tableau: CV says "Never used Tableau" -> explicit negative, not unknown
assert link["tableau"]["link_status"] == "FOUND_EXPLICIT_NEGATIVE", link["tableau"]
# docker: interest only -> unclear, not found
assert link["docker"]["link_status"] == "FOUND_UNCLEAR"
# perks section not treated as requirements (Java/Kotlin in perks)
assert "java" not in link and "kotlin" not in link
# education level
assert link["BACHELOR"]["link_status"] == "LEVEL_MET" and link["BACHELOR"]["requirement_type"] == "REQUIRED"
# years not auto-compared
assert link["5 years"]["link_status"] == "YEARS_STATED_NOT_COMPARED"
# unknown is not negative
res2 = structure_pair("t2", "SkillsCommunication", "Qualifications:Python required.")
l2 = res2["requirement_links"][0]
assert l2["link_status"] == "NOT_STATED_UNKNOWN" and l2["unknown_is_not_negative"] is True and l2["cv_record_ids"] == []
# MySQL must not count as SQL; JavaScript must not count as Java
res3 = structure_pair("t3", "SkillsMySQL, JavaScript", "Requirements:SQL and Java required.")
assert all(l["link_status"] == "NOT_STATED_UNKNOWN" for l in res3["requirement_links"]), res3["requirement_links"]
# spaced sentences: a cue in one sentence must not retype requirements in another (v3.1 splitter fix)
res5 = structure_pair("t5", "Python 5 years. Java used. AWS used.", "Requirements: Python required. Java is a plus. Docker needed to deploy. Preferred: AWS experience.")
t5 = {l["normalized_value"]: l["requirement_type"] for l in res5["requirement_links"]}
assert t5["python"] == "REQUIRED" and t5["java"] == "PREFERRED" and t5["aws"] == "PREFERRED", t5
res6 = structure_pair("t6", "Python 5 years.", "We use Python daily. Candidates should know Java. Preferred: AWS experience.")
t6 = {l["normalized_value"]: l["requirement_type"] for l in res6["requirement_links"]}
assert t6["python"] == "UNKNOWN" and t6["java"] == "UNKNOWN" and t6["aws"] == "PREFERRED", t6
# empty texts do not crash
structure_pair("t4", "", "")
print("ALL B1 TESTS PASSED")
