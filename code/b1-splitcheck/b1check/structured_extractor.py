"""HireMind B1 structured extractor (rule-based, deterministic).

Adapted from the team's evidence_v2.py (Mervat) and backend/app/skills.py (Senior-2 @ec28dfc).
What was kept: the SKILLS vocabulary, CONTEXT_ONLY handling, negation idea, offset-preserving design.
What was changed (reason in schema.md): the real dataset text has headings and sentences glued together
("SummaryDetail-oriented...", "needed.Computed ..."), so line-equals-heading section detection never fires.
Every claim carries character offsets into the ORIGINAL string and a SHA-256 of the exact span, so it can
be re-checked against the source without storing the text.
Unknown is not negative: a requirement with no CV evidence is NOT_STATED_UNKNOWN, never a negative.
"""
import bisect
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from skills_vocab_senior2_ORIGINAL import CONTEXT_ONLY, SECTION_HEADINGS, SKILLS  # noqa: E402

SCHEMA_VERSION = "structured-evidence-v3"

# ---------- CV sections (headings are glued to the first sentence in this dataset) ----------
_EXTRA_HEADINGS = {
    "skills": ["highlights", "skill highlights", "core qualifications", "core strengths", "areas of expertise", "professional skills",
               "skills summary", "computer skills", "key skills and abilities", "qualifications summary"],
    "summary": ["professional profile", "career overview", "career summary", "executive profile", "summary of qualifications",
                "professional overview"],
    "experience": ["professional experience", "work history", "relevant experience", "related experience", "career history"],
    "education": ["educational background", "education and training", "education and certifications", "academic credentials"],
    "awards": ["accomplishments"],
    "other": ["additional information", "interests", "affiliations", "personal information"],
}
HEADINGS = {}
for _sec, _hs in list(SECTION_HEADINGS.items()) + list(_EXTRA_HEADINGS.items()):
    for _h in _hs:
        HEADINGS[_h] = _sec
_HEAD_SORTED = sorted(HEADINGS, key=len, reverse=True)


def detect_heading(line):
    """Return (section, heading_length) if the line starts with a known heading (glued or alone), else None."""
    if not line or not line[0].isupper():
        return None
    low = line.lower()
    for h in _HEAD_SORTED:
        if low.startswith(h):
            nxt = line[len(h):len(h) + 1]
            if nxt == "" or not nxt.islower():
                return HEADINGS[h], len(h)
    return None


def cv_lines(text):
    """[(start, end, section)] for each non-empty line; offsets into `text`; heading text excluded from the body."""
    out, sec, pos = [], "other", 0
    for raw in text.split("\n"):
        start, end = pos, pos + len(raw)
        pos = end + 1
        body_start = start
        h = detect_heading(raw.lstrip())
        if h:
            sec = h[0]
            body_start = start + (len(raw) - len(raw.lstrip())) + h[1]
            while body_start < end and text[body_start] in ":- \t":
                body_start += 1
        if end > body_start:
            out.append((body_start, end, sec))
    return out


# ---------- sentence boundaries (glued: "needed.Computed") ----------
_BOUND = re.compile(r"(?<=[.!?;])(?=[A-Z•])|[\n•]")


def sentence_starts(text):
    return [0] + [m.end() for m in _BOUND.finditer(text)]


def sentence_span(starts, n, pos):
    i = bisect.bisect_right(starts, pos) - 1
    s = starts[i]
    e = starts[i + 1] if i + 1 < len(starts) else n
    return s, e


# ---------- skill patterns ----------
def _alts(aliases):
    return "|".join(r"\s+".join(map(re.escape, a.split())) for a in sorted(set(aliases), key=len, reverse=True))


SKILL_PATTERNS = {c: (re.compile("(?:" + _alts(al) + ")", re.I), al) for c, al in SKILLS.items() if al}


def _left_ok(text, s, matched):
    if s == 0:
        return True
    p = text[s - 1]
    if not (p.isalnum() or p in "+#._"):
        return True
    if p == "." and matched[0].isupper() and matched.lower() not in CONTEXT_ONLY and s >= 2 and (text[s - 2].islower() or text[s - 2].isdigit() or text[s - 2] in ")\"'"):
        return True  # sentence glued to the previous one: "experience.Python"
    # glued items ("...developmentExcel"): allow only for longer, capitalised, non-ambiguous matches
    return p.islower() and matched[0].isupper() and len(matched) >= 5 and matched.lower() not in CONTEXT_ONLY


def _right_ok(text, e, matched):
    if e >= len(text):
        return True
    n = text[e]
    if not (n.isalnum() or n in "+#"):
        return True
    return n.isupper() and matched[-1].islower() and len(matched) >= 5


NEG_PRE = [r"\bno\s+(?:(?:prior|previous|formal|direct)\s+)?(?:experience|knowledge|exposure)\s+(?:with|in|of)\s*$",
           r"\bno\s*$", r"\bnever\s+(?:used|worked\s+with|had\s+experience\s+(?:with|in))\s*$",
           r"\bwithout\s+(?:any\s+)?(?:experience\s+(?:with|in)\s+)?$", r"\b(?:did|do|does)\s+not\s+(?:use|have|know)\s*$",
           r"\b(?:has|have)\s+not\s+(?:used|worked\s+with)\s*$", r"\b(?:not|un)\s+familiar\s+with\s*$",
           r"\black\s+of\s+(?:experience\s+(?:with|in)\s+)?$", r"\bnot\s+required\s*:?\s*$"]
NEG_SUF = [r"^\s*(?:(?:prior|previous)\s+)?experience\s+(?:is\s+)?(?:not\s+)?(?:required|necessary|needed)\b",
           r"^(?:\s*\w+){0,2}\s+(?:was|were|is|are)\s+(?:not|never)\s+used\b"]
FUTURE_PRE = [r"\binterested\s+in\s+learning\s*$", r"\bwilling(?:ness)?\s+to\s+learn\s*$", r"\beager\s+to\s+learn\s*$",
              r"\bwant\s+to\s+learn\s*$", r"\bcurrently\s+learning\s*$", r"\bhope\s+to\s+(?:learn|use)\s*$"]


def classify_mention(sentence, s, e):
    """'NEGATIVE' (explicit lack), 'FUTURE' (interest only, not evidence) or 'POSITIVE'."""
    pre, suf = sentence[:s].lower(), sentence[e:].lower()
    if any(re.search(p, pre) for p in FUTURE_PRE):
        return "FUTURE"
    # "no X required": handled as negative-for-requirement by callers; plain "no X experience"
    if re.search(r"\bno\s*$", pre) and re.match(r"\s*(?:(?:prior|previous)\s+)?(?:experience|knowledge)\b", suf):
        return "NEGATIVE"
    if any(re.search(p, pre) for p in NEG_PRE) or any(re.search(p, suf) for p in NEG_SUF):
        return "NEGATIVE"
    return "POSITIVE"


# ---------- JD requirement typing ----------
_PREF = re.compile(r"\b(prefer(?:red|ably)?|a\s+plus|nice\s+to\s+have|bonus|desired|desirable|ideal(?:ly)?|advantageous|an?\s+asset)\b", re.I)
_REQ = re.compile(r"\b(required|requires?|must|minimum|mandatory|essential|need(?:ed|s)?\s+to|necessary|at\s+least)\b", re.I)
JD_SECTION_HEADS = [
    (r"(?:preferred|desired|bonus|nice\s+to\s+have)[\w\s/&-]{0,30}?(?:qualifications?|skills?|points|experience|requirements?)?\s*:?", "PREFERRED"),
    (r"bonus\s+points|nice\s+to\s+have|plus\s+points", "PREFERRED"),
    (r"(?:minimum\s+|basic\s+|required\s+|key\s+)?(?:qualifications?|requirements?|skills(?:\s+and\s+experience)?|experience|what\s+you(?:'ll|\s+will)?\s+(?:bring|need)|who\s+you\s+are)\s*:", "REQUIRED"),
    (r"(?:key\s+)?(?:responsibilities|duties|what\s+you(?:'ll|\s+will)\s+do|job\s+description|about\s+the\s+role)\s*:?", "NEUTRAL"),
    (r"(?:perks|benefits|compensation|what\s+we\s+offer|about\s+us|equal\s+opportunity|warning|disclaimer)\s*:?", "NOT_REQUIREMENT"),
]
_JD_HEAD_RE = [(re.compile(r"^\s*" + p, re.I), t) for p, t in JD_SECTION_HEADS]


def jd_sentences(text):
    st = sentence_starts(text)
    spans = []
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else len(text)
        if text[s:e].strip():
            spans.append((s, e))
    return spans


def jd_req_type(sentence, section_type):
    """(requirement_type, basis). Sentence cues beat section; no cue and no section => UNKNOWN (never defaults to REQUIRED)."""
    if _PREF.search(sentence):
        return "PREFERRED", "sentence_cue"
    if _REQ.search(sentence):
        return "REQUIRED", "sentence_cue"
    if section_type == "REQUIRED":
        return "REQUIRED", "section"
    if section_type == "PREFERRED":
        return "PREFERRED", "section"
    return "UNKNOWN", "none"


# ---------- education / years / dates / certification ----------
DEGREE_RANK = {"HIGH_SCHOOL": 1, "ASSOCIATE": 2, "BACHELOR": 3, "MASTER": 4, "DOCTORATE": 5}
_DEG = [
    ("DOCTORATE", re.compile(r"\b(?:ph\.?\s?d\.?|doctorate|doctor\s+of\s+\w+)", re.I), None),
    ("MASTER", re.compile(r"\b(?:master(?:'?s)?(?:\s+degree|\s+of\s+\w+|\s+in\s+\w+)|mba\b|m\.s\.?c?\b|m\.a\.\B)", re.I), None),
    ("BACHELOR", re.compile(r"\b(?:bachelor(?:'?s)?(?:\s+degree|\s+of\s+\w+|\s+in\s+\w+|\b)|b\.s\.?c?\b|b\.a\.\B)", re.I), None),
    ("ASSOCIATE", re.compile(r"\b(?:associate(?:'?s)?(?:\s+degree|\s+of\s+\w+|\s+in\s+\w+)|a\.a\.s?\b)", re.I), None),
    ("HIGH_SCHOOL", re.compile(r"\b(?:high\s+school(?:\s+diploma)?|ged\b|secondary\s+school)", re.I), None),
]
_DEGREE_GENERIC = re.compile(r"\b(?:college\s+degree|university\s+degree|degree)\b", re.I)
_YEARS = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*(\d{1,2})\s*\+?\s*)?(?:years?|yrs?)\b(?=[^.]{0,70}?(?:experience|exp\b))", re.I)
_YEARS_CV = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b(?:\s+of)?\s+(?:[\w&/-]+\s+){0,3}?experience|experience\s+(?:of\s+)?(\d{1,2})\s*\+?\s*(?:years?|yrs?)|(?:over|more\s+than)\s+(\d{1,2})\s+years?", re.I)
_DATERANGE = re.compile(r"((?:(?:0?[1-9]|1[0-2])/)?(?:19|20)\d{2})\s*(?:to|-|–|—)\s*((?:(?:0?[1-9]|1[0-2])/)?(?:19|20)\d{2}|current|present|now|ongoing)", re.I)
_CERT_ACR = re.compile(r"\b(CPA|PMP|CISSP|CFA|CMA|CIA|PHR|SPHR|CPR|ACLS|BLS|RN|LPN|CNA|MCSE|CCNA|ITIL|CDL|OSHA|CAPM|PRINCE2|SHRM-CP|SHRM-SCP|CISA|CEH)\b")
_CERT_GEN = re.compile(r"\b(?:certifications?|certificates?|licen[sc]es?|licensed)\b", re.I)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Builder:
    def __init__(self, case_id, source_type, text):
        self.case_id, self.src, self.text = case_id, source_type, text
        self.records, self.n = [], 0
        self.prefix = "R" if source_type == "JOB_DESCRIPTION" else "C"

    def add(self, category, s, e, normalized, status, confidence, req_type, **kw):
        self.n += 1
        span = self.text[s:e]
        rec = dict(case_id=self.case_id, record_id=f"{self.case_id}:{self.prefix}{self.n:03d}", source_type=self.src,
                   evidence_category=category, normalized_value=normalized, evidence_status=status, confidence=confidence,
                   requirement_type=req_type, source_start_char=s, source_end_char=e, span_sha256=sha(span),
                   review_required=bool(kw.pop("review_required", confidence != "HIGH")))
        rec.update({k: v for k, v in kw.items() if v is not None})
        self._verbatim = getattr(self, "_verbatim", {})
        self._verbatim[rec["record_id"]] = span
        self.records.append(rec)
        return rec


def _skill_mentions(text, spans, allow_context_only):
    """yield (canonical, s, e, matched_alias, in_skills_section_flag) over given [(start,end,section)] ranges."""
    for (a, b, sec) in spans:
        seg = text[a:b]
        for canon, (pat, aliases) in SKILL_PATTERNS.items():
            ctx_only = any(x in CONTEXT_ONLY for x in aliases)
            for m in pat.finditer(seg):
                s, e = a + m.start(), a + m.end()
                matched = text[s:e]
                if matched.lower() in CONTEXT_ONLY and not allow_context_only(sec):
                    continue
                if not (_left_ok(text, s, matched) and _right_ok(text, e, matched)):
                    continue
                yield canon, s, e, matched, sec


def extract_cv(case_id, text):
    B = Builder(case_id, "CV", text)
    lines = cv_lines(text)
    starts = sentence_starts(text)
    seen = set()
    for canon, s, e, matched, sec in _skill_mentions(text, lines, lambda sec: sec == "skills"):
        ss, se = sentence_span(starts, len(text), s)
        kind = classify_mention(text[ss:se], s - ss, e - ss)
        if (canon, s) in seen:
            continue
        seen.add((canon, s))
        ambiguous = matched.lower() in CONTEXT_ONLY
        if kind == "POSITIVE":
            B.add("TOOL_TECHNOLOGY" if canon in _TECH else "SKILL", s, e, canon, "EXPLICIT", "MEDIUM" if ambiguous else "HIGH", "NOT_APPLICABLE",
                  section=sec, matched_alias=matched.lower(), context_start_char=ss, context_end_char=se)
        elif kind == "NEGATIVE":
            B.add("SKILL", s, e, canon, "EXPLICIT_NEGATIVE", "MEDIUM", "NOT_APPLICABLE", section=sec, matched_alias=matched.lower(),
                  context_start_char=ss, context_end_char=se, review_required=True)
        else:
            B.add("SKILL", s, e, canon, "UNCLEAR", "LOW", "NOT_APPLICABLE", section=sec, matched_alias=matched.lower(),
                  context_start_char=ss, context_end_char=se, review_required=True,
                  context="interest in learning only; not evidence of the skill")
    # education
    for a, b, sec in lines:
        seg = text[a:b]
        found = set()
        for lvl, pat, _ in _DEG:
            for m in pat.finditer(seg):
                abbrev = len(m.group(0)) <= 6 and "." in m.group(0) or m.group(0).lower() in ("mba", "ged")
                if abbrev and sec != "education":
                    continue
                s, e = a + m.start(), a + m.end()
                if any(s < fe and e > fs for fs, fe in found):
                    continue
                found.add((s, e))
                B.add("EDUCATION", s, e, lvl, "EXPLICIT", "HIGH" if sec == "education" else "MEDIUM", "NOT_APPLICABLE", section=sec,
                      degree_name=lvl.title().replace("_", " "))
    # years of experience (explicit statements only)
    for a, b, sec in lines:
        for m in _YEARS_CV.finditer(text[a:b]):
            g = next(x for x in m.groups() if x)
            B.add("EXPERIENCE", a + m.start(), a + m.end(), f"{int(g)} years", "EXPLICIT", "MEDIUM", "NOT_APPLICABLE", section=sec,
                  years_experience=float(g), review_required=True, context="stated duration; the field it refers to is not resolved")
    # employment date ranges (strings kept as written; no durations computed)
    for a, b, sec in lines:
        if sec not in ("experience", "other"):
            continue
        for m in _DATERANGE.finditer(text[a:b]):
            B.add("EMPLOYMENT", a + m.start(), a + m.end(), f"{m.group(1)} to {m.group(2)}".lower(), "EXPLICIT", "MEDIUM", "NOT_APPLICABLE",
                  section=sec, employment_start_date=m.group(1), employment_end_date=m.group(2), date_or_timeframe=m.group(0),
                  review_required=sec != "experience")
    # certifications
    for a, b, sec in lines:
        for m in _CERT_ACR.finditer(text[a:b]):
            B.add("CERTIFICATION", a + m.start(), a + m.end(), m.group(1), "EXPLICIT", "MEDIUM", "NOT_APPLICABLE", section=sec)
    return B


_TECH = {k for k in SKILLS if k in {"python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust", "kotlin", "swift", "php", "ruby", "r",
                                    "matlab", "scala", "sql", "bash", "html", "css", "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
                                    "spring", ".net", "graphql", "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "power bi",
                                    "tableau", "excel", "spark", "hadoop", "postgresql", "mysql", "mongodb", "oracle", "redis", "aws", "azure", "gcp",
                                    "docker", "kubernetes", "git", "linux", "terraform", "figma", "jira", "sap", "crm"}}


def extract_jd(case_id, text):
    B = Builder(case_id, "JOB_DESCRIPTION", text)
    sec_type, sentences = "NEUTRAL", jd_sentences(text)
    for (a, b) in sentences:
        sent = text[a:b]
        for rx, t in _JD_HEAD_RE:
            hm = rx.match(sent)
            if hm and hm.end() > 0:
                sec_type = t
                break
        if sec_type == "NOT_REQUIREMENT":
            continue
        rtype, basis = jd_req_type(sent, sec_type)
        # skills
        for canon, s, e, matched, sec in _skill_mentions(text, [(a, b, "jd")], lambda sec: True):
            kind = classify_mention(text[a:b], s - a, e - a)
            if matched.lower() in CONTEXT_ONLY and not re.search(r"\b(?:skills?|proficien\w+|fluen\w+|knowledge|experience)\b", sent, re.I):
                continue
            if kind != "POSITIVE":
                B.add("SKILL", s, e, canon, "UNCLEAR", "LOW", "UNKNOWN", context_start_char=a, context_end_char=b, review_required=True,
                      context="negated or interest-only mention in the job text; not treated as a requirement")
                continue
            B.add("TOOL_TECHNOLOGY" if canon in _TECH else "SKILL", s, e, canon, "EXPLICIT", "HIGH" if basis != "none" else "MEDIUM", rtype,
                  matched_alias=matched.lower(), context_start_char=a, context_end_char=b, review_required=(rtype == "UNKNOWN"),
                  context=f"requirement type from {basis}")
        # degrees
        got = False
        for lvl, pat, _ in _DEG:
            for m in pat.finditer(sent):
                got = True
                B.add("EDUCATION", a + m.start(), a + m.end(), lvl, "EXPLICIT", "HIGH" if basis != "none" else "MEDIUM", rtype, degree_name=lvl.title().replace("_", " "),
                      context_start_char=a, context_end_char=b, review_required=(rtype == "UNKNOWN"), context=f"requirement type from {basis}")
        if not got:
            for m in _DEGREE_GENERIC.finditer(sent):
                B.add("EDUCATION", a + m.start(), a + m.end(), "DEGREE_LEVEL_UNSPECIFIED", "EXPLICIT", "MEDIUM", rtype, degree_name=None,
                      context_start_char=a, context_end_char=b, review_required=True, context="degree required/mentioned; level not stated")
                break
        # years
        for m in _YEARS.finditer(sent):
            lo = int(m.group(1))
            B.add("EXPERIENCE", a + m.start(), a + m.end(), f"{lo} years", "EXPLICIT", "HIGH" if basis != "none" else "MEDIUM", rtype, years_experience=float(lo),
                  context_start_char=a, context_end_char=b, review_required=True, context="minimum of stated range; field not resolved")
        # certifications
        for m in _CERT_ACR.finditer(sent):
            B.add("CERTIFICATION", a + m.start(), a + m.end(), m.group(1), "EXPLICIT", "HIGH" if basis != "none" else "MEDIUM", rtype,
                  context_start_char=a, context_end_char=b)
        for m in _CERT_GEN.finditer(sent):
            B.add("CERTIFICATION", a + m.start(), a + m.end(), "CERTIFICATION_OR_LICENSE_UNSPECIFIED", "EXPLICIT", "MEDIUM", rtype,
                  context_start_char=a, context_end_char=b, review_required=True)
            break
    return B


# ---------- linking: unknown is not negative ----------
def link(jd, cv):
    cv_by = {}
    for r in cv.records:
        cv_by.setdefault((r["evidence_category"] in ("SKILL", "TOOL_TECHNOLOGY") and "SKILL" or r["evidence_category"], r["normalized_value"]), []).append(r)
    cv_degrees = [r for r in cv.records if r["evidence_category"] == "EDUCATION" and r["normalized_value"] in DEGREE_RANK]
    cv_years = [r for r in cv.records if r["evidence_category"] == "EXPERIENCE"]
    cv_cert = [r for r in cv.records if r["evidence_category"] == "CERTIFICATION"]
    links = []
    for r in jd.records:
        if r["evidence_status"] != "EXPLICIT":
            continue
        cat, val = r["evidence_category"], r["normalized_value"]
        L = dict(requirement_id=r["record_id"], requirement_type=r["requirement_type"], category=cat, normalized_value=val,
                 cv_record_ids=[], link_status="NOT_STATED_UNKNOWN", unknown_is_not_negative=True)
        if cat in ("SKILL", "TOOL_TECHNOLOGY"):
            hits = cv_by.get(("SKILL", val), [])
            pos = [h for h in hits if h["evidence_status"] == "EXPLICIT"]
            neg = [h for h in hits if h["evidence_status"] == "EXPLICIT_NEGATIVE"]
            unc = [h for h in hits if h["evidence_status"] == "UNCLEAR"]
            if pos:
                L.update(link_status="FOUND_EXPLICIT", cv_record_ids=[h["record_id"] for h in pos], unknown_is_not_negative=False)
            elif neg:
                L.update(link_status="FOUND_EXPLICIT_NEGATIVE", cv_record_ids=[h["record_id"] for h in neg], unknown_is_not_negative=False)
            elif unc:
                L.update(link_status="FOUND_UNCLEAR", cv_record_ids=[h["record_id"] for h in unc])
        elif cat == "EDUCATION":
            if cv_degrees:
                best = max(cv_degrees, key=lambda h: DEGREE_RANK[h["normalized_value"]])
                req = DEGREE_RANK.get(val)
                ok = req is None or DEGREE_RANK[best["normalized_value"]] >= req
                L.update(link_status="LEVEL_MET" if ok else "LEVEL_BELOW_REQUIRED", cv_record_ids=[best["record_id"]], unknown_is_not_negative=False,
                         note="degree level only; field of study is not compared")
        elif cat == "EXPERIENCE":
            if cv_years:
                L.update(link_status="YEARS_STATED_NOT_COMPARED", cv_record_ids=[h["record_id"] for h in cv_years], unknown_is_not_negative=False,
                         note="the CV states durations, but the extractor cannot tell which field they refer to; human or matcher decides")
        elif cat == "CERTIFICATION":
            hit = [h for h in cv_cert if r["normalized_value"] in (h["normalized_value"], "CERTIFICATION_OR_LICENSE_UNSPECIFIED")]
            if hit:
                L.update(link_status="FOUND_EXPLICIT", cv_record_ids=[h["record_id"] for h in hit], unknown_is_not_negative=False)
        links.append(L)
        if L["cv_record_ids"]:
            for h in cv.records:
                if h["record_id"] in L["cv_record_ids"]:
                    h.setdefault("linked_requirement_ids", []).append(r["record_id"])
    return links


def structure_pair(case_id, cv_text, jd_text, with_verbatim=False):
    cv, jd = extract_cv(case_id, cv_text), extract_jd(case_id, jd_text)
    links = link(jd, cv)
    recs = jd.records + cv.records
    if with_verbatim:
        vb = {**jd._verbatim, **cv._verbatim} if hasattr(jd, "_verbatim") and hasattr(cv, "_verbatim") else {**getattr(jd, "_verbatim", {}), **getattr(cv, "_verbatim", {})}
        for r in recs:
            r["verbatim_text"] = vb[r["record_id"]]
    return dict(case_id=case_id, schema_version=SCHEMA_VERSION, cv_chars=len(cv_text), jd_chars=len(jd_text), cv_sha256=sha(cv_text),
                jd_sha256=sha(jd_text), records=recs, requirement_links=links)
