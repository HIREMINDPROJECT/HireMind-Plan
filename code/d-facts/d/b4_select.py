"""B4 evidence-aware input selection (library). Labels are never used.
select_pair(cv, jd, tok, pair, mf=None, reserve=0) -> dict(cv_segments, jd_segments, n_tokens)
  pair   : B1 structured pair (records + requirement_links)
  mf     : optional B3 match_features entry (adds semantic best-matching CV sentences as a second-priority tier)
  reserve: tokens kept free for text prepended later (the Knowledge v1 line in M4)
Job side keeps, in priority order: sentences holding requirement spans (densest first), the opening sentences, then the rest.
CV side keeps: sentences holding evidence linked to a requirement / explicit negatives / degrees / stated years (densest first),
then B3 semantic best matches, then the opening sentences, then the rest. Selected sentences keep original order; text is never rewritten."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from matcher import units  # noqa: E402

MAXLEN, SPECIAL, JD_MIN, HEAD_TOKENS = 512, 3, 200, 48


def ntoks(tok, strings):
    return [len(x) for x in tok(strings, add_special_tokens=False)["input_ids"]] if strings else []


def covered(spans, s, e):
    pos = s
    for a, b in spans:
        if b <= pos:
            continue
        if a > pos:
            return False
        pos = b
        if pos >= e:
            return True
    return pos >= e


def _prio(units_, hits, sem, head_tokens, tokens):
    pr, dens, cum = [9] * len(units_), [], 0
    for i, (a, b) in enumerate(units_):
        n = sum(1 for s, e in hits if s < b and e > a)
        dens.append(n / max(1, tokens[i]))
        if n:
            pr[i] = 0
        elif any(s < b and e > a for s, e in sem):
            pr[i] = 1
        elif cum < head_tokens:
            pr[i] = 2
        cum += tokens[i]
    return pr, dens


def _pick(tokens, pr, dens, budget):
    chosen, used = [], 0
    for i in sorted(range(len(tokens)), key=lambda i: (pr[i], -dens[i], i)):
        if used + tokens[i] <= budget:
            chosen.append(i)
            used += tokens[i]
    return sorted(chosen), used


def relevant_spans(pair):
    recs = pair["records"]
    jd = [(r["source_start_char"], r["source_end_char"]) for r in recs if r["source_type"] == "JOB_DESCRIPTION" and r["evidence_status"] == "EXPLICIT"]
    cv = [(r["source_start_char"], r["source_end_char"]) for r in recs if r["source_type"] == "CV" and (
        r.get("linked_requirement_ids") or r["evidence_status"] == "EXPLICIT_NEGATIVE" or r["evidence_category"] in ("EDUCATION", "EXPERIENCE"))]
    return cv, jd


def select_pair(cv, jd, tok, pair, mf=None, reserve=0):
    budget = MAXLEN - SPECIAL - reserve
    cvu, jdu = units(cv), units(jd)
    cvt, jdt = ntoks(tok, [cv[a:b] for a, b in cvu]), ntoks(tok, [jd[a:b] for a, b in jdu])
    cv_all, jd_all = sum(cvt), sum(jdt)
    cv_rel, jd_rel = relevant_spans(pair)
    sem = [(r["best_unit_start"], r["best_unit_end"]) for r in (mf or {}).get("requirements", []) if r.get("best_unit_start") is not None and r.get("semantic_only_candidate")]
    if cv_all + jd_all <= budget:
        sel_cv, sel_jd = list(range(len(cvu))), list(range(len(jdu)))
    else:
        jd_budget = min(jd_all, max(JD_MIN, budget - cv_all))
        pj, dj = _prio(jdu, jd_rel, [], HEAD_TOKENS, jdt)
        sel_jd, jd_used = _pick(jdt, pj, dj, jd_budget)
        pc, dc = _prio(cvu, cv_rel, sem, HEAD_TOKENS, cvt)
        sel_cv, _ = _pick(cvt, pc, dc, budget - jd_used)
    return dict(cv_segments=[cvu[i] for i in sel_cv], jd_segments=[jdu[i] for i in sel_jd], cv_rel=cv_rel, jd_rel=jd_rel,
                cv_all_tokens=cv_all, jd_all_tokens=jd_all)


def assemble(cv, jd, sel):
    return " ".join(cv[a:b] for a, b in sel["cv_segments"]), " ".join(jd[a:b] for a, b in sel["jd_segments"])
