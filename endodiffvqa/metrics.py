"""Scoring for EndoDiffVQA results: the hierarchical Detect / Specify / Strict metrics and the swap control.

Results file: a JSON list of records (or ``{"results": [...]}``)::

    {"id": "...",                     # item id
     "prediction": "(b)",             # raw model output, parsed here  — or —
     "pred_letter": "b",              # an already-resolved letter
     "control": "full",               # full | swap | perm{k}   (default "full")
     "letter_map": {"a": "c", ...}}   # displayed -> original letter, for option-permuted prompts

Every option (and the gold) is reduced to an answer *cell* (``option_cell``) and every cell to one of two classes:

    Same       BOTH, NEITHER, SAME       present in both / in neither / same value, number or standard
    Different  everything else           only in Video 1 or 2, more in Video 1 or 2, different values, better in 1 or 2

Per data source, then averaged over sources with equal weight (the 4-source macro on the dataset):

    detect    Same/Different macro-F1: ½ (F1(Same) + F1(Different)); always-"Same" scores about 33
    specify   exact accuracy on the items whose gold is Different AND that the model called Different; a source
              where that count is below --min-specify-n (20) is flagged small and left out of the macro
    strict    exact-answer macro-F1 over the gold cells within each template kind (presence, attribute, count,
              skill), averaged over the kinds present in the source: every answer type weighs the same

Under the ``swap`` control (Video 1 <-> Video 2, options unchanged) the gold is ``swap_answer``: the D1 option for an
asymmetric truth, the same letter for a symmetric one; items whose swapped truth is not offered are dropped, and the
Same/Different classes are taken on the swapped cell. When both ``full`` and ``swap`` are present the report adds the
share of items answered correctly in both orders.

CLI::

    python -m endodiffvqa.metrics --results results.json --dataset-root R [--split test]
    python -m endodiffvqa.metrics --results results.json --jsonl items.jsonl

writes ``metrics.json`` and ``metrics.md`` next to the results file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .dataset import ALL_SOURCES, LETTERS, template_kind

MIN_SPECIFY_N = 20
NO_ANSWER = "none"

SKILL_TEXT = {
    "No — clearly better in Video 1": "CLEAR_A",
    "No — clearly better in Video 2": "CLEAR_B",
    "No — slightly better in Video 1": "SLIGHT_A",
    "No — slightly better in Video 2": "SLIGHT_B",
    "Yes — to the same standard in both videos": "SAME",
}
SWAP_CELL = {"ONLY_A": "ONLY_B", "ONLY_B": "ONLY_A", "MORE_A": "MORE_B", "MORE_B": "MORE_A",
             "CLEAR_A": "CLEAR_B", "CLEAR_B": "CLEAR_A", "SLIGHT_A": "SLIGHT_B", "SLIGHT_B": "SLIGHT_A"}
SYMMETRIC_CELLS = {"BOTH", "NEITHER", "SAME"}

_V_ONLY_A = re.compile(r"(is not visible|appears) in Video 2$")
_V_ONLY_B = re.compile(r"(is not visible|appears) in Video 1;")


# --------------------------------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------------------------------
def option_cell(item: Dict, letter: str) -> str:
    """Answer cell of one option, recovered from its text (the templates render it deterministically)."""
    text = item["options"][letter]
    yes = text.startswith("Yes")
    kind = template_kind(item)
    if kind == "skill":
        return SKILL_TEXT[text]
    if kind == "count":
        if "more instruments in Video 1" in text:
            return "MORE_A"
        if "more instruments in Video 2" in text:
            return "MORE_B"
        if "same number" in text:
            return "SAME"
        return "NEITHER"
    if kind == "presence":
        if yes:
            return "BOTH"
        if "Video 1 only" in text:
            return "ONLY_A"
        if "Video 2 only" in text:
            return "ONLY_B"
        return "NEITHER"
    if yes:
        return "SAME"
    if _V_ONLY_A.search(text):
        return "ONLY_A"
    if _V_ONLY_B.search(text):
        return "ONLY_B"
    return "DIFFERENT"


def gold_cell(item: Dict) -> str:
    return item["cell"] if item.get("cell") else option_cell(item, item["answer_letter"])


def swap_answer(item: Dict) -> Optional[str]:
    """Correct letter once Video 1 and Video 2 are exchanged (options unchanged), or None if it is not offered."""
    if "swap_answer_letter" in item:
        return item["swap_answer_letter"]
    if gold_cell(item) in SYMMETRIC_CELLS:
        return item["answer_letter"]
    codes = item.get("option_codes") or {}
    d1 = [k for k, c in codes.items() if c == "D1"]
    if d1:
        return d1[0]
    want = SWAP_CELL.get(gold_cell(item))
    hits = [k for k in LETTERS if option_cell(item, k) == want] if want else []
    return hits[0] if len(hits) == 1 else None


def binary(cell: Optional[str]) -> str:
    if cell is None:
        return NO_ANSWER
    return "Same" if cell in SYMMETRIC_CELLS else "Different"


# --------------------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_LETTER_PATTERNS = [
    re.compile(r"^\(?([a-d])\)?[.:]?$"),
    re.compile(r"^\(([a-d])\)"),
    re.compile(r"^([a-d])[).:]"),
    re.compile(r"^(?:the )?(?:correct |final )?(?:answer|option)(?: is)?\s*[:\-]?\s*\(?([a-d])\)?(?:[\s.,:)]|$)"),
]


def _norm(s: str) -> str:
    s = s.lower().replace("—", "-").replace("–", "-")
    s = re.sub(r"[\"'`*]", "", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def parse_letter(text: Optional[str], options: Dict[str, str]) -> Optional[str]:
    """Displayed letter chosen by a free-form answer, or None (a parse failure)."""
    if text is None:
        return None
    t = _THINK.sub("", text).strip().lower()
    t = re.sub(r"^\*+|\*+$", "", t).strip()
    for pat in _LETTER_PATTERNS:
        m = pat.match(t)
        if m:
            return m.group(1)
    nt = _norm(t)
    hits = [k for k, v in options.items() if _norm(v) and _norm(v) in nt]
    if len(hits) > 1:
        hits = [k for k in hits if not any(k != o and _norm(options[k]) in _norm(options[o]) for o in hits)]
    return hits[0] if len(hits) == 1 else None


def resolve(rec: Dict, item: Dict) -> Dict:
    """One scored row: predicted original letter, truth under the record's control, parse status."""
    lmap = rec.get("letter_map") or {k: k for k in LETTERS}
    displayed = {d: item["options"][o] for d, o in lmap.items()}
    if rec.get("pred_letter") is not None:
        d = str(rec["pred_letter"]).strip().lower()
        d = d if d in lmap else None
    else:
        d = parse_letter(rec.get("prediction"), displayed)
    pred = lmap[d] if d else None
    control = rec.get("control", "full")
    truth = swap_answer(item) if control == "swap" else item.get("answer_letter")
    return {"id": item["id"], "control": control, "pred": pred, "parsed": pred is not None, "truth": truth}


# --------------------------------------------------------------------------------------------------------------
# Hierarchical metrics
# --------------------------------------------------------------------------------------------------------------
def f1_table(pairs: Iterable[Tuple[str, str]], labels: Iterable[str]) -> Dict[str, Dict[str, float]]:
    pairs = list(pairs)
    out = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "precision": pr, "recall": rc, "n_gold": tp + fn}
    return out


def macro_f1(pairs, labels) -> float:
    t = f1_table(pairs, labels)
    return sum(v["f1"] for v in t.values()) / len(t) if t else 0.0


def source_metrics(rows: List[Tuple[Dict, Optional[str], str]], swapped: bool,
                   min_specify_n: int = MIN_SPECIFY_N) -> Dict:
    """All metrics for the rows ``(item, predicted original letter or None, truth letter)`` of ONE data source."""
    bpairs, spec, swap_err, kinds = [], [], Counter(), defaultdict(list)
    n_diff_gold = 0
    for it, p, truth in rows:
        g_cell = gold_cell(it)
        if swapped:
            g_cell = SWAP_CELL.get(g_cell, g_cell)
        pc = option_cell(it, p) if p else None
        ok = p == truth
        g = binary(g_cell)
        bpairs.append((g, binary(pc)))
        kinds[template_kind(it)].append((g_cell, pc if pc is not None else NO_ANSWER))
        if g == "Different":
            n_diff_gold += 1
            if binary(pc) == "Different":
                spec.append(ok)
                if not ok:
                    codes = it.get("option_codes") or {}
                    swap_err["D1" if codes.get(p) == "D1" else "other"] += 1
    det = f1_table(bpairs, ("Same", "Different"))
    strict_kind = {k: macro_f1(pr, sorted({g for g, _ in pr})) for k, pr in kinds.items()}
    n_spec, n_err = len(spec), sum(swap_err.values())
    return {
        "n": len(rows), "n_different_gold": n_diff_gold,
        "parse_failures": sum(p == NO_ANSWER for _, p in bpairs) / len(bpairs),
        "detect": (det["Same"]["f1"] + det["Different"]["f1"]) / 2,
        "detect_by_class": det,
        "says_different": sum(p == "Different" for _, p in bpairs) / len(bpairs),
        "specify": sum(spec) / n_spec if n_spec else None, "specify_n": n_spec,
        "specify_small": n_spec < min_specify_n,
        "specify_errors_swap_share": swap_err["D1"] / n_err if n_err else None,
        "strict": sum(strict_kind.values()) / len(strict_kind), "strict_by_kind": strict_kind,
    }


def score_control(records: List[Dict], items: Dict[str, Dict], control: str,
                  min_specify_n: int = MIN_SPECIFY_N) -> Dict:
    rows_by_src = defaultdict(list)
    seen = set()
    for r in records:
        if r.get("control", "full") != control:
            continue
        it = items[r["id"]]
        if it["id"] in seen:
            raise ValueError(f"duplicate id {it['id']} under control {control}")
        seen.add(it["id"])
        row = resolve(r, it)
        if row["truth"] is None:  # swapped truth not offered, or an unlabelled item
            continue
        rows_by_src[it["data_source"]].append((it, row["pred"], row["truth"]))
    sources = [s for s in ALL_SOURCES if s in rows_by_src] + sorted(s for s in rows_by_src if s not in ALL_SOURCES)
    per = {s: source_metrics(rows_by_src[s], control == "swap", min_specify_n) for s in sources}
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    spec_ok = [per[s]["specify"] for s in sources if per[s]["specify"] is not None and not per[s]["specify_small"]]
    n = sum(p["n"] for p in per.values())
    return {
        "control": control, "n_items": n, "sources": sources, "min_specify_n": min_specify_n,
        "macro": {"detect": mean([per[s]["detect"] for s in sources]),
                  "specify": mean(spec_ok), "specify_partial": len(spec_ok) < len(sources),
                  "strict": mean([per[s]["strict"] for s in sources])},
        "by_source": per,
    }


def score(records: List[Dict], items: Dict[str, Dict], min_specify_n: int = MIN_SPECIFY_N) -> Dict:
    missing = [r["id"] for r in records if r["id"] not in items]
    if missing:
        raise KeyError(f"{len(missing)} result ids are not among the items, e.g. {missing[:3]}")
    controls = sorted({r.get("control", "full") for r in records})
    out = {"n_records": len(records), "controls": {c: score_control(records, items, c, min_specify_n)
                                                   for c in controls}}
    if "full" in controls and "swap" in controls:
        full = {r["id"]: resolve(r, items[r["id"]]) for r in records if r.get("control", "full") == "full"}
        pairs = []
        for r in records:
            if r.get("control") != "swap" or r["id"] not in full:
                continue
            s = resolve(r, items[r["id"]])
            if s["truth"] is not None:
                pairs.append((full[r["id"]], s, gold_cell(items[r["id"]]) not in SYMMETRIC_CELLS))
        both = [f["pred"] == f["truth"] and s["pred"] == s["truth"] for f, s, _ in pairs]
        both_asym = [f["pred"] == f["truth"] and s["pred"] == s["truth"] for f, s, a in pairs if a]
        out["swap"] = {"n": len(pairs), "n_asymmetric": len(both_asym),
                       "both_orders_correct": sum(both) / len(both) if both else None,
                       "both_orders_correct_asymmetric": sum(both_asym) / len(both_asym) if both_asym else None}
    return out


# --------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------
def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def summary_md(m: Dict, title: str) -> str:
    L = [f"# {title}", "", f"{m['n_records']} records.", ""]
    for c, b in m["controls"].items():
        srcs, M = b["sources"], b["macro"]
        L += [f"## control `{c}` ({b['n_items']} items)", "",
              "| | macro | " + " | ".join(srcs) + " |", "|---|---:|" + "---:|" * len(srcs)]
        for label, key in (("detect (Same/Different macro-F1)", "detect"),
                           ("specify (exact, when correctly called Different)", "specify"),
                           ("strict (exact-answer macro-F1)", "strict")):
            cells = []
            for s in srcs:
                v = b["by_source"][s]
                txt = _pct(v[key])
                if key == "specify":
                    txt += f" (n {v['specify_n']}{', small' if v['specify_small'] else ''})"
                cells.append(txt)
            mac = _pct(M[key]) + ("†" if key == "specify" and M["specify_partial"] else "")
            L.append(f"| {label} | {mac} | " + " | ".join(cells) + " |")
        L += ["", "| | " + " | ".join(srcs) + " |", "|---|" + "---:|" * len(srcs),
              "| answers Different | " + " | ".join(_pct(b["by_source"][s]["says_different"]) for s in srcs) + " |",
              "| specify errors on the swap option | " + " | ".join(
                  _pct(b["by_source"][s]["specify_errors_swap_share"]) for s in srcs) + " |",
              "| parse failures | " + " | ".join(_pct(b["by_source"][s]["parse_failures"]) for s in srcs) + " |", ""]
    if "swap" in m:
        x = m["swap"]
        L += [f"Swap control (n={x['n']}): both orders correct {_pct(x['both_orders_correct'])} · asymmetric items "
              f"only {_pct(x['both_orders_correct_asymmetric'])} (n={x['n_asymmetric']})", ""]
    L.append("† specify macro over fewer sources than listed (sources with fewer than "
             f"{m['controls'][next(iter(m['controls']))]['min_specify_n']} Different calls are left out).")
    return "\n".join(L)


def load_results(path: Path) -> List[Dict]:
    with open(path) as f:
        d = json.load(f)
    return d["results"] if isinstance(d, dict) else d


def main():
    from .dataset import load_jsonl, load_dataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results JSON written by endocomparer.infer (or any model)")
    ap.add_argument("--dataset-root", default=None, help="score against the dataset (default split: test)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--jsonl", default=None, help="score against items in the JSONL schema instead")
    ap.add_argument("--min-specify-n", type=int, default=MIN_SPECIFY_N)
    ap.add_argument("--out-dir", default=None, help="default: the results file's directory")
    args = ap.parse_args()

    if args.jsonl:
        items = {it["id"]: it for it in load_jsonl(args.jsonl)}
    else:
        items = {it["id"]: it for it in load_dataset(args.split, args.dataset_root)}
    recs = load_results(Path(args.results))
    m = score(recs, items, args.min_specify_n)
    out = Path(args.out_dir) if args.out_dir else Path(args.results).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    md = summary_md(m, f"EndoDiffVQA metrics — {Path(args.results).name}")
    (out / "metrics.md").write_text(md)
    print(md)
    print(f"\nwrote {out / 'metrics.json'}, {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
