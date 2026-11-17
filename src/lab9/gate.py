"""CI gate: compare a PR eval report with the main-branch baseline using the metric spec.

    python -m lab9.gate --pr reports/eval_workflow.json --baseline reports/baseline_v1.json

exit 0 = merge, exit 1 = blocked. Safety metrics: max 0, no averaging away.
Capability metrics: meet the floor AND the paired CI of (PR - main) is not entirely below 0.
Thresholds may only get stricter on the same eval-set version (ratchet).
"""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path

from lab8.console import utf8_stdio
from lab9 import stats
from lab9.eval import SPEC, load_spec, per_item


def integrity_check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> list[str]:
    """M7 (Expert): refuse to compare reports that cannot be trusted. Return a list of problems
    (empty = trustworthy). A gate that passes a broken report is worse than no gate.
    Must catch at least:
      - any metric value in pr["metrics"] that is NaN or inf
      - len(pr["rows"]) != manifest items x len(manifest seeds)   (dropped / duplicated rows)
      - the (item_id, seed) set of PR differs from the baseline's (cherry-picked seeds/items)
      - pr["evaluator_failures"] non-empty, or a row whose value for a graded spec metric is
        None (a crashed grader silently scores as "missing" and drops out of the mean)
      - base_spec given and a threshold in spec is LOOSER than in base_spec for the same
        eval-set version (ratchet: max may not rise, min may not fall, a blocking metric may
        not become non-blocking, a metric may not disappear; the same for spec["slices"],
        including dropping no_significant_drop)
      - an item's slice tags in PR rows differ from the baseline's (relabelling an item out of a
        slice hides its regression from the slice gate)"""
    problems: list[str] = []
    from lab9.graders import GRADERS
    manifest = pr.get("manifest") or {}
    rows = pr.get("rows")
    if not isinstance(rows, list):
        return ["rows is missing or not a list"]
    items, seeds = manifest.get("items"), manifest.get("seeds")
    if not isinstance(items, int) or isinstance(items, bool) or items < 0 or not isinstance(seeds, list):
        problems.append("manifest items/seeds are invalid")
    else:
        valid_seeds = all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        if not valid_seeds:
            problems.append("manifest seeds must be integers")
        elif len(set(seeds)) != len(seeds):
            problems.append("manifest seeds contain duplicates")
        if len(rows) != items * len(seeds):
            problems.append(f"row count {len(rows)} != manifest items x seeds {items * len(seeds)}")

    def index_rows(report: dict, label: str) -> tuple[list[tuple], dict[tuple, dict]]:
        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list):
            problems.append(f"{label} rows is missing or not a list")
            return [], {}
        pairs: list[tuple] = []
        indexed: dict[tuple, dict] = {}
        for row in raw_rows:
            if not isinstance(row, dict) or "item_id" not in row or "seed" not in row:
                problems.append(f"{label} row is missing item_id or seed")
                continue
            if not isinstance(row["item_id"], str) or not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
                problems.append(f"{label} row has invalid item_id or seed")
                continue
            pair = (row["item_id"], row["seed"])
            pairs.append(pair)
            if pair in indexed:
                problems.append(f"{label} duplicate row for {pair[0]}@{pair[1]}")
            indexed[pair] = row
        return pairs, indexed

    pr_pairs, pr_index = index_rows(pr, "PR")
    _, base_index = index_rows(base, "baseline")
    if set(pr_pairs) != set(base_index):
        problems.append("PR (item_id, seed) set differs from baseline")

    metrics = pr.get("metrics")
    if not isinstance(metrics, dict):
        problems.append("metrics is missing or not a map")
    else:
        for name, value in metrics.items():
            if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
                problems.append(f"metric {name} is non-finite")
    failures = pr.get("evaluator_failures")
    if failures:
        count = len(failures) if hasattr(failures, "__len__") else 1
        problems.append(f"evaluator_failures has {count} failure(s)")
    if isinstance(metrics, dict):
        for metric in spec.get("metrics", []):
            mid = metric.get("id")
            if mid in GRADERS:
                if metrics.get(mid) is None:
                    problems.append(f"graded metric {mid} is missing/None in metrics")
                for pair, row in pr_index.items():
                    if row.get(mid) is None:
                        problems.append(f"graded metric {mid} is None at {pair[0]}@{pair[1]}")
                        break

    def slice_tags(indexed: dict[tuple, dict], label: str) -> dict[str, tuple[str, ...]]:
        by_item: dict[str, tuple[str, ...]] = {}
        for (item_id, _seed), row in indexed.items():
            raw = row.get("slices") or []
            if not isinstance(raw, list):
                problems.append(f"{label} slices for {item_id} is not a list")
                tags: tuple[str, ...] = ()
            else:
                if any(not isinstance(tag, str) for tag in raw):
                    problems.append(f"{label} slices for {item_id} contain non-string tags")
                tags = tuple(sorted(tag for tag in raw if isinstance(tag, str)))
                if len(set(tags)) != len(tags):
                    problems.append(f"{label} slices for {item_id} contain duplicate tags")
            if item_id in by_item and by_item[item_id] != tags:
                problems.append(f"{label} slice tags vary across seeds for {item_id}")
            by_item[item_id] = tags
        return by_item

    if slice_tags(pr_index, "PR") != slice_tags(base_index, "baseline"):
        problems.append("PR item slice tags differ from baseline")

    # Ratchet only compares specs for the same eval-set version. A new version
    # gets a new baseline and starts a new threshold history.
    if base_spec and spec.get("eval_set_version") == base_spec.get("eval_set_version"):
        old_metrics = {m.get("id"): m for m in base_spec.get("metrics", [])}
        new_metrics = {m.get("id"): m for m in spec.get("metrics", [])}
        for mid, old in old_metrics.items():
            new = new_metrics.get(mid)
            if new is None:
                problems.append(f"ratchet: metric {mid} disappeared")
                continue
            old_th, new_th = old.get("threshold") or {}, new.get("threshold") or {}
            for key in ("min", "max", "max_increase"):
                if key in old_th and key not in new_th:
                    problems.append(f"ratchet: metric {mid} removed {key} threshold")
                elif key in old_th and key in new_th:
                    loosened = ((key in {"min"} and new_th[key] < old_th[key]) or
                                (key in {"max", "max_increase"} and new_th[key] > old_th[key]))
                    if loosened:
                        problems.append(f"ratchet: metric {mid} loosened {key} threshold")
            if old_th.get("no_significant_drop") and not new_th.get("no_significant_drop"):
                problems.append(f"ratchet: metric {mid} removed no_significant_drop")
            if old.get("blocking") and not new.get("blocking"):
                problems.append(f"ratchet: metric {mid} changed from blocking to non-blocking")

        old_slices = {s.get("name"): s for s in base_spec.get("slices", [])}
        new_slices = {s.get("name"): s for s in spec.get("slices", [])}
        for name, old in old_slices.items():
            new = new_slices.get(name)
            if new is None:
                problems.append(f"ratchet: slice {name} disappeared")
                continue
            if old.get("metric") != new.get("metric"):
                problems.append(f"ratchet: slice {name} changed metric")
            for key in ("min", "max"):
                if key in old and key not in new:
                    problems.append(f"ratchet: slice {name} removed {key} threshold")
                elif key in old and key in new:
                    loosened = (key == "min" and new[key] < old[key]) or (key == "max" and new[key] > old[key])
                    if loosened:
                        problems.append(f"ratchet: slice {name} loosened {key} threshold")
            if old.get("no_significant_drop") and not new.get("no_significant_drop"):
                problems.append(f"ratchet: slice {name} removed no_significant_drop")
            if old.get("blocking") and not new.get("blocking"):
                problems.append(f"ratchet: slice {name} changed from blocking to non-blocking")
    return problems


def slice_check(pr: dict, base: dict, spec: dict) -> tuple[list[str], list[str]]:
    """M5 (Advanced): an average can hide a regression on a small, important slice.
    For each entry of spec["slices"] ({"name", "metric", "min"?, "no_significant_drop"?, "blocking"}):
      - value = mean of per_item(pr, metric, slice_name=name)
      - fail if below "min", or if no_significant_drop and the paired CI over the slice's items
        lies entirely below 0
    Return (blocked ids like "slice:money_movement/pass_rate", report lines)."""
    blocked: list[str] = []
    lines: list[str] = []
    bs = spec.get("bootstrap", {})
    for rule in spec.get("slices", []):
        name, metric = rule["name"], rule["metric"]
        current = per_item(pr, metric, slice_name=name)
        previous = per_item(base, metric, slice_name=name)
        reasons: list[str] = []
        if not current:
            value_text = "MISSING"
            if "min" in rule:
                reasons.append("no PR items in slice")
        else:
            value = sum(current.values()) / len(current)
            value_text = f"{value:.3f}"
            if "min" in rule and value < rule["min"]:
                reasons.append(f"below min {rule['min']}")

        ci_note = ""
        if rule.get("no_significant_drop"):
            keys = sorted(set(current) & set(previous))
            if not keys:
                reasons.append("no paired items in slice")
            else:
                d, lo, hi = stats.paired_bootstrap(
                    [current[k] for k in keys], [previous[k] for k in keys],
                    n_resamples=bs.get("n_resamples", 1000), rng=bs.get("rng", 2027),
                    confidence=bs.get("confidence", 0.95),
                )
                ci_note = f"Δ {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
                if hi < 0:
                    reasons.append("significant drop")

        ident = f"slice:{name}/{metric}"
        if reasons and rule.get("blocking"):
            blocked.append(ident)
        verdict = "BLOCK" if reasons and rule.get("blocking") else ("warn" if reasons else "ok")
        lines.append(f"  {ident:<38} {value_text:<8} {ci_note:<28} {verdict}" +
                     (f" ({'; '.join(reasons)})" if reasons else ""))
    return blocked, lines


def check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> tuple[list[str], list[str]]:
    if pr["manifest"]["eval_set_sha"] != base["manifest"]["eval_set_sha"]:
        return ["eval_set_sha"], ["  EVAL SET differs from the baseline's: re-run the baseline on the same eval set"]
    problems = integrity_check(pr, base, spec, base_spec)
    if problems:
        return ["integrity"], [f"  INTEGRITY {p}" for p in problems]
    blocked, lines = [], []
    bs = spec.get("bootstrap", {})
    for m in spec["metrics"]:
        mid, th, val = m["id"], m.get("threshold", {}), pr["metrics"].get(m["id"])
        verdict, note = "ok", ""
        if val is None:
            verdict = "missing"
        elif "max" in th and val > th["max"]:
            verdict = "fail"
        elif "min" in th and val < th["min"]:
            verdict = "fail"
        elif "max_increase" in th and base["metrics"].get(mid):
            if val > base["metrics"][mid] * (1 + th["max_increase"]):
                verdict = "fail"
        if th.get("no_significant_drop") and val is not None:
            a, b = per_item(pr, mid), per_item(base, mid)
            keys = sorted(set(a) & set(b))
            d, lo, hi = stats.paired_bootstrap([a[k] for k in keys], [b[k] for k in keys],
                                               n_resamples=bs.get("n_resamples", 1000), rng=bs.get("rng", 2027))
            note = f"Δ {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
            if hi < 0:
                verdict = "fail"
        if verdict != "ok" and m.get("blocking"):
            blocked.append(mid)
        label = {"ok": "ok", "missing": "MISSING"}.get(verdict, "BLOCK" if m.get("blocking") else "warn")
        lines.append(f"  {mid:<22} {val!s:<8} {json.dumps(th):<40} {note:<28} {label}")
    if spec.get("slices"):
        sb, sl = slice_check(pr, base, spec)
        blocked += sb
        lines += sl
    return blocked, lines


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(prog="lab9.gate")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--base-spec", help="metric spec on main, for the ratchet check (M7)")
    args = ap.parse_args(argv)
    pr = json.loads(Path(args.pr).read_text(encoding="utf-8"))
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_spec = load_spec(Path(args.base_spec)) if args.base_spec else None
    blocked, lines = check(pr, base, load_spec(Path(args.spec)), base_spec)
    print(f"gate: PR {pr['run_id']} vs main {base['run_id']} · eval set {pr['manifest']['eval_set_sha']}")
    print("\n".join(lines))
    if blocked:
        print(f"exit 1 — blocked: {', '.join(blocked)}")
        return 1
    print("exit 0 — merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
