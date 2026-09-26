"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from SHAPE.stage3.contracts import write_json
from SHAPE.stage3.rank_semantics import _jsonable
from SHAPE.stage3.matched_shell import build_composition_resample_plan
from SHAPE.stage3.shell_support import audit_length_support, subset_target_test_days


def announce(message):
    print('[NYC->CA X1] ' + str(message), flush=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shell_plan(source_payload, target_payload, repeats, output, *, length_policy="strict"):
    retained, support = audit_length_support(
        source_payload["lengths"], target_payload["lengths"],
        target_payload["view"]["users"], length_policy)
    write_json(output / "preflight/day_length_support.json", support)
    missing = support["missing_CA_lengths"]
    if missing and length_policy == "strict":
        raise RuntimeError(f"CA test lengths have no exact NYC test donor: {missing}. "
                           "Strict mode preserved the full target population. For explicitly scoped subset scores, "
                           "rerun in a new output directory with --shell-length-policy common-support. "
                           "See preflight/day_length_support.json for excluded days/events/transition mass.")
    if not support["primary_events_ge_2_population"]["retained"]["days"]:
        raise RuntimeError("common support contains no events_ge_2 target days; no macro scores can be computed")
    if missing:
        counts = support["population"]
        announce(f"COMMON-SUPPORT SUBSET: retaining {counts['retained']['days']}/{counts['full']['days']} "
                 f"CA test days; excluding {counts['excluded']['days']} days, "
                 f"{counts['excluded']['events']} events and {counts['excluded']['within_day_transitions']} "
                 "transitions. These scores do not represent the full CA test population.")
    full_target = target_payload
    target_payload = subset_target_test_days(full_target, retained)
    plans, audit = build_composition_resample_plan(
        source_payload["lengths"], target_payload["lengths"], repeats=repeats,
        seed=42031, target_users=target_payload["view"]["users"])
    audit.update({"target_city": "ca", "source_city": "nyc",
                  "target_information": ["per_day_event_count", "user_grouping", "per_user_day_count"],
                  "measurement_scope": support["measurement_scope_if_run"],
                  "length_support": support,
                  "legacy_Paris_field_names_are_algorithm_identifiers": True})
    audit["maximum_source_day_reuse_per_repeat"] = [
        int(np.bincount(row, minlength=len(source_payload["lengths"])).max()) for row in plans]
    write_json(output / "preflight/shell_audit.json", _jsonable(audit))
    (output / "data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "data/target_day_selection.npz",
                        retained_day_indices=retained,
                        full_target_lengths=full_target["lengths"],
                        full_target_users=full_target["view"]["users"],
                        **({"full_target_dates": full_target["view"]["dates"]}
                           if "dates" in full_target["view"] else {}))
    np.savez_compressed(output / "data/shell_plans.npz", plans=plans,
                        target_lengths=target_payload["lengths"],
                        target_users=target_payload["view"]["users"])
    return plans, audit, retained
