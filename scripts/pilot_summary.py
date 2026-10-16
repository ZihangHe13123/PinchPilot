"""Summarize the internal pilot: task records, questionnaires and the unblinding key.

Usage: python scripts/pilot_summary.py <PinchPilot-data>/pilot <key.json>

Which record counts, decided before the key was opened: for each person, setting and task
the first record; if that run ended because the camera input went stale, the run that
continued it. Later redos do not count, and a run stopped at the two-minute limit counts as
it stands. Scrolling uses the V gesture, not the contact source, and is left out.
"""

import csv
import json
import sys
from pathlib import Path

TASKS = ("click", "double_click", "drag", "reverse", "right_click")
NAMES = {"thresholds": "现有规则（固定阈值）", "rules": "调过的规则", "cnn": "时序 CNN"}
TECHNICAL = ("stale_input", "source_changed")
TLX = ("mental", "physical", "temporal", "performance", "effort", "frustration")


def counted(records):
    """{task: record} by the rule in the module docstring."""
    chosen = {}
    for task in TASKS:
        runs = [r for r in records if r["task"] == task]
        index = 0
        while index < len(runs) - 1 and runs[index]["reason"] in TECHNICAL:
            index += 1
        if runs:
            chosen[task] = runs[index]
    return chosen


def main(pilot, key_path):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))["key"]
    rows = []
    for person in sorted(key):
        folder = Path(pilot) / person
        records = []
        for path in sorted(folder.glob("practice_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            alias = (record.get("engine_config", {}).get("contact_model_summary") or {}).get(
                "alias"
            )
            records.append((alias, record))
        answers = {
            row["setting"]: row
            for row in csv.DictReader((folder / "questionnaire.csv").open(encoding="utf-8"))
        }
        for letter, kind in sorted(key[person].items()):
            mine = [r for alias, r in records if alias == f"设置 {letter}"]
            chosen = counted(mine)
            answer = answers[letter]
            rows.append(
                {
                    "person": person,
                    "letter": letter,
                    "order": "ABC".index(letter) + 1,
                    "kind": kind,
                    "tasks_completed": sum(r["status"] == "completed" for r in chosen.values()),
                    "targets": sum(r["completed_targets"] for r in chosen.values()),
                    "targets_total": sum(r["total_targets"] for r in chosen.values()),
                    "seconds": round(sum(r["duration_s"] for r in chosen.values()), 1),
                    "misses": sum(r["misses"] for r in chosen.values()),
                    "left_misses": sum(chosen[t]["misses"] for t in TASKS[:4] if t in chosen),
                    "right_seconds": round(chosen["right_click"]["duration_s"], 1),
                    "right_misses": chosen["right_click"]["misses"],
                    "right_done": chosen["right_click"]["completed_targets"],
                    "tlx": round(sum(float(answer[k]) for k in TLX) / len(TLX), 1),
                    "rank": int(answer["rank"]),
                    "comment": answer["comment"],
                    "redone_runs": len(mine)
                    - len(chosen)
                    - sum(r["task"] == "scroll" for r in mine),
                }
            )
    return rows


if __name__ == "__main__":
    out = main(sys.argv[1], sys.argv[2])
    print(json.dumps(out, ensure_ascii=False, indent=1))
