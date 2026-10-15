"""Summarize the existing CIFAR comparisons; optionally verify frozen records.

This reads saved results only. It does not run models, select new controls or
change the per-draw tests. The generated paragraph is included by supplement.tex.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'results/tables'
OUT = ROOT / 'generated'
OUT.mkdir(exist_ok=True)
NAMES = {"FTBC96": "FTBC-96", "FTBCtuned": "FTBC-tuned"}
QUESTIONS = {"FTBC-96": "Ours96_vs_FTBC96", "FTBC-tuned": "Ours96_vs_FTBCtuned"}
DRAWS = ("C1", "C2", "C3")
WORDS = dict(enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")))


def summarize(verify_frozen=False):
    rows = list(csv.DictReader((HERE / "paired-comparisons.csv").open(newline="", encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        if row["dataset"].startswith("CIFAR") and row["candidate"] == "Ours-96" and row["control"] in QUESTIONS:
            groups[(row["dataset"], row["model"], int(row["T"]))].append(row)
    assert len(groups) == 12
    records, details = {}, []
    for key, group in sorted(groups.items()):
        primary = {r["draw"]: r for r in group if r["comparison"] == "threshold_d"}
        assert set(primary) == set(DRAWS)
        selected = primary["C1"]["control"]
        assert all(r["control"] == selected for r in primary.values())
        outcomes = {}
        for control, question in QUESTIONS.items():
            compared = {r["draw"]: r for r in group if r["comparison"] == question}
            if not compared:
                continue  # Identical tuned/default settings share the same run.
            assert set(compared) == set(DRAWS)
            outcomes[control] = [compared[d]["joint_improvement"] == "True" for d in DRAWS]
            for d in DRAWS:
                row = compared[d]
                joint = float(row["p_accuracy_gain_one_sided_unadjusted"]) < .05 and float(row["candidate_cross_entropy"]) < float(row["control_cross_entropy"])
                assert joint == outcomes[control][DRAWS.index(d)]
                if verify_frozen:
                    source = ROOT / "results/records" / Path(row["source_record"]).name
                    if source not in records:
                        records[source] = json.loads(source.read_text(encoding="utf-8"))
                    record = records[source]
                    cell = record["per_T"][f"T{key[2]}"]
                    raw = cell["questions"][question]["per_draw"][d]
                    assert record["passed"] and NAMES[cell["best_ftbc"]] == selected
                    assert cell["ftbc_total_correct"][cell["best_ftbc"]] == max(cell["ftbc_total_correct"].values())
                    assert joint == (raw["one_sided_p_candidate_better"] < .05 and raw["ce_candidate"] < raw["ce_control"])
                    assert float(row["p_accuracy_gain_one_sided_unadjusted"]) == raw["one_sided_p_candidate_better"]
                    assert float(row["candidate_cross_entropy"]) == raw["ce_candidate"]
                    assert float(row["control_cross_entropy"]) == raw["ce_control"]
        if len(outcomes) == 1:
            assert selected == "FTBC-96"
            if verify_frozen:
                assert cell["tuned_is_registered"]
            outcomes["FTBC-tuned"] = outcomes["FTBC-96"].copy()
        assert outcomes[selected] == [primary[d]["joint_improvement"] == "True" for d in DRAWS]
        other = next(c for c in QUESTIONS if c != selected)
        details.append({"dataset": key[0], "model": key[1], "T": key[2], "selected_control": selected,
                        "other_control": other, "joint_outcomes": outcomes,
                        "selected_qualifying_draws": sum(outcomes[selected]), "other_qualifying_draws": sum(outcomes[other]),
                        "different": outcomes[selected] != outcomes[other], "source_record": primary["C1"]["source_record"]})
    different = [r for r in details if r["different"]]
    fewer = [r for r in different if r["selected_qualifying_draws"] < r["other_qualifying_draws"]]
    more = [r for r in different if r["selected_qualifying_draws"] > r["other_qualifying_draws"]]
    assert len(fewer) + len(more) == len(different)
    overview = {"settings": len(details), "different_outcomes": len(different), "selected_fewer": len(fewer), "selected_more": len(more),
                "source": "paired-comparisons.csv", "frozen_records_verified": verify_frozen,
                "frozen_record_count": len(records), "new_model_execution": False, "details": details}
    lines = [f"Across {WORDS[len(different)]} CIFAR settings, the two FTBC configurations yield different draw-level joint-improvement outcomes (Table~\\ref{{tab:supp-locked-secondary}}).",
             f"Relative to the other configuration, the selected control yields fewer qualifying draws in {WORDS[len(fewer)]} settings and more in {WORDS[len(more)]}."]
    grouped = defaultdict(list)
    for r in fewer:
        grouped[(r["dataset"], r["model"])].append(r["T"])
    entries = [f"{dataset} {model} at $T=" + ",".join(map(str, times)) + "$" for (dataset, model), times in grouped.items()]
    lines.append("The reductions occur for " + " and ".join(entries) + ".")
    for r in more:
        lines.append(f"For {r['dataset']} {r['model']} at $T={r['T']}$, {r['selected_control']} is selected by aggregate accuracy, yet Ours-96 qualifies in {WORDS[r['selected_qualifying_draws']]} draws against it and in {WORDS[r['other_qualifying_draws']]} against {r['other_control']}.")
    all_other = sum(r["other_qualifying_draws"] == 3 and r["selected_qualifying_draws"] < 3 for r in details)
    lines.append(f"In {WORDS[all_other]} settings, Ours-96 qualifies in all three draws against the configuration that was not selected.")
    lines.append("The remaining CIFAR settings have identical joint-improvement outcomes for the two FTBC configurations.")
    example = [r for r in rows if r["dataset"] == "CIFAR-100" and r["model"] == "VGG-16" and r["T"] == "2" and r["comparison"] == "threshold_d"]
    assert sum(r["accuracy_test"] == "gain" for r in example) == 2
    assert all(r["cross_entropy_lower"] == "False" for r in example)
    lines.append("For CIFAR-100 VGG-16 at $T=2$, two draws have nominally significant accuracy gains against FTBC-tuned, but all three have higher cross-entropy; none therefore meets the joint criterion.")
    (OUT / "control-sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "control-sensitivity.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in overview.items() if k != "details"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-frozen", action="store_true")
    print(json.dumps(summarize(parser.parse_args().verify_frozen), indent=2))
