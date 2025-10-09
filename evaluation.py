import json
import os

INPUT_JSONL = "results/all_users_answers.jsonl"   # change if needed
OUTPUT_JSON  = "results/eval_metrics.json"

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

def as_set(x):
    if x is None:
        return set()
    if isinstance(x, list):
        return set(str(i) for i in x)
    return {str(x)}

def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

tp_micro = fp_micro = fn_micro = 0
num_q = num_em = 0
macro_p_sum = macro_r_sum = macro_f1_sum = 0.0

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        # Skip error lines (e.g., login failures)
        if "error" in rec:
            continue

        gold = as_set(rec.get("answer")) or as_set(rec.get("answer_gold"))
        pred = as_set(rec.get("memory_ids")) or as_set(rec.get("output"))

        inter = gold & pred
        tp = len(inter)
        fp = len(pred - gold)
        fn = len(gold - pred)

        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec_ = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1_  = f1(prec, rec_)
        em   = int(pred == gold)

        macro_p_sum += prec
        macro_r_sum += rec_
        macro_f1_sum += f1_
        num_em += em
        num_q += 1

        tp_micro += tp
        fp_micro += fp
        fn_micro += fn

# Macro (mean over questions)
macro_p = macro_p_sum / num_q if num_q else 0.0
macro_r = macro_r_sum / num_q if num_q else 0.0
macro_f1 = macro_f1_sum / num_q if num_q else 0.0
exact_match = num_em / num_q if num_q else 0.0

# Micro (aggregate counts)
micro_p = 0.0 if (tp_micro + fp_micro) == 0 else tp_micro / (tp_micro + fp_micro)
micro_r = 0.0 if (tp_micro + fn_micro) == 0 else tp_micro / (tp_micro + fn_micro)
micro_f1 = f1(micro_p, micro_r)

metrics = {
    "questions_evaluated": num_q,
    "exact_match": exact_match,
    "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
    "counts": {"tp": tp_micro, "fp": fp_micro, "fn": fn_micro}
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    json.dump(metrics, out, indent=2, ensure_ascii=False)

print(f"[OK] wrote metrics to {OUTPUT_JSON}")