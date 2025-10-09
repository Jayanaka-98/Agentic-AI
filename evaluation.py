import json
import os

INPUT_JSONL = "results/all_users_answers.jsonl"
OUTPUT_TXT  = "results/eval_metrics.txt"

os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

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

macro_p = macro_p_sum / num_q if num_q else 0.0
macro_r = macro_r_sum / num_q if num_q else 0.0
macro_f1 = macro_f1_sum / num_q if num_q else 0.0
exact_match = num_em / num_q if num_q else 0.0

micro_p = 0.0 if (tp_micro + fp_micro) == 0 else tp_micro / (tp_micro + fp_micro)
micro_r = 0.0 if (tp_micro + fn_micro) == 0 else tp_micro / (tp_micro + fn_micro)
micro_f1 = f1(micro_p, micro_r)

report = []
report.append("Retrieval Evaluation Metrics")
report.append("============================")
report.append(f"Questions evaluated: {num_q}")
report.append(f"Exact Match (EM):   {exact_match:.4f}")
report.append("")
report.append("Macro Averages (mean over questions)")
report.append(f"  Precision:        {macro_p:.4f}")
report.append(f"  Recall:           {macro_r:.4f}")
report.append(f"  F1:               {macro_f1:.4f}")
report.append("")
report.append("Micro Averages (global counts)")
report.append(f"  Precision:        {micro_p:.4f}")
report.append(f"  Recall:           {micro_r:.4f}")
report.append(f"  F1:               {micro_f1:.4f}")
report.append("")
report.append("Counts")
report.append(f"  TP: {tp_micro}  FP: {fp_micro}  FN: {fn_micro}")

with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
    out.write("\n".join(report) + "\n")

print(f"[OK] wrote metrics to {OUTPUT_TXT}")
