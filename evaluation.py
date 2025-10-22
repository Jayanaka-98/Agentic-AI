import json
import os
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def cosine_sim(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if not a.strip() and not b.strip():
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    A = vec.transform([a])
    B = vec.transform([b])
    return float(cosine_similarity(A, B)[0, 0])


def as_set(x):
    if x is None:
        return set()
    if isinstance(x, list):
        return set(str(i) for i in x)
    return {str(x)}


def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


# -------------------------------------------------
# Cache loaders
# -------------------------------------------------
mem_cache = {}
qa_gold_text_cache = {}


def load_user_mem_summaries(localpart: str):
    if localpart in mem_cache:
        return mem_cache[localpart]
    pth = f"./test_data/{localpart}.json"
    mapping = {}
    if os.path.exists(pth):
        try:
            with open(pth, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rep in data.get("reports", []):
                ctx = rep.get("context", {})
                mid = ctx.get("memory_id") or rep.get("id")
                summ = ctx.get("summary", "")
                if isinstance(mid, str) and mid:
                    mapping[mid] = summ or ""
        except Exception:
            mapping = {}
    mem_cache[localpart] = mapping
    return mapping


def load_user_qa_gold_text(localpart: str):
    if localpart in qa_gold_text_cache:
        return qa_gold_text_cache[localpart]
    pth = f"./qa/{localpart}.json"
    qmap = {}
    if os.path.exists(pth):
        try:
            with open(pth, "r", encoding="utf-8") as f:
                data = json.load(f)
            for qa in data.get("qa_pairs", []):
                qid = qa.get("qid")
                gold_text = qa.get("answer_gold", "")
                if qid:
                    qmap[qid] = gold_text or ""
        except Exception:
            qmap = {}
    qa_gold_text_cache[localpart] = qmap
    return qmap


# -------------------------------------------------
# Evaluation pipeline (returns metrics)
# -------------------------------------------------
def evaluate_results(exp_type: str):
    base_dir = Path("results") / exp_type
    os.makedirs(base_dir, exist_ok=True)

    INPUT_JSONL = base_dir / "all_users_answers.jsonl"
    OUTPUT_TXT = base_dir / f"{exp_type}_eval_metrics.txt"
    SIM_JSONL = base_dir / f"{exp_type}_similarities.jsonl"

    tp_micro = fp_micro = fn_micro = 0
    num_q = num_em = 0
    macro_p_sum = macro_r_sum = macro_f1_sum = 0.0
    cos_gold_sum = cos_pred_sum = num_cos = 0

    if not os.path.exists(INPUT_JSONL):
        print(f"[WARN] Missing input file: {INPUT_JSONL}")
        return None

    with open(SIM_JSONL, "w", encoding="utf-8") as sim_out, open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue

            user_email = rec.get("user", "")
            if "@" not in user_email:
                continue
            localpart = user_email.split("@", 1)[0]

            qid = rec.get("qid", "")
            gold_ids = as_set(rec.get("answer")) or as_set(rec.get("answer_gold"))
            pred_ids = as_set(rec.get("memory_ids")) or as_set(rec.get("output"))

            inter = gold_ids & pred_ids
            tp, fp, fn = len(inter), len(pred_ids - gold_ids), len(gold_ids - pred_ids)

            prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            rec_ = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            f1_ = f1(prec, rec_)
            em = int(pred_ids == gold_ids)

            macro_p_sum += prec
            macro_r_sum += rec_
            macro_f1_sum += f1_
            num_em += em
            num_q += 1

            tp_micro += tp
            fp_micro += fp
            fn_micro += fn

            mem_map = load_user_mem_summaries(localpart)
            qa_map = load_user_qa_gold_text(localpart)
            gold_text_answer = qa_map.get(qid, "")
            gold_concat = " ".join(mem_map.get(mid, "") for mid in gold_ids)
            pred_concat = " ".join(mem_map.get(mid, "") for mid in pred_ids)

            cos_gold = cosine_sim(gold_concat, gold_text_answer)
            cos_pred = cosine_sim(pred_concat, gold_text_answer)

            cos_gold_sum += cos_gold
            cos_pred_sum += cos_pred
            num_cos += 1

            sim_out.write(json.dumps({
                "user": user_email,
                "qid": qid,
                "cosine_gold_vs_answer": cos_gold,
                "cosine_pred_vs_answer": cos_pred,
            }, ensure_ascii=False) + "\n")

    # ---- Aggregate metrics ----
    macro_p = macro_p_sum / num_q if num_q else 0.0
    macro_r = macro_r_sum / num_q if num_q else 0.0
    macro_f1 = macro_f1_sum / num_q if num_q else 0.0
    exact_match = num_em / num_q if num_q else 0.0
    micro_p = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) else 0.0
    micro_r = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) else 0.0
    micro_f1 = f1(micro_p, micro_r)
    mean_cosine_pred = cos_pred_sum / num_cos if num_cos else 0.0
    mean_cosine_gold = cos_gold_sum / num_cos if num_cos else 0.0

    # ---- Write report in your desired format ----
    with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
        out.write(f"Retrieval Evaluation Metrics ({exp_type.upper()})\n")
        out.write("=" * 50 + "\n")
        out.write(f"Questions evaluated: {num_q}\n")
        out.write(f"Exact Match (EM):   {exact_match:.4f}\n\n")

        out.write("Macro Averages (mean over questions)\n")
        out.write(f"  Precision: {macro_p:.4f}\n")
        out.write(f"  Recall: {macro_r:.4f}\n")
        out.write(f"  F1: {macro_f1:.4f}\n\n")

        out.write("Micro Averages (global counts)\n")
        out.write(f"  Precision: {micro_p:.4f}\n")
        out.write(f"  Recall: {micro_r:.4f}\n")
        out.write(f"  F1: {micro_f1:.4f}\n\n")

        out.write("Cosine Similarity\n")
        out.write(f"  Mean cosine(pred summaries vs gold answers): {mean_cosine_pred:.4f}\n")
        out.write(f"  Mean cosine(gold summaries vs gold answers): {mean_cosine_gold:.4f}\n\n")

        out.write("Counts\n")
        out.write(f"  TP: {tp_micro}\n")
        out.write(f"  FP: {fp_micro}\n")
        out.write(f"  FN: {fn_micro}\n")

    print(f"[OK] {exp_type}: metrics written to {OUTPUT_TXT}")

    return {
        "exp": exp_type,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "cosine_pred": mean_cosine_pred,
    }


# -------------------------------------------------
# Side-by-side comparison chart
# -------------------------------------------------
def visualize_overall_comparison(results):
    if not results:
        print("[WARN] No experiment results to compare.")
        return

    labels = [r["exp"].upper() for r in results]
    macro_f1 = [r["macro_f1"] for r in results]
    micro_f1 = [r["micro_f1"] for r in results]
    cosine_pred = [r["cosine_pred"] for r in results]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    bars = plt.bar(labels, macro_f1, color="steelblue")
    plt.title("Macro F1")
    plt.ylim(0, 1)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 2)
    bars = plt.bar(labels, micro_f1, color="darkorange")
    plt.title("Micro F1")
    plt.ylim(0, 1)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    bars = plt.bar(labels, cosine_pred, color="seagreen")
    plt.title("Cosine Similarity (Pred vs Gold)")
    plt.ylim(0, 1)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Cross-Experiment Metric Comparison", fontsize=14, y=1.05)
    plt.tight_layout()
    out_path = Path("results") / "metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved comparison chart → {out_path}")

def visualize_normalized_gain(results):
    """Plot normalized metrics (PE as baseline = 1.0)."""
    if not results:
        print("[WARN] No results to normalize.")
        return

    # --- Identify baseline (PE) ---
    base = next((r for r in results if r["exp"].lower() == "pe"), None)
    if not base:
        print("[WARN] PE baseline not found; skipping normalization.")
        return

    # --- Compute normalized values ---
    def norm(x, base_val):
        return x / base_val if base_val != 0 else 0.0

    labels = [r["exp"].upper() for r in results]
    macro_gain = [norm(r["macro_f1"], base["macro_f1"]) for r in results]
    micro_gain = [norm(r["micro_f1"], base["micro_f1"]) for r in results]
    cos_gain = [norm(r["cosine_pred"], base["cosine_pred"]) for r in results]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    bars = plt.bar(labels, macro_gain, color="royalblue")
    plt.title("Macro F1 Gain (vs PE)")
    plt.ylim(0, max(macro_gain) * 1.2)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{b.get_height():.2f}×", ha="center")
    plt.axhline(1.0, color="gray", linestyle="--", lw=1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 2)
    bars = plt.bar(labels, micro_gain, color="darkorange")
    plt.title("Micro F1 Gain (vs PE)")
    plt.ylim(0, max(micro_gain) * 1.2)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{b.get_height():.2f}×", ha="center")
    plt.axhline(1.0, color="gray", linestyle="--", lw=1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    bars = plt.bar(labels, cos_gain, color="seagreen")
    plt.title("Cosine Gain (vs PE)")
    plt.ylim(0, max(cos_gain) * 1.2)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{b.get_height():.2f}×", ha="center")
    plt.axhline(1.0, color="gray", linestyle="--", lw=1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Normalized Metric Gains (PE = 1.0)", fontsize=14, y=1.05)
    plt.tight_layout()
    out_path = Path("results") / "metrics_normalized_gain.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved normalized gain chart → {out_path}")
    

# -------------------------------------------------
# Main entry
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize retrieval results.")
    parser.add_argument("--exp", default="all", choices=["all", "mtp", "mtp_with_sem", "pe"],
                        help="Which experiment to evaluate. Use 'all' to evaluate and compare all three.")
    args = parser.parse_args()

    if args.exp == "all":
        results = []
        for exp in ["mtp", "mtp_with_sem", "pe"]:
            r = evaluate_results(exp)
            if r:
                results.append(r)
        visualize_overall_comparison(results)
        visualize_normalized_gain(results)
    else:
        r = evaluate_results(args.exp)
        if r:
            visualize_overall_comparison([r])
