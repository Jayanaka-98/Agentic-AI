import json
import os
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


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
# Core evaluation per run
# -------------------------------------------------
def _evaluate_single_run(INPUT_JSONL: Path, base_dir: Path, exp_type: str):
    tp_micro = fp_micro = fn_micro = 0
    num_q = num_em = 0
    macro_p_sum = macro_r_sum = macro_f1_sum = 0.0
    cos_gold_sum = cos_pred_sum = num_cos = 0

    SIM_JSONL = base_dir / f"{exp_type}_similarities.jsonl"

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

    if num_q == 0:
        return None

    macro_f1 = macro_f1_sum / num_q
    micro_p = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) else 0.0
    micro_r = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) else 0.0
    micro_f1 = f1(micro_p, micro_r)
    mean_cosine_pred = cos_pred_sum / num_cos if num_cos else 0.0

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "cosine_pred": mean_cosine_pred,
    }


# -------------------------------------------------
# Aggregate across runs
# -------------------------------------------------
def evaluate_results(exp_type: str):
    base_dir = Path("results") / exp_type
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run")])
    if not run_dirs:
        run_dirs = [base_dir]

    all_run_metrics = []

    for run_dir in run_dirs:
        INPUT_JSONL = run_dir / "all_users_answers.jsonl"
        if not INPUT_JSONL.exists():
            print(f"[WARN] Missing file: {INPUT_JSONL}")
            continue
        metrics = _evaluate_single_run(INPUT_JSONL, run_dir, exp_type)
        if metrics:
            all_run_metrics.append(metrics)

    if not all_run_metrics:
        return None

    macro_f1_all = [m["macro_f1"] for m in all_run_metrics]
    micro_f1_all = [m["micro_f1"] for m in all_run_metrics]
    cosine_pred_all = [m["cosine_pred"] for m in all_run_metrics]

    avg_metrics = {
        "exp": exp_type,
        "macro_f1": np.mean(macro_f1_all),
        "micro_f1": np.mean(micro_f1_all),
        "cosine_pred": np.mean(cosine_pred_all),
        "macro_f1_all": macro_f1_all,
        "micro_f1_all": micro_f1_all,
        "cosine_pred_all": cosine_pred_all,
    }

    print(f"[OK] {exp_type}: mean ± std")
    print(f"  Macro F1   = {np.mean(macro_f1_all):.4f} ± {np.std(macro_f1_all):.4f}")
    print(f"  Micro F1   = {np.mean(micro_f1_all):.4f} ± {np.std(micro_f1_all):.4f}")
    print(f"  Cosine Sim = {np.mean(cosine_pred_all):.4f} ± {np.std(cosine_pred_all):.4f}")
    print()
    return avg_metrics


# -------------------------------------------------
# Visualization (true boxplots)
# -------------------------------------------------
def visualize_overall_comparison(results):
    if not results:
        print("[WARN] No experiment results to compare.")
        return

    labels = [r["exp"].upper() for r in results]
    macro_all = [r["macro_f1_all"] for r in results]
    micro_all = [r["micro_f1_all"] for r in results]
    cosine_all = [r["cosine_pred_all"] for r in results]

    def get_ylim(data_lists, pad=0.05):
        """Compute adaptive ylim with small padding."""
        vals = np.concatenate(data_lists)
        vmin, vmax = np.min(vals), np.max(vals)
        rng = vmax - vmin
        return max(0, vmin - pad * rng), min(1, vmax + pad * rng)

    plt.figure(figsize=(12, 4))

    # --- Macro F1 ---
    plt.subplot(1, 3, 1)
    plt.boxplot(macro_all, vert=True, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.7))
    plt.title("Macro F1")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylim(*get_ylim(macro_all))
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # --- Micro F1 ---
    plt.subplot(1, 3, 2)
    plt.boxplot(micro_all, vert=True, patch_artist=True,
                boxprops=dict(facecolor="darkorange", alpha=0.7))
    plt.title("Micro F1")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylim(*get_ylim(micro_all))
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # --- Cosine Similarity ---
    plt.subplot(1, 3, 3)
    plt.boxplot(cosine_all, vert=True, patch_artist=True,
                boxprops=dict(facecolor="seagreen", alpha=0.7))
    plt.title("Cosine Similarity (Pred vs Gold)")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylim(*get_ylim(cosine_all))
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Cross-Experiment Metric Comparison (Boxplots over 10 runs)", fontsize=14, y=1.05)
    plt.tight_layout()
    out_path = Path("results") / "metrics_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved zoomed boxplot → {out_path}")

def visualize_normalized_gain(results):
    """Plot normalized metrics (PE as baseline = 1.0) as vertical boxplots."""
    if not results:
        print("[WARN] No results to normalize.")
        return

    base = next((r for r in results if r["exp"].lower() == "pe"), None)
    if not base:
        print("[WARN] PE baseline not found; skipping normalization.")
        return

    def norm_list(values, base_vals):
        b = np.mean(base_vals)
        return [v / b if b != 0 else 0.0 for v in values]

    labels = [r["exp"].upper() for r in results]
    macro_gain = [norm_list(r["macro_f1_all"], base["macro_f1_all"]) for r in results]
    micro_gain = [norm_list(r["micro_f1_all"], base["micro_f1_all"]) for r in results]
    cos_gain = [norm_list(r["cosine_pred_all"], base["cosine_pred_all"]) for r in results]

    plt.figure(figsize=(12, 4))
    for i, (data, title, color) in enumerate([
        (macro_gain, "Macro F1 Gain (vs PE)", "royalblue"),
        (micro_gain, "Micro F1 Gain (vs PE)", "darkorange"),
        (cos_gain, "Cosine Gain (vs PE)", "seagreen")
    ]):
        plt.subplot(1, 3, i + 1)
        plt.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7))
        plt.title(title)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.axhline(1.0, color="gray", linestyle="--", lw=1)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Normalized Metric Gains (Boxplot, PE = 1.0)", fontsize=14, y=1.05)
    plt.tight_layout()
    out_path = Path("results") / "metrics_normalized_gain_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved normalized gain boxplot → {out_path}")


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
