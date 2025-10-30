import os
import random
import tempfile
import jaclang
import requests
import urllib3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from byllm.lib import Image as ByImage
from image_extraction import extract_memory_details

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
N_SAMPLES = 300
HYBRID_ALPHA = 0.6

# -------------------------------------------------
# 1. Dataset
# -------------------------------------------------
def load_laion_stream():
    """Return the LAION streaming dataset."""
    return load_dataset("laion/laion400m", split="train", streaming=True)


# -------------------------------------------------
# 2. Image Download
# -------------------------------------------------
def safe_download_image(url: str, timeout: int = 5) -> str | None:
    """Safely download an image and store temporarily."""
    try:
        resp = requests.get(url, timeout=timeout, verify=False)
        if resp.status_code != 200 or not resp.content:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception as e:
        print("Download failed:", e)
        return None


# -------------------------------------------------
# 3. Run Extraction (keep total valid count = n)
# -------------------------------------------------
def run_extraction(target_n=300, exp_type="mtp"):
    """
    Run the image extraction pipeline until `target_n` valid samples are collected.
    Continues sampling the LAION stream until enough valid results are gathered.
    """
    ds = load_laion_stream()
    results = []
    processed = 0

    for example in tqdm(ds, desc=f"Extracting {target_n} valid samples ({exp_type})"):
        if len(results) >= target_n:
            break
        processed += 1

        try:
            img_path = safe_download_image(example["url"])
            if not img_path:
                continue

            img = ByImage(img_path)
            use_byllm = False if exp_type == "pe" else True

            # Extraction
            res = extract_memory_details(img)

            if not res or not getattr(res, "summary", "").strip():
                continue

            results.append({
                "caption": example.get("caption", ""),
                "summary": res.summary,
                "who": res.who,
                "what": res.what,
                "when": res.when,
                "location_type": res.location_type,
                "follow_up_questions": res.follow_up_questions,
                "image_path": img_path,
                "exp_type": exp_type,
            })

        except Exception as e:
            print(f"Error processing ({exp_type}):", e)
            continue

    print(f"✅ Collected {len(results)} valid samples out of {processed} processed ({exp_type}).")
    return results


# -------------------------------------------------
# 4. Cosine Similarity
# -------------------------------------------------
def compute_cosine_similarity(captions, summaries):
    if not captions:
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    capt_embs = model.encode(captions, convert_to_tensor=True)
    sum_embs = model.encode(summaries, convert_to_tensor=True)
    sims = util.cos_sim(capt_embs, sum_embs).diagonal().cpu().numpy()
    return sims.tolist()


# -------------------------------------------------
# 5. BM25
# -------------------------------------------------
def compute_bm25_scores(captions, summaries):
    if not captions:
        return []
    tokenized_captions = [c.lower().split() for c in captions if c]
    if len(tokenized_captions) == 0:
        return []
    bm25 = BM25Okapi(tokenized_captions)
    scores = []
    for s in summaries:
        query = s.lower().split()
        score = np.mean(bm25.get_scores(query)) if query else 0.0
        scores.append(score)
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-8:
        return [0.0 for _ in scores]
    norm_scores = [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]
    return norm_scores


# -------------------------------------------------
# 6. Hybrid
# -------------------------------------------------
def compute_hybrid_score(cosine, bm25, alpha=HYBRID_ALPHA):
    if not cosine or not bm25:
        return []
    return (alpha * np.array(cosine) + (1 - alpha) * np.array(bm25)).tolist()


# -------------------------------------------------
# 7. Evaluation
# -------------------------------------------------
def evaluate_extraction(n=N_SAMPLES, exp_type="mtp"):
    results = run_extraction(target_n=n, exp_type=exp_type)

    if not results:
        print(f"No valid results for {exp_type}. Check dataset or connection.")
        return pd.DataFrame(), {}

    captions = [r["caption"] for r in results]
    summaries = [r["summary"] for r in results]

    cosine = compute_cosine_similarity(captions, summaries)
    bm25 = compute_bm25_scores(captions, summaries)
    hybrid = compute_hybrid_score(cosine, bm25)

    df = pd.DataFrame(results)
    df["cosine_similarity"] = cosine
    df["bm25_score"] = bm25
    df["hybrid_score"] = hybrid

    avg_metrics = {
        "experiment": exp_type,
        "avg_cosine": np.mean(cosine),
        "avg_bm25": np.mean(bm25),
        "avg_hybrid": np.mean(hybrid),
    }

    print("\n---------------- Evaluation Summary ----------------")
    print(f"Processed valid samples: {len(df)} ({exp_type})")
    for k, v in avg_metrics.items():
        if k != "experiment":
            print(f"{k}: {v:.4f}")
    print("----------------------------------------------------")

    return df, avg_metrics


# -------------------------------------------------
# 8. Visualization (Histograms)
# -------------------------------------------------
def plot_histograms(df, exp_name, fig_path):
    if df.empty:
        print("Nothing to plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(df["cosine_similarity"].dropna(), bins=20, alpha=0.6, label="Cosine")
    plt.hist(df["bm25_score"].dropna(), bins=20, alpha=0.6, label="BM25")
    plt.hist(df["hybrid_score"].dropna(), bins=20, alpha=0.6, label="Hybrid")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"Memory Extraction Evaluation ({exp_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved histogram plot to: {fig_path}")


# -------------------------------------------------
# 9. Seaborn Barplot for Averages
# -------------------------------------------------
def plot_avg_scores(summary_df, save_path):
    plt.figure(figsize=(7, 5))
    melted = summary_df.melt(
        id_vars="experiment",
        value_vars=["avg_cosine", "avg_bm25", "avg_hybrid"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=melted, x="experiment", y="score", hue="metric", palette="pastel")
    plt.title("Average Scores Across Experiments")
    plt.xlabel("Experiment")
    plt.ylabel("Average Score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved average score comparison to: {save_path}")


# -------------------------------------------------
# 10. Main
# -------------------------------------------------
if __name__ == "__main__":
    import argparse
    import seaborn as sns

    parser = argparse.ArgumentParser(description="Run memory extraction evaluation")
    parser.add_argument(
        "--exp",
        default="all",
        choices=["mtp", "mtp_with_sem", "pe", "all"],
        help="Which experiment pipeline to run. Use 'all' to run all three sequentially.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help="Number of image samples to evaluate per experiment (default: 300).",
    )
    args = parser.parse_args()

    # Always compare the three main pipelines if "all" is chosen
    experiments = (
        ["mtp", "mtp_with_sem"] if args.exp == "all" else [args.exp]
    )

    base_dir = Path.cwd() / "results"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting evaluation for experiments: {', '.join(experiments)}")
    print(f"📂 Saving all results under: {base_dir}\n")

    summary_records = []

    for exp_name in experiments:
        print(f"\n==============================")
        print(f"▶ Running experiment: {exp_name.upper()}")
        print(f"==============================")

        exp_dir = base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        csv_path = exp_dir / f"memory_extraction_eval_{exp_name}.csv"
        fig_path = exp_dir / f"memory_extraction_eval_{exp_name}.png"

        df, avg_metrics = evaluate_extraction(args.samples, exp_type=exp_name)
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed results to: {csv_path}")

        if not df.empty:
            plot_histograms(df, exp_name, fig_path)
            summary_records.append(avg_metrics)

        # Clean up temporary images
        for p in df.get("image_path", []):
            try:
                os.remove(p)
            except Exception:
                pass

    # ---- Final Comparison: Summary CSV + Seaborn barplot ----
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_csv = base_dir / "summary_scores.csv"
        summary_plot = base_dir / "comparison_barplot.png"

        # Save numeric summary
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSaved summary averages to: {summary_csv}")

        # Plot comparison using seaborn
        plt.figure(figsize=(8, 6))
        melted = summary_df.melt(
            id_vars="experiment",
            value_vars=["avg_cosine", "avg_bm25", "avg_hybrid"],
            var_name="metric",
            value_name="score",
        )
        sns.barplot(data=melted, x="experiment", y="score", hue="metric", palette="pastel")
        plt.title("Comparison of Average Scores Across Experiments")
        plt.xlabel("Experiment")
        plt.ylabel("Average Score")
        plt.ylim(0, 1)
        plt.legend(title="Metric", loc="upper right")
        plt.tight_layout()
        plt.savefig(summary_plot)
        plt.close()
        print(f"Saved comparison plot to: {summary_plot}")

        # Print summary table in console
        print("\n---------------- Overall Comparison ----------------")
        print(summary_df.to_string(index=False))
        print("----------------------------------------------------")

    print("\nAll experiments completed!")
