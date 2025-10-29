import os
import random
import tempfile
import jaclang
import requests
import urllib3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
SAVE_PATH = "memory_extraction_eval.csv"

# -------------------------------------------------
# 1. Dataset
# -------------------------------------------------
def load_laion_subset(n=300):
    ds = load_dataset("laion/laion400m", split="train", streaming=True)
    subset = []
    for i, example in enumerate(ds):
        if i >= n:
            break
        subset.append(example)
    return subset

# -------------------------------------------------
# 2. Image Download
# -------------------------------------------------
def safe_download_image(url: str, timeout: int = 5) -> str | None:
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
# 3. Run Extraction
# -------------------------------------------------
def run_extraction(subset, exp_type="mtp"):
    """
    Runs image extraction pipeline for given subset.
    - For 'pe' experiments: disables byLLM by setting use_byllm=False.
    - For 'mtp' and 'mtp_with_sem': runs with default settings (use_byllm=True).
    """
    results = []
    for item in tqdm(subset, desc=f"Extracting memories ({exp_type})"):
        try:
            img_path = safe_download_image(item["url"])
            if not img_path:
                continue

            img = ByImage(img_path)

            # Determine byLLM usage based on experiment type
            use_byllm = False if exp_type == "pe" else True

            # Call Jac memory extractor
            res = extract_memory_details(img, use_byllm=use_byllm)

            # Skip invalid or empty outputs
            if not res or not getattr(res, "summary", "").strip():
                continue

            results.append({
                "caption": item.get("caption", ""),
                "summary": res.summary,
                "who": res.who,
                "what": res.what,
                "when": res.when,
                "location_type": res.location_type,
                "follow_up_questions": res.follow_up_questions,
                "image_path": img_path,
                "exp_type": exp_type,
                "use_byllm": use_byllm
            })
        except Exception as e:
            print(f"Error processing ({exp_type}):", e)

    print(f"✅ Successfully extracted {len(results)} valid samples for {exp_type}.")
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
    subset = load_laion_subset(n)
    results = run_extraction(subset, exp_type=exp_type)

    if not results:
        print(f"❌ No valid results for {exp_type}. Check dataset or connection.")
        return pd.DataFrame()

    captions = [r["caption"] for r in results]
    summaries = [r["summary"] for r in results]

    cosine = compute_cosine_similarity(captions, summaries)
    bm25 = compute_bm25_scores(captions, summaries)
    hybrid = compute_hybrid_score(cosine, bm25)

    df = pd.DataFrame(results)
    df["cosine_similarity"] = cosine
    df["bm25_score"] = bm25
    df["hybrid_score"] = hybrid

    print("\n---------------- Evaluation Summary ----------------")
    print(f"Processed samples: {len(df)} / {n} ({exp_type})")
    if len(cosine):
        print(f"Average Cosine Similarity: {np.mean(cosine):.4f}")
    if len(bm25):
        print(f"Average BM25 Score: {np.mean(bm25):.4f}")
    if len(hybrid):
        print(f"Average Hybrid Score: {np.mean(hybrid):.4f}")
    print("----------------------------------------------------")

    return df

# -------------------------------------------------
# 8. Visualization
# -------------------------------------------------
def plot_scores(df):
    if df.empty:
        print("⚠️ Nothing to plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(df["cosine_similarity"].dropna(), bins=20, alpha=0.6, label="Cosine")
    plt.hist(df["bm25_score"].dropna(), bins=20, alpha=0.6, label="BM25")
    plt.hist(df["hybrid_score"].dropna(), bins=20, alpha=0.6, label="Hybrid")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Memory Extraction Evaluation (Cosine vs BM25 vs Hybrid)")
    plt.legend()
    plt.show()


# -------------------------------------------------
# 9. Main
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run memory extraction evaluation")
    parser.add_argument(
        "--exp",
        default="mtp",
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

    # Choose which experiments to run
    experiments = (
        ["mtp", "mtp_with_sem", "pe"] if args.exp == "all" else [args.exp]
    )

    base_dir = Path.cwd() / "results"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting evaluation for experiments: {', '.join(experiments)}")
    print(f"📂 Saving all results under: {base_dir}\n")

    for exp_name in experiments:
        print(f"\n==============================")
        print(f"▶ Running experiment: {exp_name.upper()}")
        print(f"==============================")

        # Create experiment-specific subfolder
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        csv_path = exp_dir / f"memory_extraction_eval_{exp_name}.csv"
        fig_path = exp_dir / f"memory_extraction_eval_{exp_name}.png"

        # Run full pipeline
        df = evaluate_extraction(args.samples, exp_type=exp_name)
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved results to: {csv_path}")

        # Visualization
        if not df.empty:
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
            print(f"🖼️ Saved plot to: {fig_path}")

        # Clean up temporary images
        for p in df.get("image_path", []):
            try:
                os.remove(p)
            except Exception:
                pass

    print("\n✅ All experiments completed successfully! Results are in ./rsults/")