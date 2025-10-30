import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# 1. Load CSV Results
# -------------------------------------------------
def load_results(base_dir):
    """Load all experiment CSVs under base_dir/results/<exp>/"""
    summary_records = []
    for exp_dir in Path(base_dir).glob("*"):
        if not exp_dir.is_dir():
            continue
        csv_files = list(exp_dir.glob("memory_extraction_eval_*.csv"))
        if not csv_files:
            continue

        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        avg_metrics = {
            "experiment": exp_dir.name,
            "avg_cosine": df["cosine_similarity"].mean(),
            "avg_bm25": df["bm25_score"].mean(),
            "avg_hybrid": df["hybrid_score"].mean(),
        }
        summary_records.append(avg_metrics)

    return pd.DataFrame(summary_records)


# -------------------------------------------------
# 2. Plot Comparison
# -------------------------------------------------
def plot_comparison(summary_df, save_path):
    if summary_df.empty:
        print("No valid data found for comparison.")
        return

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
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved comparison plot to: {save_path}")


# -------------------------------------------------
# 3. Main
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare memory extraction results from CSVs")
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Path to the directory containing experiment subfolders",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    summary_df = load_results(base_dir)

    if summary_df.empty:
        print("No experiment CSVs found. Check your results folder.")
    else:
        summary_csv = base_dir / "summary_scores.csv"
        summary_plot = base_dir / "comparison_barplot.png"

        summary_df.to_csv(summary_csv, index=False)
        print(f"📄 Saved summary averages to: {summary_csv}")
        print("\n---------------- Overall Comparison ----------------")
        print(summary_df.to_string(index=False))
        print("----------------------------------------------------")

        plot_comparison(summary_df, summary_plot)