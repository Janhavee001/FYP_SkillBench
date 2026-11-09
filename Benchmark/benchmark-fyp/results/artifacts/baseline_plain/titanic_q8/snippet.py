import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Compute missing statistics about each column
# ------------------------------------------------------------------
total_rows = len(df)
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / total_rows * 100).round(2)

# Print a tidy summary that a reader can use directly
print("=== Missing‑value summary ===")
print(f"Total rows: {total_rows}")
summary_df = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percent": missing_percent
})
# Show only columns that actually have missing data
summary_df = summary_df[summary_df["missing_count"] > 0]
print(summary_df)

# ------------------------------------------------------------------
# 
# 3. Visualise the missingness (optional but helpful)
# ------------------------------------------------------------------
if not summary_df.empty:
    # Ensure the output directory exists
    out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q8"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chart.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(
        x=summary_df.index,
        y="missing_percent",
        data=summary_df,
        palette="viridis"
    )
    plt.title("Percentage of Missing Values per Column")
    plt.ylabel("Missing (%)")
    plt.xlabel("Column")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()