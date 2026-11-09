import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import sys
import subprocess

# ----------------------------------------------------------------------
# Ensure required third‑party packages are installed.
# This block will attempt to install any missing packages at runtime.
# ----------------------------------------------------------------------
def install_and_import(package_name, import_name=None):
    """Install a package via pip if it cannot be imported, then import it."""
    import_name = import_name or package_name
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return __import__(import_name)

# Core scientific stack
pd   = install_and_import("pandas")
np   = install_and_import("numpy")
plt  = install_and_import("matplotlib.pyplot")
sns  = install_and_import("seaborn")
stats= install_and_import("scipy.stats")
sm   = install_and_import("statsmodels.api")

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"The data file was not found at: {csv_path}")

df = pd.read_csv(csv_path, sep=';')   # the original file uses ';' as separator

# ----------------------------------------------------------------------
# 2. Compute Pearson correlation
# ----------------------------------------------------------------------
r, p_corr = stats.pearsonr(df["volatile_acidity"], df["quality"])
print(f"Pearson correlation (volatile_acidity ↔ quality): r = {r:.4f}, p‑value = {p_corr:.4e}")

# ----------------------------------------------------------------------
# 3. Simple linear regression (OLS)
# ----------------------------------------------------------------------
X = sm.add_constant(df["volatile_acidity"])   # adds intercept term
model = sm.OLS(df["quality"], X).fit()
slope = model.params["volatile_acidity"]
intercept = model.params["const"]
p_reg = model.pvalues["volatile_acidity"]
r_squared = model.rsquared

print(f"Linear regression: quality = {intercept:.4f} + ({slope:.4f})*volatile_acidity")
print(f"  slope p‑value = {p_reg:.4e}, R² = {r_squared:.4f}")

# ----------------------------------------------------------------------
# 4. Plot
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.regplot(
    x="volatile_acidity",
    y="quality",
    data=df,
    scatter_kws={"s": 20, "alpha": 0.6},
    line_kws={"color": "red"}
)
plt.title("Wine Quality vs. Volatile Acidity")
plt.xlabel("Volatile Acidity")
plt.ylabel("Quality Score")

# Ensure output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q7"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Scatter plot saved to: {out_path}")