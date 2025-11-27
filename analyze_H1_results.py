import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------- Setup --------------------- #
base = Path("~/Desktop/AQF_Backups/extracted/experiments/artifacts/reports").expanduser()
per_station = base / "per_station_H1.csv"
preds = base / "preds_H1.csv"

df = pd.read_csv(per_station)
print(f"Loaded {len(df)} stations from {per_station.name}")
print(df.head())

# --------------------- Summary Stats --------------------- #
metrics = ["rmse", "mae", "smape", "r2"]
print("\nOverall Summary:")
print(df[metrics].describe().T)

# --------------------- Top/Bottom Performers --------------------- #
top = df.sort_values("rmse").head(10)
bottom = df.sort_values("rmse", ascending=False).head(10)

print("\nTop 10 performing stations (lowest RMSE):")
print(top[["station_id", "rmse", "mae", "smape", "r2"]])

print("\nBottom 10 performing stations (highest RMSE):")
print(bottom[["station_id", "rmse", "mae", "smape", "r2"]])

# --------------------- Distribution Plot --------------------- #
plt.figure(figsize=(8,4))
sns.histplot(df["rmse"], bins=40, kde=True, color="steelblue")
plt.title("RMSE Distribution Across Stations (H=1)")
plt.xlabel("RMSE")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --------------------- Scatter: RMSE vs R2 --------------------- #
plt.figure(figsize=(6,6))
sns.scatterplot(data=df, x="rmse", y="r2", s=40, alpha=0.7)
plt.title("RMSE vs R² per Station (H=1)")
plt.xlabel("RMSE"); plt.ylabel("R²")
plt.tight_layout()
plt.show()

# --------------------- Optional: Error trend from preds --------------------- #
pred_path = Path("~/Desktop/AQF_Backups/extracted/experiments/artifacts/reports/preds_H1.csv").expanduser()
if pred_path.exists():
    preds_df = pd.read_csv(pred_path)
    sample_station = int(df.sort_values("rmse").iloc[0]["station_id"])
    sub = preds_df.query("station_id == @sample_station").copy()
    if {"y_true","y_pred"}.issubset(sub.columns):
        plt.figure(figsize=(10,4))
        plt.plot(sub["y_true"].values[:200], label="True", linewidth=1.5)
        plt.plot(sub["y_pred"].values[:200], label="Predicted", linewidth=1.5)
        plt.title(f"Sample Forecast – Station {sample_station} (H=1)")
        plt.xlabel("Time Index"); plt.ylabel("PM2.5 Concentration")
        plt.legend(); plt.tight_layout(); plt.show()