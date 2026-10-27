from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["input_voltage_v", "output_voltage_v", "input_current_a", "output_current_a", "temperature_c"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT/"data/sample/inverter_telemetry.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    required = FEATURES + ["efficiency_pct", "performance_status"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df = df[required].dropna()
    X, yr, yc = df[FEATURES], df["efficiency_pct"].astype(float), df["performance_status"].astype(str)
    Xtr, Xte, yrtr, yrte, yctr, ycte = train_test_split(X, yr, yc, test_size=args.test_size, random_state=42, stratify=yc)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    reg.fit(Xtr, yrtr); clf.fit(Xtr, yctr)
    pr, pc = reg.predict(Xte), clf.predict(Xte)
    metrics = {
        "test_rows": int(len(Xte)),
        "regression": {"MAE": float(mean_absolute_error(yrte, pr)), "RMSE": float(np.sqrt(mean_squared_error(yrte, pr)))},
        "classification_accuracy": float(accuracy_score(ycte, pc)),
        "classification_report": classification_report(ycte, pc, output_dict=True, zero_division=0),
        "warning": "Synthetic-data metrics demonstrate code flow only; they do not establish real-world performance."
    }
    out = ROOT/"outputs"; out.mkdir(exist_ok=True)
    joblib.dump(reg, out/"efficiency_regressor.joblib")
    joblib.dump(clf, out/"status_classifier.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Models and metrics saved to {out}")

if __name__ == "__main__": main()
