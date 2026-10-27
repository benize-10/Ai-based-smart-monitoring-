from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["input_voltage_v", "output_voltage_v", "input_current_a", "output_current_a", "temperature_c"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT/"data/sample/inverter_telemetry.csv"))
    args = parser.parse_args()
    out = ROOT/"outputs"
    regfile, clffile = out/"efficiency_regressor.joblib", out/"status_classifier.joblib"
    if not regfile.exists() or not clffile.exists():
        raise FileNotFoundError("Models not found. Run python src/train.py first.")
    df = pd.read_csv(args.data).dropna(subset=FEATURES+["efficiency_pct","performance_status"])
    X, yr, yc = df[FEATURES], df["efficiency_pct"], df["performance_status"].astype(str)
    reg, clf = joblib.load(regfile), joblib.load(clffile)
    pr, pc = reg.predict(X), clf.predict(X)
    labels = sorted(set(yc) | set(pc))
    result = {
        "regression": {"MAE": float(mean_absolute_error(yr,pr)), "RMSE": float(np.sqrt(mean_squared_error(yr,pr)))},
        "classification_accuracy": float(accuracy_score(yc,pc)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(yc,pc,labels=labels).tolist(),
        "classification_report": classification_report(yc,pc,output_dict=True,zero_division=0),
        "warning": "This script evaluates all supplied rows. For final reporting, evaluate on a separate held-out test set."
    }
    (out/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k=="regression" or k=="classification_accuracy"},indent=2))
    print("Full results saved to outputs/evaluation.json")

if __name__ == "__main__": main()
