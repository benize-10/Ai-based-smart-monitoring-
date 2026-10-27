from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "inverter_telemetry.csv"

def generate_data(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2026-01-01", periods=n, freq="5min")
    vin = rng.normal(380, 18, n).clip(320, 430)
    iin = rng.normal(8, 2, n).clip(1, 15)
    temp = rng.normal(43, 10, n).clip(18, 88)
    efficiency = (96.5 - np.maximum(temp - 40, 0)*0.11 -
                  np.maximum(iin - 10, 0)*0.25 + rng.normal(0, 1, n)).clip(65, 98)
    vout = rng.normal(230, 4.5, n).clip(205, 250)
    pout = vin * iin * efficiency / 100
    iout = (pout / vout).clip(0.1, 25)
    status = np.where((efficiency < 88) | (temp > 72), "degraded", "normal").astype(object)
    abnormal = (temp > 80) | (vin < 330) | (vout < 210)
    status[abnormal] = "abnormal"
    chosen = rng.choice(n, size=max(1, n//30), replace=False)
    temp[chosen] = rng.uniform(75, 90, len(chosen))
    efficiency[chosen] = rng.uniform(65, 82, len(chosen))
    status[chosen] = "abnormal"
    return pd.DataFrame({
        "timestamp": timestamp,
        "input_voltage_v": vin.round(2),
        "output_voltage_v": vout.round(2),
        "input_current_a": iin.round(3),
        "output_current_a": iout.round(3),
        "temperature_c": temp.round(2),
        "efficiency_pct": efficiency.round(2),
        "performance_status": status
    })

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    generate_data().to_csv(OUT, index=False)
    print(f"Generated illustrative dataset: {OUT}")
