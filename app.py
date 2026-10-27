from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))
from generate_sample_data import generate_data

FEATURES = ["input_voltage_v","output_voltage_v","input_current_a","output_current_a","temperature_c"]
st.set_page_config(page_title="Smart Inverter Monitor", page_icon="⚡", layout="wide")
st.title("⚡ AI-Based Smart Inverter Performance Monitoring")
st.caption("Educational prototype for inverter telemetry analysis, efficiency estimation, and status classification.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload inverter telemetry CSV", type=["csv"])
    use_sample = st.checkbox("Use synthetic sample data when no file is uploaded", value=True)
    st.warning("Synthetic data is illustrative only. This is not a certified protection system.")

if uploaded is not None:
    try: df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()
elif use_sample:
    df = generate_data()
else:
    st.info("Upload a CSV or enable the sample data option."); st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error("Missing required feature columns: " + ", ".join(missing)); st.stop()
for c in FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=FEATURES)
if df.empty: st.error("No valid numeric rows found."); st.stop()

a,b,c = st.columns(3)
a.metric("Records", f"{len(df):,}")
b.metric("Average temperature", f"{df.temperature_c.mean():.1f} °C")
c.metric("Average input voltage", f"{df.input_voltage_v.mean():.1f} V")

st.subheader("Telemetry trends")
fig, ax = plt.subplots(figsize=(10,4))
for col in ["input_voltage_v","output_voltage_v","temperature_c"]:
    ax.plot(df[col].reset_index(drop=True).head(300), label=col.replace("_"," "))
ax.set_xlabel("Record (first 300)"); ax.grid(True, alpha=.25); ax.legend()
st.pyplot(fig); plt.close(fig)

left,right = st.columns(2)
with left:
    st.subheader("Performance status")
    if "performance_status" in df:
        st.bar_chart(df["performance_status"].astype(str).value_counts())
    else: st.info("Add labelled performance_status values to view class distribution.")
with right:
    st.subheader("Efficiency")
    if "efficiency_pct" in df:
        eff = pd.to_numeric(df["efficiency_pct"], errors="coerce").dropna()
        if len(eff):
            st.line_chart(eff.reset_index(drop=True).head(500))
            st.metric("Mean recorded efficiency", f"{eff.mean():.2f}%")
    else: st.info("No efficiency_pct target column was found.")

st.subheader("Model predictions")
regfile, clffile = ROOT/"outputs/efficiency_regressor.joblib", ROOT/"outputs/status_classifier.joblib"
if regfile.exists() and clffile.exists():
    import joblib
    reg, clf = joblib.load(regfile), joblib.load(clffile)
    result = df[FEATURES].copy()
    result["predicted_efficiency_pct"] = reg.predict(df[FEATURES])
    result["predicted_status"] = clf.predict(df[FEATURES])
    st.dataframe(result.head(100), use_container_width=True)
else:
    st.info("No trained models found. Run `python src/train.py` in the project terminal.")
st.subheader("Data preview")
st.dataframe(df.head(30), use_container_width=True)
st.download_button("Download loaded data as CSV", df.to_csv(index=False).encode(), "inverter_data.csv", "text/csv")
