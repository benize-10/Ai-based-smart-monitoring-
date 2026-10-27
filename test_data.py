from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from generate_sample_data import generate_data

def test_generated_data_schema_and_ranges():
    df = generate_data(n=50)
    assert len(df) == 50
    assert {"input_voltage_v","output_voltage_v","input_current_a","output_current_a","temperature_c","efficiency_pct","performance_status"}.issubset(df.columns)
    assert df["efficiency_pct"].between(0,100).all()
    assert set(df["performance_status"]).issubset({"normal","degraded","abnormal"})
