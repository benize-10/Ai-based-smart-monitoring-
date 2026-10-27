# Dataset guide

`sample/inverter_telemetry.csv` is synthetic demonstration data, not measured from hardware.

Required columns:
- `timestamp`: ISO timestamp
- `input_voltage_v`, `output_voltage_v`: voltage in volts
- `input_current_a`, `output_current_a`: current in amperes
- `temperature_c`: temperature in degrees Celsius
- `efficiency_pct`: target efficiency percentage
- `performance_status`: target class (`normal`, `degraded`, `abnormal`)

For real data, document source, license, sampling interval, sensor calibration, inverter model, units, test conditions, and class definitions.
